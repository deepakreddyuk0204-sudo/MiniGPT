"""Exercise one update, checkpoint reload, and memorization on genuine training text."""

import argparse
from dataclasses import asdict
import gc
import math
from pathlib import Path
import resource
import statistics
import sys
import tempfile
import time

import psutil
import torch

from .data import Corpus, DEFAULT_DIR
from .engine import load_checkpoint, make_optimizer, save_checkpoint, seed_all, update
from .generation import generate
from .model import GPT, GPTConfig, next_token_loss
from .runtime import choose_device, write_json


def memory_sample(device):
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    sample = {
        "system_available_bytes": psutil.virtual_memory().available,
        "process_rss_bytes": psutil.Process().memory_info().rss,
        "process_peak_rss_bytes": usage if sys.platform == "darwin" else usage * 1024,
    }
    if device.type == "mps":
        torch.mps.synchronize()
        sample.update({
            "mps_tensor_bytes": torch.mps.current_allocated_memory(),
            "mps_driver_bytes": torch.mps.driver_allocated_memory(),
            "mps_recommended_bytes": torch.mps.recommended_max_memory(),
        })
    return sample


def preflight(corpus, config, batch_size, lr, seed, device):
    """Run four disposable AdamW updates at the intended shape before fitting."""
    seed_all(seed, device)
    model = GPT(config).to(device)
    optimizer = make_optimizer(model, lr)
    sampler = torch.Generator().manual_seed(seed)
    samples = [memory_sample(device)]
    seconds = []
    losses = []
    for _ in range(4):
        begin = time.monotonic()
        x, y = corpus.get_batch("train", batch_size, config.context_length, sampler)
        losses.append(update(model, optimizer, x, y))
        samples.append(memory_sample(device))
        seconds.append(time.monotonic() - begin)
    # RSS and Metal counters can overlap in unified memory. Judge each separately.
    enough_system = min(s["system_available_bytes"] for s in samples) > 512 * 2**20
    enough_gpu = device.type != "mps" or all(
        s["mps_driver_bytes"] < 0.75 * s["mps_recommended_bytes"] for s in samples
    )
    report = {
        "device": str(device), "config": asdict(config), "batch_size": batch_size,
        "passed": enough_system and enough_gpu and all(map(math.isfinite, losses)),
        "step_seconds": seconds, "warm_median_seconds": statistics.median(seconds[1:]),
        "losses": losses, "samples": samples,
    }
    del model, optimizer
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return report


def run_checks(corpus, device):
    seed = 17
    seed_all(seed, device)
    sampler = torch.Generator().manual_seed(seed)
    config = GPTConfig(context_length=32, n_layer=1, n_head=2, n_embd=48)
    model = GPT(config).to(device)
    lr = 0.005
    optimizer = make_optimizer(model, lr)
    x, y = corpus.get_batch("train", 2, config.context_length, sampler)
    x, y = x.to(device), y.to(device)
    old_weight = model.token_embedding.weight.detach().clone()
    initial = update(model, optimizer, x, y)
    if torch.equal(model.token_embedding.weight, old_weight):
        raise RuntimeError("Smoke check failed: the optimizer did not change weights")
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "smoke.pt"
        save_checkpoint(path, model, optimizer, 1, sampler, corpus.fingerprint, {"lr": lr})
        restored, _ = load_checkpoint(path, device)
        with torch.inference_mode():
            torch.testing.assert_close(restored(x), model(x))
        sampled = generate(restored, x[:1], 4, seed=seed)
        assert sampled.shape == (1, config.context_length + 4)
    # Resample nothing: exactly the same 64 next-byte targets on every update.
    history = [{"step": 0, "loss": initial}]
    for step in range(2, 121):
        update(model, optimizer, x, y)
        if step % 10 == 0:
            with torch.inference_mode():
                loss = next_token_loss(model(x), y).item()
            history.append({"step": step, "loss": loss})
            if loss < 0.10:
                break
    passed = history[-1]["loss"] < 0.10
    return {
        "device": str(device), "seed": seed, "config": asdict(config),
        "smoke": {"weight_changed": True, "reload_logits_match": True, "generated_bytes": 4},
        "overfit": {"targets": y.numel(), "lr": lr, "loss_threshold": 0.10,
                    "max_steps": 120, "history": history, "passed": passed},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--output", type=Path, default=Path("runs/checks.json"))
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    args = parser.parse_args()
    start = time.monotonic()
    report = run_checks(Corpus.load(args.data), choose_device(args.device))
    report["runtime_seconds"] = time.monotonic() - start
    write_json(args.output, report)
    overfit = report["overfit"]
    print(f"Smoke passed. Fixed-batch losses: {overfit['history']}. Saved {args.output}")
    if not overfit["passed"]:
        raise SystemExit("Fixed-batch memorization missed its loss threshold")


if __name__ == "__main__":
    main()
