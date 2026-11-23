"""Train a bounded experiment, or resume one up to a new total step count."""

import argparse
from dataclasses import asdict
import fcntl
import math
from pathlib import Path
import time

import torch

from .data import Corpus, DEFAULT_DIR
from .diagnostics import preflight
from .engine import evaluate, load_checkpoint, make_optimizer, restore_training
from .engine import save_checkpoint, seed_all, update
from .model import GPT, GPTConfig
from .runtime import choose_device, write_json


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("fit", "resume"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, default=Path("runs/alice"))
        p.add_argument("--data", type=Path, default=DEFAULT_DIR)
        p.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
        p.add_argument("--steps", type=int, default=600, help="target total optimizer updates")
        p.add_argument("--seconds", type=float, default=600, help="wall budget for this training loop")
        if name == "fit":
            p.add_argument("--seed", type=int, default=2026)
            p.add_argument("--lr", type=float, default=0.001)
            p.add_argument("--batch-size", type=int, default=8)
            p.add_argument("--context", type=int, default=128)
            p.add_argument("--width", type=int, default=160)
            p.add_argument("--heads", type=int, default=4)
            p.add_argument("--layers", type=int, default=4)
            p.add_argument("--dropout", type=float, default=0.)
    return parser


def run(args):
    start = time.monotonic()
    if args.steps < 1 or not math.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError("--steps and --seconds must be positive and finite")
    device = choose_device(args.device)
    corpus = Corpus.load(args.data)
    checkpoint = args.out / "last.pt"
    sampler = torch.Generator()
    step = 0
    if args.command == "resume":
        model, payload = load_checkpoint(checkpoint, "cpu")
        settings = payload["settings"]
        config = model.config
        step = payload["step"]
        if step >= args.steps:
            raise ValueError(f"Checkpoint is at step {step}. Increase the total --steps to resume.")
        if payload["device"] != device.type or corpus.fingerprint != payload["corpus_fingerprint"]:
            raise ValueError("Resume requires the checkpoint's original corpus and device.")
    else:
        if checkpoint.exists():
            raise FileExistsError(f"{checkpoint} already exists. Use resume or a different --out.")
        config = GPTConfig(context_length=args.context, n_embd=args.width, n_head=args.heads,
                           n_layer=args.layers, dropout=args.dropout)
        if args.batch_size < 1 or not math.isfinite(args.lr) or args.lr <= 0:
            raise ValueError("batch size and learning rate must be positive")
        settings = {"lr": args.lr, "seed": args.seed, "batch_size": args.batch_size,
                    "history": [], "training_seconds": 0.}
    batch_size, lr, seed = settings["batch_size"], settings["lr"], settings["seed"]
    probe = preflight(corpus, config, batch_size, lr, seed, device)
    write_json(args.out / "preflight.json", probe)
    if not probe["passed"]:
        raise RuntimeError("Preflight missed its memory or finite-loss threshold. See preflight.json.")
    eta = (args.steps - step) * probe["warm_median_seconds"]
    print(f"Preflight passed on {device}. Estimated update time for this budget: {eta:.1f} s.", flush=True)

    if args.command == "resume":
        model.to(device)
        optimizer = restore_training(payload, model, sampler, corpus.fingerprint)
        del payload
    else:
        # Disposable preflight weights never enter the experiment.
        seed_all(seed, device)
        sampler.manual_seed(seed)
        model = GPT(config).to(device)
        optimizer = make_optimizer(model, lr)
    begin_fit = time.monotonic()
    prior_seconds = settings["training_seconds"]
    history = settings["history"]
    if not history:
        history.append({"step": 0, "validation": evaluate(model, corpus.splits["val"], batch_size)})
    reason = "step_limit"
    last_loss = None

    def record():
        elapsed = time.monotonic() - begin_fit
        validation = evaluate(model, corpus.splits["val"], batch_size)
        row = {"step": step, "batch_loss_before_update": last_loss, "validation": validation}
        if history[-1]["step"] == step:
            history[-1].update(row)
        else:
            history.append(row)
        settings["training_seconds"] = prior_seconds + time.monotonic() - begin_fit
        save_checkpoint(checkpoint, model, optimizer, step, sampler, corpus.fingerprint, settings)
        write_json(args.out / "history.json", {"history": history})
        print(f"Step {step:4d} | validation {validation['loss']:.4f} | {elapsed:.1f} s", flush=True)

    while step < args.steps:
        if time.monotonic() - begin_fit >= args.seconds:
            reason = "time_limit"
            break
        x, y = corpus.get_batch("train", batch_size, config.context_length, sampler)
        last_loss = update(model, optimizer, x, y)
        step += 1
        if step % 100 == 0 or step == args.steps:
            record()
    # Keep partial progress at the wall-clock limit, including a zero-update run.
    if history[-1]["step"] != step or not checkpoint.exists():
        record()
    report = {
        "device": str(device), "torch_version": str(torch.__version__), "dtype": "float32",
        "config": asdict(config), "parameters": sum(p.numel() for p in model.parameters()),
        "corpus_fingerprint": corpus.fingerprint, "seed": seed, "lr": lr,
        "batch_size": batch_size, "total_steps": step,
        "training_targets_seen": step * batch_size * config.context_length,
        "requested_total_steps": args.steps, "session_time_budget_seconds": args.seconds,
        "stop_reason": reason, "session_wall_seconds": time.monotonic() - start,
        "total_training_loop_seconds": settings["training_seconds"],
    }
    write_json(args.out / "run.json", report)
    print(f"Saved {checkpoint} ({reason}).", flush=True)


def main():
    args = make_parser().parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # macOS/Linux automatically release this lock if the process exits.
    with (args.out / ".train.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another training process is writing to this run directory.")
        run(args)


if __name__ == "__main__":
    main()
