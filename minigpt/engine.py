"""Training steps, chapter-local evaluation, and resumable local checkpoints."""

from collections.abc import Sequence
from dataclasses import asdict
import math
from pathlib import Path
import random
import tempfile
from typing import BinaryIO

import torch
from torch.nn.utils.rnn import pad_sequence

from .model import GPT, GPTConfig, next_token_loss


def seed_all(seed: int, device: str | torch.device) -> None:
    """Seed initialization and dropout. The data sampler has its own generator."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.device(device).type == "mps":
        torch.mps.manual_seed(seed)


def make_optimizer(model: GPT, lr: float) -> torch.optim.AdamW:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)


def update(model: GPT, optimizer: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor) -> float:
    """Run one AdamW step and return the batch loss before the weight update."""
    model.train()
    device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    logits = model(x.to(device))
    loss = next_token_loss(logits, y.to(device))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return loss.item()


@torch.inference_mode()
def evaluate(
    model: GPT, documents: Sequence[torch.Tensor | Sequence[int]], batch_size: int
) -> dict[str, float | int]:
    """Score every within-document transition once with context reset per chunk."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = next(model.parameters()).device
    context = model.config.context_length
    chunks = []
    for document in documents:
        ids = torch.as_tensor(document, dtype=torch.long)
        if ids.ndim != 1:
            raise ValueError("each document must be a one-dimensional token sequence")
        for start in range(0, len(ids) - 1, context):
            chunks.append(ids[start : start + context + 1])
    if not chunks:
        raise ValueError("evaluation requires at least one within-document transition")

    training = model.training
    model.eval()
    total_nll = 0.0
    total_targets = 0
    try:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            x = pad_sequence([c[:-1] for c in batch], batch_first=True).to(device)
            # Right padding cannot affect earlier logits. -100 is ignored by CE.
            y = pad_sequence([c[1:] for c in batch], batch_first=True, padding_value=-100).to(device)
            count = sum(len(c) - 1 for c in batch)
            total_nll += next_token_loss(model(x), y).item() * count
            total_targets += count
    finally:
        model.train(training)
    mean_nll = total_nll / total_targets
    return {"loss": mean_nll, "perplexity": math.exp(mean_nll), "targets": total_targets}


def save_checkpoint(
    path: str | Path, model: GPT, optimizer: torch.optim.AdamW, step: int,
    sampler: torch.Generator, corpus_fingerprint: str, settings: dict,
) -> None:
    """Atomically replace a checkpoint after writing the complete training state."""
    path = Path(path)
    device = next(model.parameters()).device.type
    payload = {
        "version": 1,
        "config": asdict(model.config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "corpus_fingerprint": corpus_fingerprint,
        "settings": settings,
        "device": device,
        "python_rng": random.getstate(),
        "cpu_rng": torch.get_rng_state(),
        "mps_rng": torch.mps.get_rng_state() if device == "mps" else None,
        "sampler_rng": sampler.get_state(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
    try:
        with temporary:
            torch.save(payload, temporary)
        Path(temporary.name).replace(path)
    finally:
        Path(temporary.name).unlink(missing_ok=True)


def load_checkpoint(path: str | Path | BinaryIO, device: str | torch.device) -> tuple[GPT, dict]:
    """Load for inference on either device. Keep RNG and optimizer payload on CPU."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("version") != 1:
        raise ValueError("unsupported checkpoint version")
    model = GPT(GPTConfig(**payload["config"]))
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), payload


def restore_training(
    payload: dict, model: GPT, sampler: torch.Generator, corpus_fingerprint: str
) -> torch.optim.AdamW:
    """Rebuild AdamW and restore the training RNGs after model construction."""
    if payload["corpus_fingerprint"] != corpus_fingerprint:
        raise ValueError("checkpoint corpus fingerprint does not match the prepared corpus")
    device = next(model.parameters()).device.type
    if device != payload["device"]:
        raise ValueError(f"resume on the original device ({payload['device']}) to restore its RNG")
    optimizer = make_optimizer(model, payload["settings"]["lr"])
    optimizer.load_state_dict(payload["optimizer"])
    sampler.set_state(payload["sampler_rng"])
    random.setstate(payload["python_rng"])
    torch.set_rng_state(payload["cpu_rng"])
    if device == "mps":
        torch.mps.set_rng_state(payload["mps_rng"])
    model.train()
    return optimizer
