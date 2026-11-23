"""Autoregressive decoding with a rolling context and an independent sampling seed."""

import math

import torch

from .model import GPT


@torch.inference_mode()
def generate(
    model: GPT,
    ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int = 0,
) -> torch.Tensor:
    """Return the entire prompt followed by max_new_tokens sampled IDs per row."""
    if ids.ndim != 2 or ids.size(0) == 0 or ids.size(1) == 0:
        raise ValueError("the prompt must be a non-empty [batch, time] tensor")
    if ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("prompt IDs must be integers (int32 or int64)")
    if type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a non-negative integer")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    vocab_size = model.config.vocab_size
    if top_k is not None and (type(top_k) is not int or not 1 <= top_k <= vocab_size):
        raise ValueError(f"top_k must be an integer in [1, {vocab_size}] or None")
    if torch.any((ids < 0) | (ids >= vocab_size)).item():
        raise ValueError("a prompt ID lies outside the model vocabulary")
    if ids.device != model.token_embedding.weight.device:
        raise ValueError("prompt and model must be on the same device")

    rng = torch.Generator(device="cpu").manual_seed(seed)
    training = model.training
    model.eval()
    try:
        result = ids
        for _ in range(max_new_tokens):
            window = result[:, -model.config.context_length :]
            logits = model(window)[:, -1, :]
            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                # Only the small final vocabulary vector goes to the CPU sampler.
                # Transfer before casting; MPS has no float64 kernel.
                scores = logits.cpu().double()
                if top_k is not None:
                    scores, candidates = scores.topk(top_k, dim=-1)
                scores = (scores - scores.amax(dim=-1, keepdim=True)) / temperature
                draw = torch.multinomial(scores.softmax(dim=-1), 1, generator=rng)
                next_id = draw if top_k is None else candidates.gather(1, draw)
            result = torch.cat((result, next_id.to(ids)), dim=1)
        return result
    finally:
        model.train(training)
