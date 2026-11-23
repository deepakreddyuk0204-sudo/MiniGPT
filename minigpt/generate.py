"""Generate a continuation from local MiniGPT weights."""

import argparse
from pathlib import Path

import torch

from .engine import load_checkpoint
from .generation import generate
from .runtime import choose_device
from .tokenizer import ByteTokenizer


def continuation(model, prompt, max_new_tokens, temperature, top_k, seed):
    if not prompt:
        raise ValueError("Enter at least one character to start the continuation.")
    device = model.token_embedding.weight.device
    ids = torch.tensor([ByteTokenizer.encode(prompt)], dtype=torch.long, device=device)
    full = generate(model, ids, max_new_tokens, temperature, top_k, seed)
    return ByteTokenizer.decode(full[0, ids.size(1):].tolist(), errors="replace")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/alice/last.pt"))
    parser.add_argument("--device", choices=["cpu", "auto", "mps"], default="cpu")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    model, _ = load_checkpoint(args.checkpoint, choose_device(args.device))
    print(args.prompt + continuation(model, args.prompt, args.tokens, args.temperature,
                                     args.top_k, args.seed))


if __name__ == "__main__":
    main()
