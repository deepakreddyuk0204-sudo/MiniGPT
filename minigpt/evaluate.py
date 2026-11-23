"""Evaluate a frozen checkpoint on every chapter transition and generate fixed samples."""

import argparse
from hashlib import file_digest
import math
from pathlib import Path
import time

import torch

from .data import Corpus, DEFAULT_DIR
from .engine import evaluate, load_checkpoint
from .generate import continuation
from .runtime import choose_device, write_json
from .tokenizer import ByteTokenizer


def unigram_scores(corpus):
    # Add one count per byte so unseen held-out bytes also have finite log loss.
    counts = torch.bincount(torch.cat(corpus.splits["train"]), minlength=ByteTokenizer.vocab_size)
    smoothed = counts.double() + 1
    nll = -(smoothed / smoothed.sum()).log()
    result = {}
    for name, documents in corpus.splits.items():
        targets = torch.cat([doc[1:] for doc in documents])
        loss = nll[targets].mean().item()
        result[name] = {"loss": loss, "perplexity": math.exp(loss), "targets": len(targets)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/alice/last.pt"))
    parser.add_argument("--data", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation.json"))
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    args = parser.parse_args()
    start = time.monotonic()
    corpus = Corpus.load(args.data)
    device = choose_device(args.device)
    # Read a single snapshot even if a trainer concurrently replaces the checkpoint.
    with args.checkpoint.open("rb") as f:
        digest = file_digest(f, "sha256").hexdigest()
        f.seek(0)
        model, payload = load_checkpoint(f, device)
    if corpus.fingerprint != payload["corpus_fingerprint"]:
        raise ValueError("Evaluation corpus does not match the checkpoint.")
    scores = {name: evaluate(model, documents, payload["settings"]["batch_size"])
              for name, documents in corpus.splits.items()}
    # CPU sampling makes it easy to reproduce the samples with the default generate CLI.
    model.cpu()
    sampling = {"max_new_tokens": 120, "temperature": 0.8, "top_k": 20, "seed": 42}
    samples = [
        {"prompt": prompt, "continuation": continuation(model, prompt, **sampling)}
        for prompt in ("Alice ", "The Rabbit ")
    ]
    report = {
        "checkpoint_sha256": digest, "corpus_fingerprint": corpus.fingerprint,
        "step": payload["step"], "evaluation_device": str(device), "scores": scores,
        "train_unigram_baseline": unigram_scores(corpus), "sampling_device": "cpu",
        "sampling": sampling, "samples": samples, "runtime_seconds": time.monotonic() - start,
    }
    write_json(args.output, report)
    for split, metrics in scores.items():
        print(f"{split}: loss {metrics['loss']:.4f}, perplexity {metrics['perplexity']:.3f}, "
              f"{metrics['targets']:,} targets")
    print(f"Saved evaluation and unedited samples to {args.output}")


if __name__ == "__main__":
    main()
