"""Local runtime selection and human-readable experiment records."""

import json
import os
from pathlib import Path
import tempfile

import torch


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable on this host. Choose --device cpu.")
    return torch.device(name)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as f:
        temporary = Path(f.name)
        try:
            json.dump(obj, f, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.replace(f.name, path)
        finally:
            temporary.unlink(missing_ok=True)
