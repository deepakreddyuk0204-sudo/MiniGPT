"""Fetch, resume and verify the official quantized Qwen model using macOS curl."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
from hashlib import file_digest
import json
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "minigpt/qwen_manifest.json"
BLOCK_SIZE = 8 * 1024**2
CURL = ["curl", "--fail", "--silent", "--show-error", "--location", "--proto", "=https",
        "--proto-redir", "=https", "--connect-timeout", "20", "--speed-limit", "4096",
        "--speed-time", "45"]


def verify(path: Path, manifest: dict) -> None:
    if path.stat().st_size == manifest["size_bytes"]:
        with path.open("rb") as stream:
            if file_digest(stream, "sha256").hexdigest() == manifest["sha256"]:
                return
    raise ValueError(f"Size or SHA-256 mismatch for {path}. Move it aside and rerun.")


def chunks_for(directory: Path, filename: str, offset: int, size: int):
    """Partition the missing suffix, reusing complete ranges left by earlier runs."""
    prefix = filename + ".range."
    saved = {}
    for path in directory.glob(prefix + "*"):
        suffix = path.name[len(prefix):]
        if suffix.isdecimal():
            start = int(suffix)
            stop = start + path.stat().st_size
            if start < stop <= size:
                saved[start] = (stop, path)
    chunks = []
    # A partial merge may have stopped partway through the first saved chunk.
    for start, (stop, path) in sorted(saved.items()):
        if stop <= offset:
            continue
        while offset < start:
            end = min(start, offset + BLOCK_SIZE)
            chunks.append((offset, end, directory / f"{prefix}{offset}", False))
            offset = end
        chunks.append((start, stop, path, True))
        offset = stop
    while offset < size:
        end = min(size, offset + BLOCK_SIZE)
        chunks.append((offset, end, directory / f"{prefix}{offset}", False))
        offset = end
    return chunks


def fetch_range(url: str, size: int, chunk) -> None:
    start, stop, destination, reused = chunk
    if reused:
        return
    expected = f"bytes {start}-{stop - 1}/{size}"
    # Only a fully received and correctly addressed response becomes a reusable range.
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        headers, body = Path(scratch) / "headers", Path(scratch) / "body"
        for attempt in range(3):
            request_url = f"{url}?range={start}-{stop-1}&t={time.time_ns()}"
            result = subprocess.run(
                CURL + ["--range", f"{start}-{stop-1}", "--dump-header", str(headers),
                        "--max-filesize", str(stop-start), "--output", str(body),
                        "--write-out", "%{http_code}", request_url],
                stdout=subprocess.PIPE, text=True,
            )
            if result.returncode in (-signal.SIGINT, 128 + signal.SIGINT):
                raise KeyboardInterrupt
            if result.returncode == 0 and result.stdout == "206":
                matches = re.findall(r"^content-range:\s*([^\r\n]+)", headers.read_text(),
                                     flags=re.IGNORECASE | re.MULTILINE)
                if matches and matches[-1].strip() == expected and body.stat().st_size == stop - start:
                    body.replace(destination)
                    return
            time.sleep(attempt + 1)
    raise RuntimeError(f"The server did not return the requested range {expected}.")


def acquire(directory: Path, manifest: dict, workers: int) -> Path:
    filename = manifest["filename"]
    destination = directory / filename
    partial = directory / (filename + ".part")
    if destination.exists():
        verify(destination, manifest)
        return destination
    size = manifest["size_bytes"]
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        raise ValueError(f"Oversized partial download at {partial}. Move it aside and rerun.")
    chunks = chunks_for(directory, filename, offset, size)
    needed = sum(stop - start for start, stop, _, saved in chunks if not saved)
    # While assembling, at most one chunk is present twice.
    if needed + BLOCK_SIZE > shutil.disk_usage(directory).free and chunks:
        raise OSError(f"Allow {needed + BLOCK_SIZE:,} bytes for the remaining download.")
    url = f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}/{filename}"
    if chunks:
        print(f"Verified ranges requested for {needed:,} of {size:,} bytes.", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {chunk[0]: pool.submit(fetch_range, url, size, chunk) for chunk in chunks}
        try:
            with partial.open("ab") as stream:
                for start, stop, path, _ in chunks:
                    pending[start].result()
                    with path.open("rb") as src:
                        src.seek(stream.tell() - start)
                        shutil.copyfileobj(src, stream, length=1024**2)
                    stream.flush()
                    if stream.tell() != stop:
                        raise OSError("Misaligned download range")
                    path.unlink()
                    print(f"Received {100 * stop / size:.1f}%", flush=True)
        except (RuntimeError, KeyboardInterrupt) as error:
            for future in pending.values():
                future.cancel()
            if isinstance(error, KeyboardInterrupt):
                raise
            print("Range transfer unavailable. Resuming the contiguous download.", flush=True)
            subprocess.run(CURL + ["--retry", "3", "--continue-at", "-", "--output",
                                  str(partial), url + f"?t={time.time_ns()}"], check=True)
    verify(partial, manifest)
    partial.replace(destination)
    for path in directory.glob(filename + ".range.*"):
        if path.name.split(".")[-1].isdecimal():
            path.unlink()
    return destination


def download(directory: Path, manifest: dict, workers: int = 6) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".download.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another model download owns this directory.") from None
        return acquire(directory, manifest, workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "models")
    parser.add_argument("--workers", type=int, default=6, choices=range(1, 17))
    args = parser.parse_args()
    model = download(args.directory, json.loads(MANIFEST.read_text(encoding="utf-8")), args.workers)
    print(f"Verified model: {model}")
