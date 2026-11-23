"""Install the pinned native llama.cpp release into this checkout on Apple silicon."""

import fcntl
from hashlib import file_digest
from pathlib import Path
import platform
import subprocess
import tarfile
import tempfile

# GitHub release asset digest for the build at commit 4577932fab595eaa8b27ad71734d5de1f3c07d8c.
VERSION = "b12503"
ASSET = f"llama-{VERSION}-bin-macos-arm64.tar.gz"
URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{VERSION}/{ASSET}"
SHA256 = "e88d7e91198d0de6ceb3d33711ac01ee0d5b519cd9ba4e81854e3bba84f22214"
TOOLS = Path(__file__).resolve().parents[1] / ".tools" / "llama.cpp"


def install() -> Path:
    if (platform.system(), platform.machine()) != ("Darwin", "arm64"):
        raise SystemExit("This runtime build requires native arm64 Python on macOS.")
    TOOLS.mkdir(parents=True, exist_ok=True)
    target = TOOLS / VERSION
    with (TOOLS / ".install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            with tempfile.TemporaryDirectory(dir=TOOLS) as scratch:
                archive = Path(scratch) / ASSET
                subprocess.run(
                    ["curl", "--fail", "--location", "--silent", "--show-error",
                     "--retry", "3", "--output", str(archive), URL], check=True,
                )
                with archive.open("rb") as stream:
                    digest = file_digest(stream, "sha256").hexdigest()
                if digest != SHA256:
                    raise ValueError(f"Unexpected runtime archive SHA-256: {digest}")
                with tarfile.open(archive, "r:gz") as bundle:
                    bundle.extractall(scratch, filter="data")
                staging = Path(scratch) / f"llama-{VERSION}"
                subprocess.run([staging / "llama-server", "--version"], check=True)
                staging.rename(target)
    server = target / "llama-server"
    subprocess.run([server, "--version"], check=True)
    return server


if __name__ == "__main__":
    print(install())
