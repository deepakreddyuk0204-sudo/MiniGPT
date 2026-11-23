"""Exercise interruption recovery without fetching public model weights."""

import hashlib
import subprocess

import pytest

from scripts import download_chat_model as downloader


def tiny_manifest(data):
    return {"repository": "test/model", "revision": "pinned", "filename": "fixture.gguf",
            "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def test_range_download_reuses_partial_merge_and_publishes_once(tmp_path, monkeypatch):
    expected = b"0123456789abcdef"
    manifest = tiny_manifest(expected)
    (tmp_path / "fixture.gguf.part").write_bytes(expected[:6])
    (tmp_path / "fixture.gguf.range.4").write_bytes(expected[4:8])
    (tmp_path / "fixture.gguf.range.14").write_bytes(expected[14:])
    requested = []

    def fake_curl(args, **kwargs):
        a, b = map(int, args[args.index("--range") + 1].split("-"))
        requested.append((a, b))
        downloader.Path(args[args.index("--output") + 1]).write_bytes(expected[a:b + 1])
        downloader.Path(args[args.index("--dump-header") + 1]).write_text(
            f"HTTP/2 206\r\nContent-Range: bytes {a}-{b}/{len(expected)}\r\n\r\n")
        return subprocess.CompletedProcess(args, 0, "206")

    monkeypatch.setattr(downloader, "BLOCK_SIZE", 4)
    monkeypatch.setattr(downloader.subprocess, "run", fake_curl)
    result = downloader.download(tmp_path, manifest)
    assert result.read_bytes() == expected
    assert set(requested) == {(8, 11), (12, 13)}
    assert sorted(p.name for p in tmp_path.iterdir()) == [".download.lock", "fixture.gguf"]
    assert downloader.download(tmp_path, manifest) == result
    assert len(requested) == 2


def test_corrupt_complete_file_cannot_be_published(tmp_path):
    expected = b"abcd"
    (tmp_path / "fixture.gguf.part").write_bytes(b"efgh")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        downloader.download(tmp_path, tiny_manifest(expected))
    assert not (tmp_path / "fixture.gguf").exists()


def test_wrong_range_header_is_rejected(tmp_path, monkeypatch):
    def wrong_range(args, **kwargs):
        downloader.Path(args[args.index("--output") + 1]).write_bytes(b"1234")
        downloader.Path(args[args.index("--dump-header") + 1]).write_text(
            "HTTP/2 206\r\nContent-Range: bytes 1-4/8\r\n\r\n")
        return subprocess.CompletedProcess(args, 0, "206")

    monkeypatch.setattr(downloader.subprocess, "run", wrong_range)
    monkeypatch.setattr(downloader.time, "sleep", lambda seconds: None)
    output = tmp_path / "range.0"
    with pytest.raises(RuntimeError, match="requested range"):
        downloader.fetch_range("https://example.invalid", 8, (0, 4, output, False))
    assert not output.exists()
