import io
import json
from hashlib import sha256

import pytest
import torch

from minigpt import data
from minigpt.data import Corpus, extract_chapter
from minigpt.tokenizer import ByteTokenizer


@pytest.mark.parametrize("text", ["", "Alice\n", "café e\u0301", "兔🐇"])
def test_tokenizer_round_trip(text):
    ids = ByteTokenizer.encode(text)
    assert all(0 <= i < ByteTokenizer.vocab_size for i in ids)
    assert ByteTokenizer.decode(ids) == text
    assert len(ids) == len(text.encode("utf-8"))


def test_invalid_sampled_utf8():
    with pytest.raises(UnicodeDecodeError):
        ByteTokenizer.decode([255])
    assert ByteTokenizer.decode([255], errors="replace") == "�"


def test_preprocessing():
    markup = """<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Skip</title></head>
       <body><section><h2>I</h2><p>“A   <i>word</i>!”</p><img alt="Skip"/>
       <p>a<br/>b⁠—c…</p></section></body></html>"""
    assert extract_chapter(markup.encode()) == '"A word!"\n\na b--c...\n'


def test_seeded_shift_without_crossing_chapters():
    chapters = [torch.tensor([0, 1, 2, 3]), torch.tensor([10, 11, 12, 13])]
    corpus = Corpus({"train": chapters}, "test")
    x, y = corpus.get_batch("train", 32, 3, torch.Generator().manual_seed(1))
    again = corpus.get_batch("train", 32, 3, torch.Generator().manual_seed(1))
    assert x.shape == y.shape == (32, 3)
    assert x.dtype == y.dtype == torch.long
    assert torch.equal(x, again[0]) and torch.equal(y, again[1])
    assert torch.equal(x + 1, y)
    assert set(x[:, 0].tolist()) == {0, 10}
    with pytest.raises(ValueError, match="No chapter"):
        corpus.get_batch("train", 1, 4, torch.Generator())


def test_prepare_cache_and_integrity(tmp_path, monkeypatch):
    # Purpose-built fixture exercises the downloader offline.
    source = b'<html xmlns="http://www.w3.org/1999/xhtml"><body><p>abcd</p></body></html>'
    chapter = {
        "number": 1,
        "split": "train",
        "raw_sha256": sha256(source).hexdigest(),
        "prepared_sha256": sha256(b"abcd\n").hexdigest(),
    }
    manifest = tmp_path / "fixture.json"
    manifest.write_text(json.dumps({"raw_base": "https://example.invalid", "chapters": [chapter]}))
    monkeypatch.setattr(data, "MANIFEST", manifest)
    output = tmp_path / "prepared"

    monkeypatch.setattr(data, "urlopen", lambda *args, **kwargs: io.BytesIO(b"wrong bytes"))
    with pytest.raises(ValueError, match="SHA-256"):
        data.prepare(output)
    assert not (output / "01.txt").exists()

    monkeypatch.setattr(data, "urlopen", lambda *args, **kwargs: io.BytesIO(source))
    data.prepare(output)
    monkeypatch.setattr(data, "urlopen", lambda *args, **kwargs: pytest.fail("Unnecessary download"))
    data.prepare(output)
    corpus = Corpus.load(output)
    assert corpus.splits["train"][0].tolist() == ByteTokenizer.encode("abcd\n")
    assert corpus.fingerprint == Corpus.load(output).fingerprint
    (output / "01.txt").write_text("changed")
    with pytest.raises(ValueError, match="SHA-256"):
        Corpus.load(output)
    with pytest.raises(ValueError, match="SHA-256"):
        data.prepare(output)


def test_pinned_chapter_partition():
    manifest = json.loads(data.MANIFEST.read_text(encoding="utf-8"))
    chapters = manifest["chapters"]
    assert [c["number"] for c in chapters] == list(range(1, 13))
    assert [c["split"] for c in chapters] == ["train"] * 10 + ["val", "test"]
    assert sum(c["raw_bytes"] for c in chapters) == 186_803
    assert sum(c["prepared_bytes"] for c in chapters) == 142_840
