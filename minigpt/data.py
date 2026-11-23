"""Prepare and sample a pinned, chapter-split edition of Alice in Wonderland."""

import argparse
from bisect import bisect_right
from dataclasses import dataclass
from hashlib import sha256
from itertools import accumulate
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen
import xml.etree.ElementTree as ET

import torch

from .tokenizer import ByteTokenizer

MANIFEST = Path(__file__).with_name("alice_manifest.json")
DEFAULT_DIR = Path("data/prepared")
XHTML = "{http://www.w3.org/1999/xhtml}"
PUNCTUATION = str.maketrans({
    "\u2060": "", "‘": "'", "’": "'", "“": '"', "”": '"',
    "—": "--", "–": "-", "…": "...",
})


def extract_chapter(raw: bytes) -> str:
    root = ET.fromstring(raw)
    # Verse lines are separated by <br/> in the source XHTML.
    for line_break in root.iter(f"{XHTML}br"):
        line_break.text = " "
    paragraphs = root.findall(f".//{XHTML}body//{XHTML}p")
    lines = []
    for paragraph in paragraphs:
        text = "".join(paragraph.itertext()).translate(PUNCTUATION)
        normalized = " ".join(text.split())
        if normalized:
            lines.append(normalized)
    if not lines:
        raise ValueError("Chapter contains no paragraph text")
    return "\n\n".join(lines) + "\n"


def verify(content: bytes, expected: str, source: str) -> None:
    observed = sha256(content).hexdigest()
    if observed != expected:
        raise ValueError(f"SHA-256 mismatch for {source}. Expected {expected}, got {observed}")


def prepare(output: Path = DEFAULT_DIR) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    for chapter in manifest["chapters"]:
        n = chapter["number"]
        dest = output / f"{n:02}.txt"
        if dest.exists():
            verify(dest.read_bytes(), chapter["prepared_sha256"], str(dest))
            continue

        url = f"{manifest['raw_base']}/chapter-{n}.xhtml"
        with urlopen(url, timeout=30) as response:
            raw = response.read()
        verify(raw, chapter["raw_sha256"], url)
        prepared = extract_chapter(raw).encode("utf-8")
        verify(prepared, chapter["prepared_sha256"], f"prepared chapter {n}")
        # A reader can only see complete chapters. Rerunning reuses verified files.
        temporary = tempfile.NamedTemporaryFile(dir=output, delete=False)
        temporary_path = Path(temporary.name)
        try:
            with temporary:
                temporary.write(prepared)
            temporary_path.replace(dest)
        finally:
            temporary_path.unlink(missing_ok=True)


@dataclass
class Corpus:
    splits: dict[str, list[torch.Tensor]]
    fingerprint: str

    @classmethod
    def load(cls, path: Path = DEFAULT_DIR) -> "Corpus":
        path = Path(path)
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        splits: dict[str, list[torch.Tensor]] = {"train": [], "val": [], "test": []}
        for chapter in manifest["chapters"]:
            file = path / f"{chapter['number']:02}.txt"
            if not file.is_file():
                raise FileNotFoundError(f"Missing {file}. Run python -m minigpt.data --output {path}")
            content = file.read_bytes()
            verify(content, chapter["prepared_sha256"], str(file))
            ids = ByteTokenizer.encode(content.decode("utf-8"))
            splits[chapter["split"]].append(torch.tensor(ids, dtype=torch.long))
        canonical_manifest = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        return cls(splits, sha256(canonical_manifest.encode()).hexdigest())

    def get_batch(
        self,
        split: str,
        batch_size: int,
        context_length: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample uniformly among all valid starting positions, with replacement."""
        if batch_size <= 0 or context_length <= 0:
            raise ValueError("batch_size and context_length must be positive")
        chapters = self.splits[split]
        # Reading T inputs plus one final target gives N - T starts in a chapter.
        ends = list(accumulate(max(0, len(chapter) - context_length) for chapter in chapters))
        if not ends or ends[-1] == 0:
            raise ValueError(f"No chapter in {split!r} has {context_length + 1} bytes")
        positions = torch.randint(ends[-1], (batch_size,), generator=generator).tolist()
        windows = []
        for position in positions:
            index = bisect_right(ends, position)
            start = position - (ends[index - 1] if index else 0)
            windows.append(chapters[index][start : start + context_length + 1])
        batch = torch.stack(windows)
        return batch[:, :-1], batch[:, 1:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    prepare(args.output)
    corpus = Corpus.load(args.output)
    for split, docs in corpus.splits.items():
        print(f"{split}: {len(docs)} chapters, {sum(map(len, docs)):,} byte tokens")
    print(f"Verified corpus fingerprint: {corpus.fingerprint}")


if __name__ == "__main__":
    main()
