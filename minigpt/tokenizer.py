"""UTF-8 bytes are the vocabulary. No fitting or special tokens are required."""

from collections.abc import Iterable


class ByteTokenizer:
    vocab_size = 256

    @staticmethod
    def encode(text: str) -> list[int]:
        return list(text.encode("utf-8"))

    @staticmethod
    def decode(token_ids: Iterable[int], *, errors: str = "strict") -> str:
        """Sampled bytes may form invalid UTF-8. Use errors='replace' for display."""
        return bytes(token_ids).decode("utf-8", errors=errors)
