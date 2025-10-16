"""Text chunking utilities."""

from __future__ import annotations

from typing import Any, Dict, List

import structlog

try:  # pragma: no cover - tokenizer optional in tests
    import tiktoken
except Exception:  # pragma: no cover - fallback path
    tiktoken = None  # type: ignore


class TextChunker:
    """Handle text chunking with overlap preservation."""

    def __init__(self) -> None:
        self.logger = structlog.get_logger("text_chunker")
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base") if tiktoken else None
        except Exception as exc:  # pragma: no cover - tokenizer init may fail
            self.logger.warning(
                "Could not load tiktoken, using character-based chunking",
                error=str(exc),
            )
            self.tokenizer = None

    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks."""
        if not text or not text.strip():
            return []

        try:
            if self.tokenizer:
                return self._chunk_by_tokens(text, chunk_size, overlap)
            return self._chunk_by_characters(text, chunk_size * 4, overlap * 4)
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.error("Text chunking failed", error=str(exc))
            return [{"text": text, "chunk_index": 0, "token_count": len(text.split())}]

    def _chunk_by_tokens(self, text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """Chunk text by token count using tiktoken."""
        tokens = self.tokenizer.encode(text)  # type: ignore[union-attr]
        chunks: List[Dict[str, Any]] = []

        start = 0
        chunk_index = 0

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)  # type: ignore[union-attr]

            chunks.append(
                {
                    "text": chunk_text,
                    "chunk_index": chunk_index,
                    "token_count": len(chunk_tokens),
                    "start_token": start,
                    "end_token": end,
                }
            )

            start = end - overlap
            chunk_index += 1

            if end >= len(tokens):
                break

        return chunks

    def _chunk_by_characters(self, text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """Fallback chunking by character count."""
        chunks: List[Dict[str, Any]] = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]

            chunks.append(
                {
                    "text": chunk_text,
                    "chunk_index": chunk_index,
                    "token_count": len(chunk_text.split()),
                    "start_char": start,
                    "end_char": end,
                }
            )

            start = end - overlap
            chunk_index += 1

            if end >= len(text):
                break

        return chunks


__all__ = ["TextChunker"]
