"""Duplicate detection helpers."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional, Set, Tuple

from .logging import get_logger
from .state import state

logger = get_logger()


class DuplicateDetector:
    """Advanced duplicate detection using content hashing."""

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        content_index: Optional[Dict[str, str]] = None,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self._hash_store = content_index if content_index is not None else state.content_hashes
        self.content_fingerprints: Dict[str, Tuple[str, Set[str]]] = {}

    def generate_content_hash(self, content: str) -> str:
        """Generate SHA-256 hash of normalized content."""
        normalized = " ".join(content.lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def generate_content_fingerprint(self, content: str) -> Set[str]:
        """Generate content fingerprint using word sets for similarity detection."""
        words = set(content.lower().split())
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        }
        return words - stop_words

    def is_duplicate(self, content: str, url: str) -> Tuple[bool, str, float]:
        """Check if content is duplicate and return (is_duplicate, hash, similarity_score)."""
        content_hash = self.generate_content_hash(content)

        if content_hash in self._hash_store:
            original_url = self._hash_store[content_hash]
            logger.info(
                "Exact duplicate content detected", url=url, original_url=original_url
            )
            return True, content_hash, 1.0

        content_fingerprint = self.generate_content_fingerprint(content)

        for existing_hash, (existing_url, existing_fingerprint) in self.content_fingerprints.items():
            if len(content_fingerprint) == 0 or len(existing_fingerprint) == 0:
                continue

            intersection = len(content_fingerprint & existing_fingerprint)
            union = len(content_fingerprint | existing_fingerprint)
            similarity = intersection / union if union > 0 else 0.0

            if similarity >= self.similarity_threshold:
                logger.info(
                    "Similar content detected",
                    url=url,
                    original_url=existing_url,
                    similarity=similarity,
                )
                return True, content_hash, similarity

        self._hash_store[content_hash] = url
        self.content_fingerprints[content_hash] = (url, content_fingerprint)
        return False, content_hash, 0.0

    def get_duplicate_stats(self) -> Dict[str, Any]:
        """Return duplicate detection statistics."""
        return {
            "total_content_hashes": len(self._hash_store),
            "total_fingerprints": len(self.content_fingerprints),
            "similarity_threshold": self.similarity_threshold,
        }
