"""Embedding cache implementation."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
from typing import Optional

import numpy as np
import structlog


class EmbeddingCache:
    """Model-specific caching for embeddings."""

    def __init__(self) -> None:
        self.logger = structlog.get_logger("embedding_cache")
        self.cache_dir = Path("/app/cache/embeddings")
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            fallback_dir = Path(tempfile.gettempdir()) / "analyzer_cache" / "embeddings"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            self.logger.warning(
                "Falling back to temp directory for embedding cache",
                fallback=str(fallback_dir),
            )
            self.cache_dir = fallback_dir
        self.memory_cache: dict[str, np.ndarray] = {}
        self.max_memory_items = 1000

    def _get_cache_key(self, text: str, model_id: str) -> str:
        """Generate cache key for text and model combination."""
        content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"{model_id}_{content_hash}"

    def get_embedding(self, text: str, model_id: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding if available."""
        cache_key = self._get_cache_key(text, model_id)

        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]

        cache_file = self.cache_dir / f"{cache_key}.npy"
        if cache_file.exists():
            try:
                embedding = np.load(cache_file)
                if len(self.memory_cache) < self.max_memory_items:
                    self.memory_cache[cache_key] = embedding
                return embedding
            except Exception as exc:  # pragma: no cover - defensive fallback
                self.logger.warning(
                    "Failed to load cached embedding",
                    cache_key=cache_key,
                    error=str(exc),
                )

        return None

    def store_embedding(self, text: str, model_id: str, embedding: np.ndarray) -> None:
        """Store embedding in cache."""
        cache_key = self._get_cache_key(text, model_id)

        if len(self.memory_cache) < self.max_memory_items:
            self.memory_cache[cache_key] = embedding

        cache_file = self.cache_dir / f"{cache_key}.npy"
        try:
            np.save(cache_file, embedding)
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.warning(
                "Failed to cache embedding",
                cache_key=cache_key,
                error=str(exc),
            )

    def clear_model_cache(self, model_id: str) -> None:
        """Clear cache for specific model."""
        keys_to_remove = [key for key in self.memory_cache if key.startswith(f"{model_id}_")]
        for key in keys_to_remove:
            del self.memory_cache[key]

        for cache_file in self.cache_dir.glob(f"{model_id}_*.npy"):
            try:
                cache_file.unlink()
            except Exception as exc:  # pragma: no cover - defensive fallback
                self.logger.warning(
                    "Failed to remove cache file",
                    file=str(cache_file),
                    error=str(exc),
                )


__all__ = ["EmbeddingCache"]
