"""Embedding generation utilities."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import structlog

from .cache import EmbeddingCache
from .model_management import ModelManager


class _SentenceTransformerProtocol(Protocol):
    def encode(self, inputs: List[str], convert_to_numpy: bool = True) -> Any: ...


class EmbeddingGenerator:
    """Generate embeddings with configurable models and dynamic switching."""

    def __init__(
        self,
        model_manager: ModelManager,
        embedding_cache: EmbeddingCache,
        hardware_detector: Optional[Any] = None,
    ) -> None:
        self.logger = structlog.get_logger("embedding_generator")
        self.model_manager = model_manager
        self.embedding_cache = embedding_cache
        self.hardware_detector = hardware_detector
        self.current_model_id: Optional[str] = None
        self.current_model: Optional[_SentenceTransformerProtocol] = None
        self._torch = self._import_torch()
        self._torch_cuda = getattr(self._torch, "cuda", None) if self._torch else None
        self._sentence_transformer_cls = self._import_sentence_transformer()
        self.device = "cuda" if self._torch_cuda and getattr(self._torch_cuda, "is_available", lambda: False)() else "cpu"

        self._initialize_default_model()

    def _initialize_default_model(self) -> None:
        """Initialize with a default embedding model."""
        try:
            hardware_info = {}
            if self.hardware_detector is not None:
                hardware_info = self.hardware_detector.detect_hardware()
            recommendation = self.model_manager.recommend_model(hardware_info)
            default_model = recommendation["recommended_model"]
        except Exception:
            default_model = "all-minilm"

        self.switch_model(default_model)

    def switch_model(self, model_id: str) -> bool:
        """Switch to a different embedding model without service restart."""
        if model_id == self.current_model_id and self.current_model is not None:
            return True

        models_config = self.model_manager.get_available_models()
        if model_id not in models_config:
            self.logger.error("Unknown embedding model", model_id=model_id)
            return False

        model_config = models_config[model_id]

        try:
            model_name = model_config.get("model_name", model_id)
            model_name_mapping = {
                "all-minilm": "all-MiniLM-L6-v2",
                "nomic-embed-text": "nomic-ai/nomic-embed-text-v1",
                "mxbai-embed-large": "mixedbread-ai/mxbai-embed-large-v1",
            }
            actual_model_name = model_name_mapping.get(model_id, model_name)

            self.logger.info(
                "Loading embedding model",
                model_id=model_id,
                model_name=actual_model_name,
            )

            if self._sentence_transformer_cls is None:
                raise RuntimeError("sentence_transformers is not available")

            new_model = self._sentence_transformer_cls(actual_model_name, device=self.device)

            if self.current_model is not None:
                del self.current_model
                if self._torch_cuda and getattr(self._torch_cuda, "is_available", lambda: False)():
                    empty_cache = getattr(self._torch_cuda, "empty_cache", None)
                    if callable(empty_cache):
                        empty_cache()

            self.current_model = new_model
            self.current_model_id = model_id

            self.logger.info(
                "Successfully switched embedding model",
                model_id=model_id,
                device=self.device,
                dimensions=model_config["dimensions"],
            )
            return True

        except Exception as exc:
            self.logger.error(
                "Failed to switch embedding model",
                model_id=model_id,
                error=str(exc),
            )
            return False

    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a list of texts."""
        if not self.current_model:
            raise RuntimeError("No embedding model loaded")

        if not texts:
            return []

        embeddings: List[Optional[np.ndarray]] = [None] * len(texts)
        texts_to_encode: List[str] = []
        encode_indices: List[int] = []
        cache_hits = 0

        for idx, text in enumerate(texts):
            cached_embedding = self.embedding_cache.get_embedding(text, self.current_model_id)  # type: ignore[arg-type]
            if cached_embedding is not None:
                embeddings[idx] = cached_embedding
                cache_hits += 1
            else:
                texts_to_encode.append(text)
                encode_indices.append(idx)

        if texts_to_encode:
            try:
                batch_embeddings = self.current_model.encode(texts_to_encode, convert_to_numpy=True)
            except Exception as exc:
                self.logger.error("Failed to generate embeddings", error=str(exc))
                batch_embeddings = [
                    np.zeros(self.model_manager.get_available_models()[self.current_model_id]["dimensions"])  # type: ignore[index]
                    for _ in texts_to_encode
                ]

            for idx, embedding, original_text in zip(encode_indices, batch_embeddings, texts_to_encode):
                embeddings[idx] = embedding
                self.embedding_cache.store_embedding(original_text, self.current_model_id, embedding)  # type: ignore[arg-type]

        self.logger.info(
            "Generated embeddings",
            total_texts=len(texts),
            cache_hits=cache_hits,
            new_embeddings=len(texts) - cache_hits,
        )

        dimensions = self.model_manager.get_available_models()[self.current_model_id]["dimensions"]  # type: ignore[index]
        return [vector if vector is not None else np.zeros(dimensions) for vector in embeddings]

    def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about currently loaded model."""
        if not self.current_model_id:
            return {"error": "No model loaded"}

        model_config = self.model_manager.get_available_models().get(self.current_model_id, {})
        return {
            "model_id": self.current_model_id,
            "model_name": model_config.get("name", "Unknown"),
            "dimensions": model_config.get("dimensions", 0),
            "device": self.device,
            "description": model_config.get("description", ""),
        }

    @staticmethod
    def _import_torch():
        """Import torch lazily to support test monkeypatching."""
        try:  # pragma: no cover - optional dependency
            return import_module("torch")
        except Exception:
            return None

    @staticmethod
    def _import_sentence_transformer():
        """Import SentenceTransformer lazily."""
        try:  # pragma: no cover - optional dependency
            module = import_module("sentence_transformers")
            return getattr(module, "SentenceTransformer", None)
        except Exception:
            return None


__all__ = ["EmbeddingGenerator"]
