"""Compatibility layer exposing the analyzer package under the legacy `main` module."""

from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType
from typing import Any

from analyzer import (
    AnalysisRequest,
    AnalysisResponse,
    ContentItem,
    EmbeddingCache,
    EmbeddingGenerator,
    EmbeddingRequest,
    EmbeddingResponse,
    HardwareDetector,
    HardwareInfo,
    ModelManager,
    ModelPerformanceMetrics,
    ModelRecommendation,
    ModelSwitchRequest,
    OllamaClient,
    PerformanceMonitor,
    QdrantConnectionInfo,
    QdrantManager,
    TextChunker,
    app,
    configure_logging,
    create_app,
    get_state,
    process_complete_analysis_job,
    process_embeddings_job,
    state,
)
from analyzer.state import COMPONENT_NAMES, AnalyzerState

__all__ = [
    "AnalysisRequest",
    "AnalysisResponse",
    "AnalyzerState",
    "COMPONENT_NAMES",
    "ContentItem",
    "EmbeddingCache",
    "EmbeddingGenerator",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "HardwareDetector",
    "HardwareInfo",
    "ModelManager",
    "ModelPerformanceMetrics",
    "ModelRecommendation",
    "ModelSwitchRequest",
    "OllamaClient",
    "PerformanceMonitor",
    "QdrantConnectionInfo",
    "QdrantManager",
    "TextChunker",
    "app",
    "configure_logging",
    "create_app",
    "get_state",
    "process_complete_analysis_job",
    "process_embeddings_job",
    "state",
] + list(COMPONENT_NAMES)


class _AnalyzerCompatibilityModule(ModuleType):
    """Module proxy that forwards component access to the shared analyzer state."""

    _component_names = set(COMPONENT_NAMES)

    def __getattr__(self, name: str) -> Any:
        if name in self._component_names:
            return getattr(state, name)
        if name in {"torch", "sentence_transformers", "qdrant_client", "tiktoken"}:
            try:
                return import_module(name)
            except ModuleNotFoundError as exc:
                raise AttributeError(name) from exc
        if name == "QdrantClient":
            try:
                module = import_module("qdrant_client")
            except ModuleNotFoundError as exc:
                raise AttributeError(name) from exc
            attr = getattr(module, "QdrantClient", None)
            if attr is None:
                raise AttributeError(name)
            return attr
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._component_names:
            setattr(state, name, value)
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in self._component_names:
            setattr(state, name, None)
        else:
            super().__delattr__(name)


_module = _AnalyzerCompatibilityModule(__name__)
for _name, _value in list(globals().items()):
    if _name in {"_AnalyzerCompatibilityModule", "_module"}:
        continue
    setattr(_module, _name, _value)

sys.modules[__name__] = _module
