"""Runtime state container for analyzer components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

COMPONENT_NAMES: Tuple[str, ...] = (
    "hardware_detector",
    "model_manager",
    "embedding_cache",
    "text_chunker",
    "embedding_generator",
    "ollama_client",
    "qdrant_manager",
    "performance_monitor",
)


@dataclass
class AnalyzerState:
    """Container keeping track of initialized analyzer components."""

    hardware_detector: Any = None
    model_manager: Any = None
    embedding_cache: Any = None
    text_chunker: Any = None
    embedding_generator: Any = None
    ollama_client: Any = None
    qdrant_manager: Any = None
    performance_monitor: Any = None

    def as_dict(self) -> Dict[str, Any]:
        """Return a shallow dict of component references."""
        return {name: getattr(self, name) for name in COMPONENT_NAMES}

    def ready(self, *components: str) -> bool:
        """Check if specific components are initialized."""
        names: Iterable[str] = components or COMPONENT_NAMES
        return all(getattr(self, name, None) is not None for name in names)

    def reset(self) -> None:
        """Reset the state back to an uninitialized configuration."""
        for name in COMPONENT_NAMES:
            setattr(self, name, None)


state = AnalyzerState()

__all__ = ["AnalyzerState", "COMPONENT_NAMES", "state"]
