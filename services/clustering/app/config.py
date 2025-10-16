"""
Service configuration constants for the clustering service package.

These values centralise environment-driven knobs so the rest of the codebase
can depend on a single source of truth without sprinkling os.getenv calls.
"""

from __future__ import annotations

import os

QDRANT_HOST: str = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
MAX_WORKERS: int = int(os.getenv("CLUSTERING_MAX_WORKERS", "4"))
QDRANT_SCROLL_LIMIT: int = int(os.getenv("CLUSTERING_QDRANT_SCROLL_LIMIT", "10000"))

__all__ = [
    "QDRANT_HOST",
    "QDRANT_PORT",
    "MAX_WORKERS",
    "QDRANT_SCROLL_LIMIT",
]
