"""
Qdrant client wrapper used by the clustering service.
"""

from __future__ import annotations

from qdrant_client import QdrantClient

from ..config import QDRANT_HOST, QDRANT_PORT

qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

__all__ = ["qdrant_client"]
