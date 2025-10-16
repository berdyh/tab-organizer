"""Client factories for external services."""

from __future__ import annotations

from qdrant_client import QdrantClient

from .config import ExportSettings


def create_qdrant_client(settings: ExportSettings) -> QdrantClient:
    """Instantiate the Qdrant client using the provided settings."""
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


__all__ = ["create_qdrant_client", "QdrantClient"]
