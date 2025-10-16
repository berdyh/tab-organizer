"""External service clients used by the chatbot."""

from __future__ import annotations

import httpx
from qdrant_client import QdrantClient

from .config import Settings


def create_qdrant_client(settings: Settings) -> QdrantClient:
    """Instantiate a Qdrant client using service settings."""
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def create_ollama_client(settings: Settings) -> httpx.AsyncClient:
    """Instantiate an asynchronous HTTP client for Ollama."""
    return httpx.AsyncClient(
        base_url=str(settings.ollama_base_url), timeout=settings.ollama_timeout
    )

