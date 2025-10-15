"""Qdrant client helpers for the Session service."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http import models as http_models

logger = structlog.get_logger("session.qdrant")


class SessionVectorStore:
    """Wrapper around Qdrant operations used by the session service."""

    def __init__(self) -> None:
        host = os.getenv("QDRANT_HOST", "qdrant")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        self._client = QdrantClient(host=host, port=port)

    @property
    def client(self) -> QdrantClient:
        return self._client

    def ensure_collection(self, collection_name: str, vector_size: int = 384) -> None:
        """Ensure the collection exists, creating it if necessary."""
        collections = self._client.get_collections().collections
        collection_names = {col.name for col in collections}
        if collection_name in collection_names:
            return

        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=http_models.VectorParams(
                size=vector_size,
                distance=http_models.Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection", collection_name=collection_name)

    def delete_collection(self, collection_name: str) -> None:
        """Remove a collection if it exists."""
        try:
            self._client.delete_collection(collection_name=collection_name)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to delete collection",
                collection_name=collection_name,
                error=str(exc),
            )

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        info = self._client.get_collection(collection_name)
        return {
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status,
        }

    def scroll_points(
        self,
        collection_name: str,
        limit: int = 100,
        offset: Optional[str] = None,
    ) -> Sequence[http_models.Record]:
        result, _ = self._client.scroll(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        return result

    def export_points(self, collection_name: str) -> Dict[str, Any]:
        exported_points: List[Dict[str, Any]] = []
        next_offset: Optional[str] = None

        while True:
            batch, next_offset = self._client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=next_offset,
                with_payload=True,
                with_vectors=True,
            )
            if not batch:
                break

            for record in batch:
                exported_points.append(
                    {
                        "id": record.id,
                        "vector": record.vector,
                        "payload": record.payload,
                    }
                )

            if next_offset is None:
                break

        return {
            "collection_name": collection_name,
            "exported_points": len(exported_points),
            "points": exported_points,
        }

    def import_points(
        self,
        collection_name: str,
        points: Iterable[Dict[str, Any]],
        vector_size: int = 384,
    ) -> None:
        self.ensure_collection(collection_name, vector_size=vector_size)
        batches: List[http_models.PointStruct] = []
        for point in points:
            batches.append(
                http_models.PointStruct(
                    id=point.get("id"),
                    vector=point["vector"],
                    payload=point.get("payload", {}),
                )
            )

        if batches:
            self._client.upsert(collection_name=collection_name, points=batches)

    def delete_points(self, collection_name: str, point_ids: Iterable[str]) -> None:
        ids = list(point_ids)
        if not ids:
            return
        selector = http_models.PointIdsList(points=ids)
        self._client.delete(collection_name=collection_name, points_selector=selector)


session_vector_store = SessionVectorStore()


__all__ = ["SessionVectorStore", "session_vector_store"]
