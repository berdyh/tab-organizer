"""Utilities for interacting with Qdrant."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)


class QdrantManager:
    """Enhanced Qdrant client with model-specific metadata and advanced querying."""

    def __init__(self, host: str = "qdrant", port: int = 6333) -> None:
        self.host = host
        self.port = port
        self.client: Optional[QdrantClient] = None
        self.logger = structlog.get_logger("qdrant_manager")

    async def initialize(self) -> None:
        """Initialize Qdrant client."""
        try:
            from importlib import import_module

            main_module = import_module("main")
            client_cls = getattr(main_module, "QdrantClient", QdrantClient)
        except Exception:
            client_cls = QdrantClient

        try:
            self.client = client_cls(host=self.host, port=self.port)
            self.logger.info("Qdrant client initialized", host=self.host, port=self.port)
        except Exception as exc:
            self.logger.error("Failed to initialize Qdrant client", error=str(exc))
            raise

    async def ensure_collection_exists(self, collection_name: str, vector_size: int, distance: Distance = Distance.COSINE) -> None:
        """Ensure collection exists with proper configuration."""
        if not self.client:  # pragma: no cover - defensive programming
            raise RuntimeError("Qdrant client not initialized")

        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]

            if collection_name not in collection_names:
                self.logger.info("Creating new collection", collection_name=collection_name, vector_size=vector_size)
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=distance),
                )
            else:
                collection_info = self.client.get_collection(collection_name)
                existing_size = collection_info.config.params.vectors.size
                if existing_size != vector_size:
                    self.logger.warning(
                        "Vector size mismatch",
                        collection=collection_name,
                        expected=vector_size,
                        existing=existing_size,
                    )
        except Exception as exc:
            self.logger.error(
                "Error ensuring collection exists",
                collection_name=collection_name,
                error=str(exc),
            )
            raise

    async def store_analyzed_content(
        self,
        collection_name: str,
        content_items: List[Dict[str, Any]],
        embedding_model: str,
        llm_model: Optional[str] = None,
    ) -> int:
        """Store analyzed content with model-specific metadata."""
        if not self.client:  # pragma: no cover - defensive programming
            raise RuntimeError("Qdrant client not initialized")

        try:
            points: List[PointStruct] = []

            for item in content_items:
                metadata = {
                    "content_id": item["content_id"],
                    "chunk_index": item.get("chunk_index", 0),
                    "text": item["text"],
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "token_count": item.get("token_count", 0),
                    "embedding_model": embedding_model,
                    "llm_model": llm_model,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "summary": item.get("summary"),
                    "keywords": item.get("keywords"),
                    "quality_score": item.get("quality_score"),
                    "quality_assessment": item.get("quality_assessment"),
                    "embedding_generation_time": item.get("embedding_generation_time"),
                    "llm_processing_time": item.get("llm_processing_time"),
                    **item.get("original_metadata", {}),
                }

                point_id = f"{item['content_id']}_chunk_{item.get('chunk_index', 0)}"
                point = PointStruct(id=point_id, vector=item["embedding"], payload=metadata)
                if not isinstance(getattr(point, "payload", None), dict):
                    setattr(point, "payload", metadata)
                points.append(point)

            if points:
                self.client.upsert(collection_name=collection_name, points=points)
                self.logger.info(
                    "Stored analyzed content",
                    collection=collection_name,
                    points_count=len(points),
                    embedding_model=embedding_model,
                    llm_model=llm_model,
                )

            return len(points)
        except Exception as exc:
            self.logger.error(
                "Error storing analyzed content",
                collection_name=collection_name,
                error=str(exc),
            )
            raise

    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get detailed information about a collection."""
        if not self.client:  # pragma: no cover - defensive programming
            raise RuntimeError("Qdrant client not initialized")

        try:
            collection_info = self.client.get_collection(collection_name)
            points_count = self.client.count(collection_name).count

            sample_points = self.client.scroll(
                collection_name=collection_name,
                limit=5,
                with_payload=True,
                with_vectors=False,
            )[0]

            model_stats: Dict[str, Any] = {}
            for point in sample_points:
                embedding_model = point.payload.get("embedding_model")
                llm_model = point.payload.get("llm_model")

                if embedding_model:
                    model_stats.setdefault("embedding_models", set()).add(embedding_model)
                if llm_model:
                    model_stats.setdefault("llm_models", set()).add(llm_model)

            for key, value in model_stats.items():
                if isinstance(value, set):
                    model_stats[key] = list(value)

            return {
                "collection_name": collection_name,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.name,
                "points_count": points_count,
                "model_usage": model_stats,
                "status": collection_info.status.name,
            }
        except Exception as exc:
            self.logger.error(
                "Error getting collection info",
                collection_name=collection_name,
                error=str(exc),
            )
            raise

    async def search_similar_content(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        embedding_model_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar content with optional model filtering."""
        if not self.client:  # pragma: no cover - defensive programming
            raise RuntimeError("Qdrant client not initialized")

        try:
            search_filter = None
            if embedding_model_filter:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="embedding_model",
                            match=MatchValue(value=embedding_model_filter),
                        )
                    ]
                )

            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=limit,
                with_payload=True,
            )

            return [{"id": result.id, "score": result.score, "payload": result.payload} for result in results]
        except Exception as exc:
            self.logger.error(
                "Error searching similar content",
                collection_name=collection_name,
                error=str(exc),
            )
            raise


__all__ = ["QdrantManager"]
