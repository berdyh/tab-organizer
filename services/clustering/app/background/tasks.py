"""
Background task helpers for long-running clustering operations.
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Tuple

import numpy as np

from ..config import QDRANT_SCROLL_LIMIT
from ..logging import logger
from ..models import HDBSCANRequest, UMAPRequest
from ..state import (
    active_jobs,
    dimensionality_reducer,
    executor,
    hdbscan_clusterer,
    qdrant_client,
)


async def check_qdrant_connection() -> bool:
    """Check if Qdrant is accessible."""
    try:
        await asyncio.get_event_loop().run_in_executor(executor, qdrant_client.get_collections)
        return True
    except Exception:  # pragma: no cover - network failure hard to simulate
        return False


async def process_umap_reduction(job_id: str, request: UMAPRequest) -> None:
    """Background task to process UMAP dimensionality reduction."""
    try:
        job = active_jobs[job_id]
        job["status"] = "loading_data"
        job["message"] = "Loading embeddings from Qdrant"

        collection_name = f"session_{request.session_id}"

        collections = await asyncio.get_event_loop().run_in_executor(
            executor, qdrant_client.get_collections
        )

        collection_exists = any(col.name == collection_name for col in collections.collections)

        if not collection_exists:
            job["status"] = "error"
            job["message"] = f"Session {request.session_id} not found"
            return

        # Retrieve embeddings
        search_result = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: qdrant_client.scroll(
                collection_name=collection_name,
                limit=QDRANT_SCROLL_LIMIT,
                with_payload=True,
                with_vectors=True,
            ),
        )

        points = search_result[0]

        if not points:
            job["status"] = "error"
            job["message"] = "No embeddings found in session"
            return

        embeddings: List[List[float]] = []
        point_ids: List[int] = []

        for point in points:
            if point.vector and (
                not request.embedding_model or point.payload.get("embedding_model") == request.embedding_model
            ):
                embeddings.append(point.vector)
                point_ids.append(point.id)

        if not embeddings:
            job["status"] = "error"
            job["message"] = f"No embeddings found for model {request.embedding_model}"
            return

        embeddings_array = np.array(embeddings)

        job["status"] = "processing"
        job["message"] = f"Reducing {len(embeddings_array)} embeddings"

        custom_params: Dict[str, float | int | str] = {}
        if request.n_neighbors is not None:
            custom_params["n_neighbors"] = request.n_neighbors
        if request.min_dist is not None:
            custom_params["min_dist"] = request.min_dist
        if request.metric is not None:
            custom_params["metric"] = request.metric

        reduced_embeddings, performance_metrics = await dimensionality_reducer.reduce_embeddings(
            embeddings=embeddings_array,
            embedding_model=request.embedding_model,
            n_components=request.n_components,
            custom_params=custom_params if custom_params else None,
            batch_size=request.batch_size,
            memory_limit_mb=request.memory_limit_mb,
        )

        job["status"] = "storing"
        job["message"] = "Storing reduced embeddings"

        for index, point_id in enumerate(point_ids):
            await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda index=index, point_id=point_id: qdrant_client.set_payload(
                    collection_name=collection_name,
                    payload={
                        "reduced_embedding": reduced_embeddings[index].tolist(),
                        "umap_metrics": performance_metrics,
                        "reduction_timestamp": time.time(),
                    },
                    points=[point_id],
                ),
            )

        job.update(
            {
                "status": "completed",
                "message": "UMAP reduction completed successfully",
                "performance_metrics": performance_metrics,
                "reduced_dimensions": request.n_components,
                "original_dimensions": embeddings_array.shape[1],
                "n_samples": len(embeddings_array),
                "end_time": time.time(),
            }
        )

        logger.info("UMAP reduction job completed", job_id=job_id, **performance_metrics)

    except Exception as error:  # pragma: no cover - logging branch
        job = active_jobs[job_id]
        job["status"] = "error"
        job["message"] = f"Error: {error}"
        job["error_details"] = str(error)
        logger.error("UMAP reduction job failed", job_id=job_id, error=str(error))


async def process_hdbscan_clustering(job_id: str, request: HDBSCANRequest) -> None:
    """Background task to process HDBSCAN clustering."""
    try:
        job = active_jobs[job_id]
        job["status"] = "loading_data"
        job["message"] = "Loading embeddings from Qdrant"

        collection_name = f"session_{request.session_id}"

        collections = await asyncio.get_event_loop().run_in_executor(
            executor, qdrant_client.get_collections
        )

        collection_exists = any(col.name == collection_name for col in collections.collections)

        if not collection_exists:
            job["status"] = "error"
            job["message"] = f"Session {request.session_id} not found"
            return

        search_result = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: qdrant_client.scroll(
                collection_name=collection_name,
                limit=QDRANT_SCROLL_LIMIT,
                with_payload=True,
                with_vectors=True,
            ),
        )

        points = search_result[0]

        if not points:
            job["status"] = "error"
            job["message"] = "No embeddings found in session"
            return

        embeddings: List[List[float]] = []
        point_ids: List[int] = []
        embedding_model = "unknown"

        for point in points:
            if request.use_reduced_embeddings and "reduced_embedding" in point.payload:
                embeddings.append(point.payload["reduced_embedding"])
                point_ids.append(point.id)
                embedding_model = point.payload.get("embedding_model", "unknown")
            elif not request.use_reduced_embeddings and point.vector is not None:
                embeddings.append(point.vector)
                point_ids.append(point.id)
                embedding_model = point.payload.get("embedding_model", "unknown")

        if not embeddings:
            job["status"] = "error"
            job["message"] = (
                "No reduced embeddings found. Run UMAP reduction first."
                if request.use_reduced_embeddings
                else "No embeddings found in session"
            )
            return

        embeddings_array = np.array(embeddings)

        job["status"] = "processing"
        job["message"] = f"Clustering {len(embeddings_array)} embeddings with HDBSCAN"

        custom_params: Dict[str, float | int | str] = {}
        if request.min_cluster_size is not None:
            custom_params["min_cluster_size"] = request.min_cluster_size
        if request.min_samples is not None:
            custom_params["min_samples"] = request.min_samples
        if request.cluster_selection_epsilon is not None:
            custom_params["cluster_selection_epsilon"] = request.cluster_selection_epsilon
        if request.alpha is not None:
            custom_params["alpha"] = request.alpha
        if request.metric is not None:
            custom_params["metric"] = request.metric

        cluster_labels, clustering_metrics = await hdbscan_clusterer.cluster_embeddings(
            embeddings=embeddings_array,
            session_id=request.session_id,
            custom_params=custom_params if custom_params else None,
            auto_optimize=request.auto_optimize,
            embedding_model=embedding_model,
        )

        job["status"] = "storing"
        job["message"] = "Storing cluster assignments"

        for index, point_id in enumerate(point_ids):
            await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda index=index, point_id=point_id: qdrant_client.set_payload(
                    collection_name=collection_name,
                    payload={
                        "cluster_id": int(cluster_labels[index]),
                        "clustering_metrics": clustering_metrics,
                        "clustering_timestamp": time.time(),
                        "clustering_method": "hdbscan",
                    },
                    points=[point_id],
                ),
            )

        job.update(
            {
                "status": "completed",
                "message": "HDBSCAN clustering completed successfully",
                "clustering_metrics": clustering_metrics,
                "n_clusters": clustering_metrics["n_clusters"],
                "n_noise_points": clustering_metrics["n_noise_points"],
                "silhouette_score": clustering_metrics.get("silhouette_score"),
                "end_time": time.time(),
            }
        )

        logger.info(
            "HDBSCAN clustering job completed",
            job_id=job_id,
            n_clusters=clustering_metrics["n_clusters"],
            silhouette_score=clustering_metrics.get("silhouette_score"),
        )

    except Exception as error:  # pragma: no cover - logging branch
        job = active_jobs[job_id]
        job["status"] = "error"
        job["message"] = f"Error: {error}"
        job["error_details"] = str(error)
        logger.error("HDBSCAN clustering job failed", job_id=job_id, error=str(error))


__all__ = ["check_qdrant_connection", "process_umap_reduction", "process_hdbscan_clustering"]
