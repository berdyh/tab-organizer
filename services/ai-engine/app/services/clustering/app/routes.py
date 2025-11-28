"""
FastAPI routes for the clustering service.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from qdrant_client.models import FieldCondition, Filter, MatchValue

from .config import QDRANT_SCROLL_LIMIT
from .background.tasks import check_qdrant_connection, process_hdbscan_clustering, process_umap_reduction
from .logging import logger
from .models import (
    HDBSCANRequest,
    HDBSCANResponse,
    RecommendationRequest,
    RecommendationResponse,
    SimilaritySearchRequest,
    SimilaritySearchResponse,
    UMAPRequest,
    UMAPResponse,
    VisualizationRequest,
)
from .state import (
    active_jobs,
    dimensionality_reducer,
    hdbscan_clusterer,
    qdrant_client,
    executor,
    similarity_search_engine,
    visualization_generator,
)
from .umap import ModelAwareUMAPConfig


router = APIRouter()


async def _scroll_session_points(
    session_id: str,
    *,
    scroll_filter: Optional[Filter] = None,
    with_vectors: bool = False,
) -> List[Any]:
    """Fetch points for a session collection from Qdrant."""
    collection_name = f"session_{session_id}"
    try:
        points, _next_page = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: qdrant_client.scroll(
                collection_name=collection_name,
                limit=QDRANT_SCROLL_LIMIT,
                with_payload=True,
                with_vectors=with_vectors,
                scroll_filter=scroll_filter,
            ),
        )
        return points
    except Exception as error:
        message = str(error).lower()
        status_code = 404 if "not found" in message else 500
        logger.error(
            "Failed to fetch session data from Qdrant",
            session_id=session_id,
            error=str(error),
        )
        raise HTTPException(
            status_code=status_code,
            detail=f"Failed to fetch clustering data for session {session_id}",
        ) from error


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "clustering",
        "timestamp": time.time(),
        "qdrant_connected": await check_qdrant_connection(),
    }


@router.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint."""
    return {
        "service": "Clustering Service",
        "version": "1.0.0",
        "status": "running",
        "features": [
            "UMAP dimensionality reduction",
            "Model-aware optimization",
            "Batch processing",
            "Interactive visualizations",
            "Performance metrics",
        ],
    }


@router.post("/umap/reduce", response_model=UMAPResponse)
async def reduce_dimensions(request: UMAPRequest, background_tasks: BackgroundTasks) -> UMAPResponse:
    """Reduce embedding dimensions using UMAP with model-aware optimization."""
    job_id = f"umap_{request.session_id}_{int(time.time())}"
    active_jobs[job_id] = {
        "status": "started",
        "message": "Initializing UMAP reduction",
        "start_time": time.time(),
    }
    background_tasks.add_task(process_umap_reduction, job_id, request)
    return UMAPResponse(job_id=job_id, status="started", message="UMAP reduction job started")


@router.get("/umap/status/{job_id}")
async def get_umap_status(job_id: str) -> Dict[str, Any]:
    """Get status of a UMAP reduction job."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return active_jobs[job_id]


@router.get("/umap/models/{embedding_model}/config")
async def get_model_config(embedding_model: str, n_samples: int = 1000, memory_limit_mb: int = 2048) -> Dict[str, Any]:
    """Get model-aware configuration for a given embedding model."""
    config = ModelAwareUMAPConfig.get_config(embedding_model, n_samples, memory_limit_mb)
    return {"embedding_model": embedding_model, "config": config, "n_samples": n_samples, "memory_limit_mb": memory_limit_mb}


@router.get("/umap/models/supported")
async def get_supported_models() -> Dict[str, Any]:
    """List supported embedding models for UMAP optimization."""
    return {"supported_models": list(ModelAwareUMAPConfig.EMBEDDING_MODEL_CONFIGS.keys())}


@router.post("/visualize/create")
async def create_visualization(request: VisualizationRequest) -> Dict[str, Any]:
    """Create interactive visualization of dimensionality-reduced embeddings."""
    try:
        collection_name = f"session_{request.session_id}"

        collections = await asyncio.get_event_loop().run_in_executor(executor, qdrant_client.get_collections)
        collection_exists = any(col.name == collection_name for col in collections.collections)

        if not collection_exists:
            raise HTTPException(status_code=404, detail=f"Session {request.session_id} not found")

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
            raise HTTPException(status_code=404, detail="No embeddings found for visualization")

        reduced_embeddings: List[List[float]] = []
        metadata: List[Dict[str, Any]] = []

        for point in points:
            if "reduced_embedding" in point.payload:
                reduced_embeddings.append(point.payload["reduced_embedding"])
                metadata.append(
                    {
                        "cluster_id": point.payload.get("cluster_id", -1),
                        "embedding_model": point.payload.get("embedding_model", "unknown"),
                        "quality_score": point.payload.get("quality_score", 0.0),
                        "title": point.payload.get("title", ""),
                        "url": point.payload.get("url", ""),
                    }
                )

        if not reduced_embeddings:
            raise HTTPException(status_code=404, detail="No reduced embeddings found. Run UMAP reduction first.")

        reduced_array = np.array(reduced_embeddings)

        performance_metrics = None
        if points and "umap_metrics" in points[0].payload:
            performance_metrics = points[0].payload["umap_metrics"]

        return await visualization_generator.create_cluster_plot(
            reduced_embeddings=reduced_array,
            metadata=metadata,
            plot_type=request.plot_type,
            color_by=request.color_by,
            include_metrics=request.include_metrics,
            performance_metrics=performance_metrics,
        )

    except HTTPException:
        raise
    except Exception as error:
        logger.error("Error creating visualization", error=str(error))
        raise HTTPException(status_code=500, detail=str(error))


@router.get("/performance/metrics/{job_id}")
async def get_performance_metrics(job_id: str) -> Dict[str, Any]:
    """Get detailed performance metrics for a UMAP reduction job."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job_data = active_jobs[job_id]
    if "performance_metrics" not in job_data:
        return {"message": "Performance metrics not yet available"}
    return job_data["performance_metrics"]


@router.get("/performance/models")
async def get_model_performance() -> Dict[str, Any]:
    """Get performance comparison across different embedding models."""
    cached_models = dimensionality_reducer.list_cached_models()
    performance_data = {
        model_key: dimensionality_reducer.get_performance_metrics(model_key)
        for model_key in cached_models
        if dimensionality_reducer.get_performance_metrics(model_key)
    }
    return {"model_performance": performance_data, "total_models_cached": len(cached_models)}


@router.post("/hdbscan/cluster", response_model=HDBSCANResponse)
async def cluster_with_hdbscan(request: HDBSCANRequest, background_tasks: BackgroundTasks) -> HDBSCANResponse:
    """Perform HDBSCAN clustering with parameter optimization."""
    job_id = f"hdbscan_{request.session_id}_{int(time.time())}"
    active_jobs[job_id] = {
        "status": "started",
        "message": "Initializing HDBSCAN clustering",
        "start_time": time.time(),
    }
    background_tasks.add_task(process_hdbscan_clustering, job_id, request)
    return HDBSCANResponse(job_id=job_id, status="started", message="HDBSCAN clustering job started")


@router.get("/hdbscan/status/{job_id}")
async def get_hdbscan_status(job_id: str) -> Dict[str, Any]:
    """Get status of HDBSCAN clustering job."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return active_jobs[job_id]


@router.get("/hdbscan/parameters/optimize")
async def get_optimized_parameters(
    session_id: str,
    embedding_model: str = "unknown",
    n_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Get optimized HDBSCAN parameters for a dataset."""
    try:
        if not n_samples:
            collection_name = f"session_{session_id}"
            collections = await asyncio.get_event_loop().run_in_executor(
                executor, qdrant_client.get_collections
            )
            collection_exists = any(col.name == collection_name for col in collections.collections)
            if not collection_exists:
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

            collection_info = await asyncio.get_event_loop().run_in_executor(
                executor, qdrant_client.get_collection, collection_name
            )
            n_samples = collection_info.points_count

        dummy_embeddings = np.random.randn(min(n_samples, 1000), 768)
        params = hdbscan_clusterer._optimize_parameters(dummy_embeddings, n_samples, embedding_model)
        return {
            "session_id": session_id,
            "embedding_model": embedding_model,
            "n_samples": n_samples,
            "optimized_parameters": params,
            "parameter_explanation": {
                "min_cluster_size": "Minimum number of points required to form a cluster",
                "min_samples": "Minimum number of points in neighborhood for core point",
                "metric": "Distance metric used for clustering",
                "alpha": "Controls cluster selection strictness",
                "cluster_selection_epsilon": "Distance threshold for cluster merging",
            },
        }
    except HTTPException:
        raise
    except Exception as error:
        logger.error("Error optimizing HDBSCAN parameters", error=str(error))
        raise HTTPException(status_code=500, detail=str(error))


@router.get("/hdbscan/metrics/{job_id}")
async def get_cluster_metrics(job_id: str) -> Dict[str, Any]:
    """Get detailed clustering validation metrics."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job_data = active_jobs[job_id]
    if "clustering_metrics" not in job_data:
        return {"message": "Clustering metrics not yet available"}
    return job_data["clustering_metrics"]


@router.get("/hdbscan/hierarchy/{clusterer_key}")
async def get_cluster_hierarchy(clusterer_key: str) -> Dict[str, Any]:
    """Get hierarchical clustering information."""
    hierarchy = hdbscan_clusterer.get_cluster_hierarchy(clusterer_key)
    if hierarchy is None:
        raise HTTPException(status_code=404, detail="Clusterer not found or hierarchy not available")
    return hierarchy


@router.post("/hdbscan/predict/{clusterer_key}")
async def predict_clusters(clusterer_key: str, embeddings: List[List[float]]) -> Dict[str, Any]:
    """Predict cluster labels for new embeddings."""
    try:
        new_embeddings = np.array(embeddings)
        predictions = hdbscan_clusterer.predict_cluster(clusterer_key, new_embeddings)
        if predictions is None:
            raise HTTPException(status_code=404, detail="Clusterer not found or prediction failed")
        return {"clusterer_key": clusterer_key, "predictions": predictions.tolist(), "n_predictions": len(predictions)}
    except HTTPException:
        raise
    except Exception as error:
        logger.error("Error predicting clusters", error=str(error))
        raise HTTPException(status_code=500, detail=str(error))


@router.get("/clusters/{session_id}")
async def get_clusters_overview(session_id: str) -> List[Dict[str, Any]]:
    """Return aggregate information for clusters within a session."""
    points = await _scroll_session_points(session_id)
    if not points:
        return []

    clusters: Dict[int, Dict[str, Any]] = {}
    for point in points:
        payload = point.payload or {}
        cluster_id = payload.get("cluster_id", -1)
        info = clusters.setdefault(
            cluster_id,
            {
                "count": 0,
                "quality_sum": 0.0,
                "models": set(),
                "min_ts": None,
                "max_ts": None,
                "sample_titles": [],
            },
        )
        info["count"] += 1
        quality = payload.get("quality_score")
        if isinstance(quality, (int, float)):
            info["quality_sum"] += float(quality)
        model = payload.get("embedding_model")
        if model:
            info["models"].add(model)
        timestamp = payload.get("timestamp")
        if isinstance(timestamp, (int, float)):
            info["min_ts"] = timestamp if info["min_ts"] is None else min(info["min_ts"], timestamp)
            info["max_ts"] = timestamp if info["max_ts"] is None else max(info["max_ts"], timestamp)
        title = payload.get("title")
        if title and title not in info["sample_titles"]:
            info["sample_titles"].append(title)
            if len(info["sample_titles"]) > 3:
                info["sample_titles"] = info["sample_titles"][:3]

    overview: List[Dict[str, Any]] = []
    for cluster_id, info in clusters.items():
        average_quality = info["quality_sum"] / info["count"] if info["count"] else 0.0
        timestamp_range = (
            {"min": info["min_ts"], "max": info["max_ts"]}
            if info["min_ts"] is not None and info["max_ts"] is not None
            else None
        )
        overview.append(
            {
                "id": cluster_id,
                "label": "Noise" if cluster_id == -1 else f"Cluster {cluster_id}",
                "size": info["count"],
                "average_quality": average_quality,
                "embedding_models": sorted(info["models"]),
                "timestamp_range": timestamp_range,
                "sample_titles": info["sample_titles"],
            }
        )

    overview.sort(key=lambda item: (item["id"] == -1, -item["size"], item["id"]))
    return overview


@router.get("/clusters/details/{cluster_id}")
async def get_cluster_details(
    cluster_id: int,
    session_id: str = Query(..., description="Session identifier for the cluster"),
    limit: int = Query(200, ge=1, le=QDRANT_SCROLL_LIMIT),
) -> Dict[str, Any]:
    """Return detailed information for a specific cluster."""
    scroll_filter = Filter(
        must=[
            FieldCondition(
                key="cluster_id",
                match=MatchValue(value=cluster_id),
            )
        ]
    )
    points = await _scroll_session_points(
        session_id,
        scroll_filter=scroll_filter,
        with_vectors=False,
    )
    items: List[Dict[str, Any]] = []
    for point in points[:limit]:
        payload = point.payload or {}
        item = {
            "id": point.id,
            "title": payload.get("title"),
            "url": payload.get("url"),
            "cluster_id": payload.get("cluster_id", cluster_id),
            "quality_score": payload.get("quality_score"),
            "embedding_model": payload.get("embedding_model"),
            "timestamp": payload.get("timestamp"),
            "content_type": payload.get("content_type"),
            "metadata": payload.get("metadata"),
        }
        if "reduced_embedding" in payload:
            item["reduced_embedding"] = payload["reduced_embedding"]
        items.append(item)

    return {
        "cluster_id": cluster_id,
        "session_id": session_id,
        "size": len(points),
        "items": items,
    }


@router.get("/clusters/{session_id}/visualize")
async def visualize_clusters(
    session_id: str,
    type: str = Query("2d", regex="^(2d|3d)$", description="Visualization type"),
    color_by: str = Query("cluster", description="Coloring strategy"),
    include_metrics: bool = Query(True, description="Include performance metrics"),
) -> Dict[str, Any]:
    """Generate a visualization for a session's clusters."""
    request = VisualizationRequest(
        session_id=session_id,
        plot_type="3d" if type.lower() == "3d" else "2d",
        color_by=color_by,
        include_metrics=include_metrics,
    )
    return await create_visualization(request)


@router.post("/similarity/search", response_model=SimilaritySearchResponse)
async def similarity_search(request: SimilaritySearchRequest) -> SimilaritySearchResponse:
    """
    Perform vector similarity search to find similar content.
    """
    try:
        if not request.query_embedding and not request.query_text:
            raise HTTPException(status_code=400, detail="Either query_embedding or query_text must be provided")

        if not request.query_embedding:
            raise HTTPException(
                status_code=400,
                detail="query_text to embedding conversion not implemented. Please provide query_embedding.",
            )

        query_embedding = np.array(request.query_embedding)

        results, search_metadata = await similarity_search_engine.vector_similarity_search(
            session_id=request.session_id,
            query_embedding=query_embedding,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            filter_cluster_id=request.filter_cluster_id,
            use_reduced_embeddings=request.use_reduced_embeddings,
        )

        query_info = {
            "session_id": request.session_id,
            "embedding_dimensions": len(request.query_embedding),
            "top_k": request.top_k,
            "similarity_threshold": request.similarity_threshold,
            "filter_cluster_id": request.filter_cluster_id,
            "use_reduced_embeddings": request.use_reduced_embeddings,
        }

        return SimilaritySearchResponse(results=results, query_info=query_info, search_metadata=search_metadata)

    except HTTPException:
        raise
    except Exception as error:
        logger.error("Error performing similarity search", error=str(error))
        raise HTTPException(status_code=500, detail=str(error))


@router.post("/recommendations/content-based", response_model=RecommendationResponse)
async def get_content_based_recommendations(request: RecommendationRequest) -> RecommendationResponse:
    """
    Generate content-based recommendations based on user interactions.
    """
    try:
        if not request.user_interactions:
            raise HTTPException(status_code=400, detail="user_interactions cannot be empty")

        if request.recommendation_type == "content_based":
            recommendations, metadata = await similarity_search_engine.content_based_recommendations(
                session_id=request.session_id,
                user_interactions=request.user_interactions,
                top_k=request.top_k,
                diversity_factor=request.diversity_factor,
            )
        elif request.recommendation_type == "collaborative":
            recommendations, metadata = await similarity_search_engine.collaborative_filtering_recommendations(
                session_id=request.session_id,
                user_interactions=request.user_interactions,
                top_k=request.top_k,
            )
        elif request.recommendation_type == "hybrid":
            content_recs, content_meta = await similarity_search_engine.content_based_recommendations(
                session_id=request.session_id,
                user_interactions=request.user_interactions,
                top_k=request.top_k // 2,
                diversity_factor=request.diversity_factor,
            )
            collab_recs, collab_meta = await similarity_search_engine.collaborative_filtering_recommendations(
                session_id=request.session_id,
                user_interactions=request.user_interactions,
                top_k=request.top_k // 2,
            )

            recommendations = (content_recs + collab_recs)[: request.top_k]
            metadata = {
                "recommendation_type": "hybrid",
                "content_based_metadata": content_meta,
                "collaborative_metadata": collab_meta,
                "total_recommendations": len(recommendations),
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported recommendation_type: {request.recommendation_type}",
            )

        recommendation_metadata = {
            "session_id": request.session_id,
            "recommendation_type": request.recommendation_type,
            "user_interactions_count": len(request.user_interactions),
            "top_k": request.top_k,
            "diversity_factor": request.diversity_factor,
            **metadata,
        }

        return RecommendationResponse(recommendations=recommendations, recommendation_metadata=recommendation_metadata)

    except HTTPException:
        raise
    except Exception as error:
        logger.error("Error generating recommendations", error=str(error))
        raise HTTPException(status_code=500, detail=str(error))


@router.get("/similarity/search/{session_id}/clusters/{cluster_id}")
async def get_similar_from_cluster(
    session_id: str,
    cluster_id: int,
    top_k: int = 10,
    similarity_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Get items similar to a specific cluster centroid."""
    try:
        collection_name = f"session_{session_id}"

        search_result = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="cluster_id",
                            match=MatchValue(value=cluster_id),
                        )
                    ]
                ),
                limit=1000,
                with_payload=True,
                with_vectors=True,
            ),
        )

        cluster_points = search_result[0]

        if not cluster_points:
            raise HTTPException(status_code=404, detail=f"No points found in cluster {cluster_id}")

        cluster_embeddings = [point.vector for point in cluster_points if point.vector is not None]
        if not cluster_embeddings:
            raise HTTPException(status_code=404, detail=f"No embeddings found for cluster {cluster_id}")

        cluster_centroid = np.mean(cluster_embeddings, axis=0)

        results, search_metadata = await similarity_search_engine.vector_similarity_search(
            session_id=session_id,
            query_embedding=cluster_centroid,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filter_cluster_id=None,
            use_reduced_embeddings=False,
        )

        return {
            "cluster_id": cluster_id,
            "cluster_size": len(cluster_points),
            "centroid_dimensions": len(cluster_centroid),
            "similar_items": results,
            "search_metadata": search_metadata,
        }
    except HTTPException:
        raise
    except Exception as error:
        logger.error("Error finding cluster similar items", error=str(error))
        raise HTTPException(status_code=500, detail=str(error))


@router.get("/recommendations/trending/{session_id}")
async def get_trending_recommendations(
    session_id: str,
    time_window_hours: int = 24,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Get trending content recommendations based on recent activity patterns."""
    try:
        collection_name = f"session_{session_id}"
        current_time = time.time()
        time_threshold = current_time - (time_window_hours * 3600)

        search_result = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: qdrant_client.scroll(
                collection_name=collection_name,
                limit=1000,
                with_payload=True,
                with_vectors=False,
            ),
        )

        points = search_result[0]

        trending_items: List[Dict[str, Any]] = []
        for point in points:
            timestamp = point.payload.get("timestamp", current_time)
            quality_score = point.payload.get("quality_score", 0.5)
            cluster_id = point.payload.get("cluster_id", -1)

            recency_score = max(0.0, 1 - (current_time - timestamp) / (time_window_hours * 3600))
            trending_score = 0.6 * recency_score + 0.4 * quality_score

            trending_items.append(
                {
                    "point_id": point.id,
                    "title": point.payload.get("title", ""),
                    "url": point.payload.get("url", ""),
                    "cluster_id": cluster_id,
                    "quality_score": quality_score,
                    "trending_score": trending_score,
                    "recency_score": recency_score,
                    "timestamp": timestamp,
                }
            )

        trending_items.sort(key=lambda item: item["trending_score"], reverse=True)
        top_trending = trending_items[:top_k]

        metadata = {
            "session_id": session_id,
            "time_window_hours": time_window_hours,
            "total_items_considered": len(trending_items),
            "top_k": top_k,
            "average_trending_score": float(
                np.mean([item["trending_score"] for item in top_trending])
            )
            if top_trending
            else 0.0,
            "algorithm": "recency_quality_weighted",
        }

        return {"trending_recommendations": top_trending, "metadata": metadata}
    except Exception as error:
        logger.error("Error generating trending recommendations", error=str(error))
        raise HTTPException(status_code=500, detail=str(error))


__all__ = ["router"]
