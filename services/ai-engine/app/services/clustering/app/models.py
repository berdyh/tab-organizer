"""
Pydantic models shared across the clustering service API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class UMAPRequest(BaseModel):
    session_id: str
    embedding_model: str
    n_components: int = 2
    n_neighbors: Optional[int] = None
    min_dist: Optional[float] = None
    metric: Optional[str] = None
    batch_size: int = 1000
    memory_limit_mb: int = 2048


class UMAPResponse(BaseModel):
    job_id: str
    status: str
    message: str
    reduced_dimensions: Optional[int] = None
    original_dimensions: Optional[int] = None
    n_samples: Optional[int] = None


class VisualizationRequest(BaseModel):
    session_id: str
    plot_type: str = "2d"  # "2d" or "3d"
    color_by: str = "cluster"  # "cluster", "model", "quality"
    include_metrics: bool = True


class HDBSCANRequest(BaseModel):
    session_id: str
    min_cluster_size: Optional[int] = None
    min_samples: Optional[int] = None
    cluster_selection_epsilon: Optional[float] = None
    alpha: Optional[float] = None
    metric: Optional[str] = None
    auto_optimize: bool = True
    use_reduced_embeddings: bool = True


class HDBSCANResponse(BaseModel):
    job_id: str
    status: str
    message: str
    n_clusters: Optional[int] = None
    n_noise_points: Optional[int] = None
    silhouette_score: Optional[float] = None


class ClusterValidationMetrics(BaseModel):
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    n_clusters: int
    n_noise_points: int
    cluster_sizes: List[int]
    stability_score: Optional[float] = None


class SimilaritySearchRequest(BaseModel):
    session_id: str
    query_text: Optional[str] = None
    query_embedding: Optional[List[float]] = None
    top_k: int = 10
    similarity_threshold: float = 0.7
    filter_cluster_id: Optional[int] = None
    use_reduced_embeddings: bool = False


class SimilaritySearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    query_info: Dict[str, Any]
    search_metadata: Dict[str, Any]


class RecommendationRequest(BaseModel):
    session_id: str
    user_interactions: List[Dict[str, Any]]  # List of user interactions (clicks, views, etc.)
    recommendation_type: str = "content_based"  # "content_based", "collaborative", "hybrid"
    top_k: int = 10
    diversity_factor: float = 0.3  # Balance between relevance and diversity


class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    recommendation_metadata: Dict[str, Any]


__all__ = [
    "UMAPRequest",
    "UMAPResponse",
    "VisualizationRequest",
    "HDBSCANRequest",
    "HDBSCANResponse",
    "ClusterValidationMetrics",
    "SimilaritySearchRequest",
    "SimilaritySearchResponse",
    "RecommendationRequest",
    "RecommendationResponse",
]
