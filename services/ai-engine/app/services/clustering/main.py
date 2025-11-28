"""Entrypoint for the clustering service."""

from __future__ import annotations

from urllib.robotparser import RobotFileParser  # For consistency with other services/tests

from .app import app, create_app
from .app.clients.qdrant import qdrant_client
from .app.executor import executor
from .app.hdbscan_service import HDBSCANClusterer
from .app.jobs import active_jobs
from .app.models import (
    ClusterValidationMetrics,
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
from .app.routes import router
from .app.similarity import SimilaritySearchEngine
from .app.state import (
    dimensionality_reducer,
    hdbscan_clusterer,
    similarity_search_engine,
    visualization_generator,
)
from .app.umap import DimensionalityReducer, ModelAwareUMAPConfig
from .app.visualization import VisualizationGenerator


# Re-export key classes and singletons for backward compatibility.
__all__ = [
    "app",
    "create_app",
    "router",
    "ModelAwareUMAPConfig",
    "DimensionalityReducer",
    "VisualizationGenerator",
    "HDBSCANClusterer",
    "SimilaritySearchEngine",
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
    "executor",
    "qdrant_client",
    "active_jobs",
    "dimensionality_reducer",
    "hdbscan_clusterer",
    "visualization_generator",
    "similarity_search_engine",
    "RobotFileParser",
]
