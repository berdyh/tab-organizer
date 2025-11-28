"""
Centralised state container for clustering service singletons.
"""

from __future__ import annotations

from .clients.qdrant import qdrant_client
from .executor import executor
from .hdbscan_service import HDBSCANClusterer
from .jobs import active_jobs
from .similarity import SimilaritySearchEngine
from .umap import DimensionalityReducer
from .visualization import VisualizationGenerator

dimensionality_reducer = DimensionalityReducer()
visualization_generator = VisualizationGenerator()
hdbscan_clusterer = HDBSCANClusterer()
similarity_search_engine = SimilaritySearchEngine()

__all__ = [
    "qdrant_client",
    "executor",
    "dimensionality_reducer",
    "visualization_generator",
    "hdbscan_clusterer",
    "similarity_search_engine",
    "active_jobs",
]
