"""Analyzer service package exports."""

from .app import app, create_app, get_state
from .cache import EmbeddingCache
from .embeddings import EmbeddingGenerator
from .hardware import HardwareDetector
from .logging import configure_logging
from .model_management import ModelManager
from .ollama_client import OllamaClient
from .performance import PerformanceMonitor
from .qdrant_manager import QdrantManager
from .schemas import (
    AnalysisRequest,
    AnalysisResponse,
    ContentItem,
    EmbeddingRequest,
    EmbeddingResponse,
    HardwareInfo,
    ModelPerformanceMetrics,
    ModelRecommendation,
    ModelSwitchRequest,
    QdrantConnectionInfo,
)
from .state import COMPONENT_NAMES, AnalyzerState, state
from .tasks import process_complete_analysis_job, process_embeddings_job
from .text_processing import TextChunker

__all__ = [
    "AnalysisRequest",
    "AnalysisResponse",
    "AnalyzerState",
    "COMPONENT_NAMES",
    "ContentItem",
    "EmbeddingCache",
    "EmbeddingGenerator",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "HardwareDetector",
    "HardwareInfo",
    "ModelManager",
    "ModelPerformanceMetrics",
    "ModelRecommendation",
    "ModelSwitchRequest",
    "OllamaClient",
    "PerformanceMonitor",
    "QdrantConnectionInfo",
    "QdrantManager",
    "TextChunker",
    "app",
    "configure_logging",
    "create_app",
    "get_state",
    "process_complete_analysis_job",
    "process_embeddings_job",
    "state",
]
