"""FastAPI application wiring for the analyzer service."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional, cast

import structlog
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder

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
from .state import AnalyzerState, COMPONENT_NAMES, state as default_state
from .tasks import process_complete_analysis_job, process_embeddings_job
from .text_processing import TextChunker

logger = structlog.get_logger("analyzer_app")


def _require(component: Optional[object], name: str) -> object:
    """Ensure the requested component exists."""
    if component is None:
        raise HTTPException(status_code=503, detail=f"{name} not initialized")
    return component


def create_app(state_instance: AnalyzerState | None = None) -> FastAPI:
    """Create and configure a FastAPI application instance."""
    configure_logging()

    state_ref = state_instance or default_state
    app = FastAPI(
        title="Content Analyzer Service",
        description="Generates embeddings and summaries for scraped content with configurable models",
        version="1.0.0",
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        logger.info("Initializing Content Analyzer Service")
        try:
            state_ref.reset()
            state_ref.hardware_detector = HardwareDetector()
            state_ref.model_manager = ModelManager()
            state_ref.embedding_cache = EmbeddingCache()
            state_ref.text_chunker = TextChunker()
            state_ref.embedding_generator = EmbeddingGenerator(
                state_ref.model_manager,
                state_ref.embedding_cache,
                state_ref.hardware_detector,
            )

            state_ref.ollama_client = OllamaClient()
            await state_ref.ollama_client.initialize()

            state_ref.qdrant_manager = QdrantManager()
            await state_ref.qdrant_manager.initialize()

            state_ref.performance_monitor = PerformanceMonitor()

            logger.info("Content Analyzer Service initialized successfully")
        except Exception as exc:
            logger.error("Failed to initialize Content Analyzer Service", error=str(exc))
            raise

        try:
            yield
        finally:
            logger.info("Shutting down Content Analyzer Service")

    app.router.lifespan_context = lifespan

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        generator = state_ref.embedding_generator
        detector = state_ref.hardware_detector

        try:
            model_info = generator.get_current_model_info() if generator else {}
            hardware_info = detector.detect_hardware() if detector else {}
            return {
                "status": "healthy",
                "service": "analyzer",
                "timestamp": time.time(),
                "current_model": model_info,
                "hardware": {
                    "available_ram_gb": hardware_info.get("available_ram_gb", 0),
                    "ram_usage_percent": hardware_info.get("ram_usage_percent", 0),
                    "has_gpu": hardware_info.get("has_gpu", False),
                },
            }
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Health check failed", error=str(exc))
            return {
                "status": "unhealthy",
                "service": "analyzer",
                "timestamp": time.time(),
                "error": str(exc),
            }

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "service": "Content Analyzer Service",
            "version": "1.0.0",
            "status": "running",
            "features": [
                "Configurable embedding models",
                "Dynamic model switching",
                "Hardware-aware recommendations",
                "Text chunking with overlap",
                "Model-specific caching",
            ],
        }

    @app.get("/hardware", response_model=HardwareInfo)
    async def get_hardware_info():
        """Get current hardware information."""
        detector = _require(state_ref.hardware_detector, "Hardware detector")
        hardware_info = detector.detect_hardware()
        return HardwareInfo(**hardware_info)

    @app.get("/models/available")
    async def get_available_models():
        """Get all available embedding models."""
        manager = _require(state_ref.model_manager, "Model manager")
        return manager.get_available_models()

    @app.get("/models/current")
    async def get_current_model():
        """Get information about currently loaded model."""
        generator = _require(state_ref.embedding_generator, "Embedding generator")
        return generator.get_current_model_info()

    @app.get("/models/recommend", response_model=ModelRecommendation)
    async def get_model_recommendation():
        """Get hardware-based model recommendation."""
        detector = _require(state_ref.hardware_detector, "Hardware detector")
        manager = _require(state_ref.model_manager, "Model manager")
        hardware_info = detector.detect_hardware()
        recommendation = manager.recommend_model(hardware_info)
        return ModelRecommendation(**recommendation)

    @app.post("/models/switch")
    async def switch_model(request: ModelSwitchRequest):
        """Switch to a different embedding model."""
        generator = _require(state_ref.embedding_generator, "Embedding generator")
        success = generator.switch_model(request.embedding_model)

        if success:
            return {
                "status": "success",
                "message": f"Switched to model: {request.embedding_model}",
                "current_model": generator.get_current_model_info(),
            }
        raise HTTPException(status_code=400, detail=f"Failed to switch to model: {request.embedding_model}")

    @app.post("/embeddings/generate", response_model=EmbeddingResponse)
    async def generate_embeddings(request: EmbeddingRequest, background_tasks: BackgroundTasks):
        """Generate embeddings for content items."""
        if not all([state_ref.embedding_generator, state_ref.text_chunker, state_ref.qdrant_manager]):
            raise HTTPException(status_code=503, detail="Services not initialized")

        generator = cast(EmbeddingGenerator, state_ref.embedding_generator)

        job_id = str(uuid.uuid4())

        if request.embedding_model:
            success = generator.switch_model(request.embedding_model)
            if not success:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to switch to model: {request.embedding_model}",
                )

        background_tasks.add_task(
            process_embeddings_job,
            state_ref,
            job_id,
            request.content_items,
            request.session_id,
            request.chunk_size,
            request.chunk_overlap,
        )

        return EmbeddingResponse(
            job_id=job_id,
            status="processing",
            message=f"Started embedding generation for {len(request.content_items)} items",
        )

    @app.get("/embeddings/status/{job_id}")
    async def get_embedding_status(job_id: str):
        """Get status of embedding generation job."""
        return {
            "job_id": job_id,
            "status": "completed",
            "message": "Embedding generation completed",
        }

    @app.post("/analysis/complete", response_model=AnalysisResponse)
    async def analyze_content(request: AnalysisRequest, background_tasks: BackgroundTasks):
        """Perform complete content analysis including embeddings, summaries, and quality assessment."""
        _require(state_ref.embedding_generator, "Embedding generator")
        _require(state_ref.text_chunker, "Text chunker")
        _require(state_ref.ollama_client, "Ollama client")
        _require(state_ref.qdrant_manager, "Qdrant manager")

        job_id = str(uuid.uuid4())

        if request.embedding_model:
            generator = state_ref.embedding_generator
            if generator and not generator.switch_model(request.embedding_model):
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to switch to embedding model: {request.embedding_model}",
                )

        background_tasks.add_task(
            process_complete_analysis_job,
            state_ref,
            job_id,
            request.content_items,
            request.session_id,
            request.llm_model or "llama3.2:3b",
            request.generate_summary,
            request.extract_keywords,
            request.assess_quality,
        )

        return AnalysisResponse(
            job_id=job_id,
            status="processing",
            message=f"Started complete analysis for {len(request.content_items)} items",
        )

    @app.get("/llm/models/available")
    async def get_available_llm_models():
        """Get all available LLM models from Ollama."""
        ollama = _require(state_ref.ollama_client, "Ollama client")
        return {
            "available_models": ollama.available_models,
            "fallback_chains": ollama.fallback_chains,
        }

    @app.post("/llm/models/ensure/{model_id}")
    async def ensure_llm_model(model_id: str):
        """Ensure LLM model is available, pull if necessary."""
        ollama = _require(state_ref.ollama_client, "Ollama client")
        success = await ollama.ensure_model_available(model_id)

        if success:
            return {
                "status": "success",
                "message": f"Model {model_id} is available",
                "available_models": ollama.available_models,
            }
        raise HTTPException(status_code=400, detail=f"Failed to ensure model availability: {model_id}")

    @app.get("/qdrant/collections/{collection_name}/info", response_model=QdrantConnectionInfo)
    async def get_collection_info(collection_name: str):
        """Get information about a Qdrant collection."""
        qdrant = _require(state_ref.qdrant_manager, "Qdrant manager")
        try:
            info = await qdrant.get_collection_info(collection_name)
            return QdrantConnectionInfo(
                host=qdrant.host,
                port=qdrant.port,
                collection_name=info["collection_name"],
                vector_size=info["vector_size"],
                distance_metric=info["distance_metric"],
                points_count=info["points_count"],
            )
        except Exception as exc:
            raise HTTPException(status_code=404, detail=f"Collection not found or error: {str(exc)}") from exc

    @app.get("/qdrant/collections/{collection_name}/search")
    async def search_collection(
        collection_name: str,
        query_text: str,
        limit: int = 10,
        embedding_model_filter: Optional[str] = None,
    ):
        """Search for similar content in a collection."""
        qdrant = _require(state_ref.qdrant_manager, "Qdrant manager")
        generator = _require(state_ref.embedding_generator, "Embedding generator")

        try:
            query_embeddings = generator.generate_embeddings([query_text])
            if not query_embeddings:
                raise HTTPException(status_code=400, detail="Failed to generate query embedding")

            results = await qdrant.search_similar_content(
                collection_name=collection_name,
                query_vector=query_embeddings[0].tolist(),
                limit=limit,
                embedding_model_filter=embedding_model_filter,
            )

            return {
                "collection": collection_name,
                "query": query_text,
                "results": results,
                "embedding_model_used": generator.current_model_id,
                "used_model": generator.current_model_id,
            }
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Collection search failed",
                collection_name=collection_name,
                error=str(exc),
            )
            raise HTTPException(status_code=500, detail="Search failed") from exc

    @app.get("/performance/models")
    async def get_model_performance():
        """Get performance metrics for all models."""
        monitor = _require(state_ref.performance_monitor, "Performance monitor")
        metrics = monitor.get_all_metrics()
        summary = monitor.get_resource_summary()
        encoded_metrics = []
        for metric in metrics:
            if hasattr(metric, "model_dump"):
                encoded_metrics.append(metric.model_dump())
            else:
                encoded_metrics.append(
                    {
                        "model_id": getattr(metric, "model_id", None),
                        "model_type": getattr(metric, "model_type", None),
                        "total_requests": getattr(metric, "total_requests", None),
                        "successful_requests": getattr(metric, "successful_requests", None),
                        "failed_requests": getattr(metric, "failed_requests", None),
                        "average_response_time": getattr(metric, "average_response_time", None),
                        "average_tokens_per_second": getattr(metric, "average_tokens_per_second", None),
                    }
                )

        return {
            "model_metrics": encoded_metrics,
            "system_resources": summary,
            "resource_summary": summary,
        }

    @app.get("/performance/models/{model_id}", response_model=ModelPerformanceMetrics)
    async def get_model_metrics(model_id: str):
        """Get performance metrics for a specific model."""
        monitor = _require(state_ref.performance_monitor, "Performance monitor")
        metrics = monitor.get_model_metrics(model_id)
        if not metrics:
            raise HTTPException(status_code=404, detail=f"No performance data for model: {model_id}")
        return metrics

    @app.post("/performance/record-resources")
    async def record_system_resources():
        """Manually trigger system resource recording."""
        monitor = _require(state_ref.performance_monitor, "Performance monitor")
        monitor.record_system_resources()
        return {
            "status": "success",
            "message": "System resources recorded",
            "summary": monitor.get_resource_summary(),
        }

    return app


app = create_app()


def get_state() -> AnalyzerState:
    """Return the shared analyzer state instance."""
    return default_state


__all__ = [
    "COMPONENT_NAMES",
    "create_app",
    "get_state",
    "app",
]
