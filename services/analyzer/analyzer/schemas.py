"""Pydantic models used by the analyzer service."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class ContentItem(BaseModel):
    """Content item submitted for analysis."""

    id: str
    content: str
    title: Optional[str] = None
    url: Optional[HttpUrl] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmbeddingRequest(BaseModel):
    """Request payload for background embedding generation."""

    content_items: List[ContentItem]
    session_id: str
    embedding_model: Optional[str] = None
    chunk_size: int = Field(default=512, ge=100, le=2048)
    chunk_overlap: int = Field(default=50, ge=0, le=200)


class EmbeddingResponse(BaseModel):
    """Response body for embedding job submission."""

    job_id: str
    status: str
    message: str


class ModelSwitchRequest(BaseModel):
    """Request payload for switching embedding model."""

    embedding_model: str


class HardwareInfo(BaseModel):
    """Summary of detected hardware capabilities."""

    ram_gb: float
    cpu_count: int
    has_gpu: bool
    gpu_memory_gb: float
    gpu_name: str
    available_ram_gb: float
    ram_usage_percent: float


class ModelRecommendation(BaseModel):
    """Recommended embedding model selection details."""

    recommended_model: str
    reason: str
    alternatives: List[str]
    performance_estimate: Dict[str, Any]


class AnalysisRequest(BaseModel):
    """Request payload for complete analysis jobs."""

    content_items: List[ContentItem]
    session_id: str
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    generate_summary: bool = True
    extract_keywords: bool = True
    assess_quality: bool = True

    def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        kwargs.setdefault("mode", "json")
        data = super().model_dump(*args, **kwargs)
        data["content_items"] = [item.model_dump(mode="json") for item in self.content_items]
        return data

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[override]
        return self.model_dump(*args, **kwargs)


class AnalysisResponse(BaseModel):
    """Response body for analysis job submission."""

    job_id: str
    status: str
    message: str


class ModelPerformanceMetrics(BaseModel):
    """Performance metrics aggregated per model."""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    model_type: str  # "llm" or "embedding"
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    average_tokens_per_second: Optional[float] = None
    last_used: datetime
    resource_usage: Dict[str, float]


class QdrantConnectionInfo(BaseModel):
    """Metadata describing a Qdrant collection."""

    host: str
    port: int
    collection_name: str
    vector_size: int
    distance_metric: str
    points_count: int


__all__ = [
    "AnalysisRequest",
    "AnalysisResponse",
    "ContentItem",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "HardwareInfo",
    "ModelPerformanceMetrics",
    "ModelRecommendation",
    "ModelSwitchRequest",
    "QdrantConnectionInfo",
]
