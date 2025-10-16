"""
Pydantic response models shared across monitoring endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel


class HealthStatus(BaseModel):
    status: str
    timestamp: float
    services: Dict[str, Dict[str, Any]]
    system_metrics: Dict[str, Any]


class MetricsResponse(BaseModel):
    metrics: Dict[str, Any]
    timestamp: float
    collection_interval: int


class AlertResponse(BaseModel):
    alerts: List[Dict[str, Any]]
    total_count: int
    active_count: int
    timestamp: float


class PerformanceReport(BaseModel):
    service_performance: Dict[str, Dict[str, Any]]
    system_performance: Dict[str, Any]
    recommendations: List[str]
    timestamp: float


class TraceResponse(BaseModel):
    traces: List[Dict[str, Any]]
    total_count: int
    timestamp: float
