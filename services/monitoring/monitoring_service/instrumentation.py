"""
Prometheus instrumentation helpers for the monitoring service.
"""

from __future__ import annotations

import time
from typing import Callable

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    CONTENT_TYPE_LATEST,
    REGISTRY,
    generate_latest,
)
from fastapi.responses import PlainTextResponse

from .logging import get_logger

logger = get_logger("instrumentation")

# Core metrics exposed for shared use throughout the service.
MONITORING_REQUESTS = Counter(
    "monitoring_requests_total",
    "Total monitoring requests",
    ["endpoint", "status"],
)
MONITORING_DURATION = Histogram(
    "monitoring_request_duration_seconds", "Monitoring request duration"
)
SERVICE_HEALTH_STATUS = Gauge(
    "service_health_status",
    "Service health status (1=healthy, 0=unhealthy)",
    ["service"],
)
CONTAINER_CPU_USAGE = Gauge(
    "container_cpu_usage_percent",
    "Container CPU usage percentage",
    ["container"],
)
CONTAINER_MEMORY_USAGE = Gauge(
    "container_memory_usage_bytes",
    "Container memory usage in bytes",
    ["container"],
)
ALERT_COUNT = Counter(
    "alerts_total",
    "Total alerts generated",
    ["severity", "service"],
)


def register_request_middleware(app: FastAPI) -> None:
    """Attach instrumentation middleware to track request metrics."""

    @app.middleware("http")
    async def _track_requests(request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        response = await call_next(request)

        duration = time.time() - start_time
        MONITORING_REQUESTS.labels(
            endpoint=request.url.path,
            status=response.status_code,
        ).inc()
        MONITORING_DURATION.observe(duration)
        logger.info(
            "Monitoring request processed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration=duration,
        )

        return response


def metrics_endpoint() -> PlainTextResponse:
    """Return a response compatible with Prometheus scraping."""
    return PlainTextResponse(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
