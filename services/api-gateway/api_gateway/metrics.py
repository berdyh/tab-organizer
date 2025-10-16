"""Prometheus metrics used by the API Gateway."""

import time
from typing import Optional

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest


REQUEST_COUNT = Counter(
    "api_gateway_requests_total",
    "Total API requests",
    labelnames=["method", "endpoint", "status"],
)
REQUEST_DURATION = Histogram(
    "api_gateway_request_duration_seconds", "Request duration in seconds"
)


def observe_request(
    method: str,
    endpoint: str,
    status_code: int,
    start_time: float,
) -> float:
    """Update counters for a processed request."""
    duration = time.time() - start_time
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status_code).inc()
    REQUEST_DURATION.observe(duration)
    return duration


def latest_metrics() -> bytes:
    """Return raw metrics to expose on /metrics."""
    return generate_latest()


__all__ = [
    "REQUEST_COUNT",
    "REQUEST_DURATION",
    "CONTENT_TYPE_LATEST",
    "observe_request",
    "latest_metrics",
]

