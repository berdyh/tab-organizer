"""
Compatibility entry point for the monitoring service.

The actual implementation now lives inside the ``monitoring_service`` package.
This module remains so existing tooling that imports ``services.monitoring.main``
continues to function without modification.
"""

from __future__ import annotations

from .monitoring_service import (
    ALERT_COUNT,
    CONTAINER_CPU_USAGE,
    CONTAINER_MEMORY_USAGE,
    MONITORING_DURATION,
    MONITORING_REQUESTS,
    MonitoringComponents,
    MonitoringSettings,
    SERVICE_HEALTH_STATUS,
    create_app,
    get_app,
)
from .monitoring_service.alerts.manager import AlertManager
from .monitoring_service.health.monitor import HealthMonitor
from .monitoring_service.instrumentation import metrics_endpoint
from .monitoring_service.logging import get_logger, setup_monitoring_logging
from .monitoring_service.metrics.collector import MetricsCollector
from .monitoring_service.performance.tracker import PerformanceTracker
from .monitoring_service.tracing.distributed import DistributedTracer

app = create_app()

__all__ = [
    "ALERT_COUNT",
    "AlertManager",
    "CONTAINER_CPU_USAGE",
    "CONTAINER_MEMORY_USAGE",
    "DistributedTracer",
    "HealthMonitor",
    "MetricsCollector",
    "MonitoringComponents",
    "MonitoringSettings",
    "MONITORING_DURATION",
    "MONITORING_REQUESTS",
    "PerformanceTracker",
    "SERVICE_HEALTH_STATUS",
    "app",
    "create_app",
    "get_app",
    "get_logger",
    "metrics_endpoint",
    "setup_monitoring_logging",
]


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8091)
