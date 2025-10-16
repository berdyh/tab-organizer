"""
Reusable FastAPI dependencies for accessing monitoring components.
"""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, Request, status

from ..alerts.manager import AlertManager
from ..core.components import MonitoringComponents
from ..metrics.collector import MetricsCollector
from ..health.monitor import HealthMonitor
from ..performance.tracker import PerformanceTracker
from ..tracing.distributed import DistributedTracer


def get_components(request: Request) -> MonitoringComponents:
    components = getattr(request.app.state, "components", None)
    if not components:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Monitoring service components unavailable",
        )
    return components


def get_metrics_collector(
    components: MonitoringComponents = Depends(get_components),
) -> MetricsCollector:
    if not components.metrics:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Metrics collector not initialised",
        )
    return components.metrics


def get_health_monitor(
    components: MonitoringComponents = Depends(get_components),
) -> HealthMonitor:
    if not components.health:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Health monitor not initialised",
        )
    return components.health


def maybe_get_health_monitor(
    components: MonitoringComponents = Depends(get_components),
) -> Optional[HealthMonitor]:
    return components.health


def get_alert_manager(
    components: MonitoringComponents = Depends(get_components),
) -> AlertManager:
    if not components.alerts:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Alert manager not initialised",
        )
    return components.alerts


def get_performance_tracker(
    components: MonitoringComponents = Depends(get_components),
) -> PerformanceTracker:
    if not components.performance:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Performance tracker not initialised",
        )
    return components.performance


def get_tracer(
    components: MonitoringComponents = Depends(get_components),
) -> DistributedTracer:
    if not components.tracing:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Distributed tracer not initialised",
        )
    return components.tracing
