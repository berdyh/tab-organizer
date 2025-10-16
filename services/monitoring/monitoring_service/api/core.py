"""
Primary API routes for the monitoring service.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from ..config import MonitoringSettings
from ..core.components import MonitoringComponents
from ..core.system import collect_system_metrics
from ..instrumentation import (
    CONTAINER_CPU_USAGE,
    CONTAINER_MEMORY_USAGE,
    SERVICE_HEALTH_STATUS,
    metrics_endpoint,
    ALERT_COUNT,
)
from ..logging import get_logger
from ..metrics.collector import MetricsCollector
from ..performance.tracker import PerformanceTracker
from ..alerts.manager import AlertManager
from ..tracing.distributed import DistributedTracer
from ..health.monitor import HealthMonitor
from .dependencies import (
    get_alert_manager,
    get_health_monitor,
    get_components,
    get_metrics_collector,
    get_performance_tracker,
    get_tracer,
    maybe_get_health_monitor,
)
from .models import (
    AlertResponse,
    HealthStatus,
    MetricsResponse,
    PerformanceReport,
    TraceResponse,
)

logger = get_logger("api.core")
router = APIRouter()


@router.get("/health", response_model=HealthStatus, tags=["Health"])
async def health_check(
    health_monitor: Optional[HealthMonitor] = Depends(maybe_get_health_monitor),
) -> HealthStatus:
    """Get comprehensive health status of all services and system."""
    if not health_monitor:
        return HealthStatus(
            status="unavailable",
            timestamp=time.time(),
            services={},
            system_metrics=await collect_system_metrics(),
        )

    health_status = await health_monitor.get_comprehensive_health()
    system_metrics = await collect_system_metrics()

    return HealthStatus(
        status=health_status.get("status", "unknown"),
        timestamp=time.time(),
        services=health_status.get("services", {}),
        system_metrics=system_metrics,
    )


@router.get("/health/simple", tags=["Health"])
async def simple_health_check() -> Dict[str, Any]:
    """Fast path health check for container orchestration."""
    return {"status": "ok", "timestamp": time.time()}


@router.get("/health/services", tags=["Health"])
async def services_health(
    health_monitor: HealthMonitor = Depends(get_health_monitor),
) -> Dict[str, Any]:
    """Get health status of all monitored services."""
    services_status = await health_monitor.check_all_services()

    for service_name, status in services_status.items():
        SERVICE_HEALTH_STATUS.labels(service=service_name).set(
            1 if status.get("healthy", False) else 0
        )

    return {"services": services_status, "timestamp": time.time()}


@router.get("/metrics", tags=["Metrics"])
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    return metrics_endpoint()


@router.get("/metrics/detailed", response_model=MetricsResponse, tags=["Metrics"])
async def detailed_metrics(
    metrics_collector: MetricsCollector = Depends(get_metrics_collector),
) -> MetricsResponse:
    """Get detailed metrics from all services."""
    metrics = await metrics_collector.collect_all_metrics()
    return MetricsResponse(
        metrics=metrics,
        timestamp=time.time(),
        collection_interval=metrics_collector.collection_interval,
    )


@router.get("/metrics/system", tags=["Metrics"])
async def system_metrics() -> Dict[str, Any]:
    """Get system-level metrics (CPU, memory, disk, network)."""
    return {"system_metrics": await collect_system_metrics(), "timestamp": time.time()}


@router.get("/metrics/containers", tags=["Metrics"])
async def container_metrics(
    metrics_collector: MetricsCollector = Depends(get_metrics_collector),
) -> Dict[str, Any]:
    """Get Docker container metrics."""
    container_metrics = await metrics_collector.collect_container_metrics()

    for container_name, metrics in container_metrics.items():
        if "cpu_percent" in metrics:
            CONTAINER_CPU_USAGE.labels(container=container_name).set(
                metrics["cpu_percent"]
            )
        if "memory_usage" in metrics:
            CONTAINER_MEMORY_USAGE.labels(container=container_name).set(
                metrics["memory_usage"]
            )

    return {"container_metrics": container_metrics, "timestamp": time.time()}


@router.get("/alerts", response_model=AlertResponse, tags=["Alerts"])
async def get_alerts(
    alert_manager: AlertManager = Depends(get_alert_manager),
) -> AlertResponse:
    """Get all alerts with filtering options."""
    alerts = await alert_manager.get_all_alerts()
    active_alerts = [alert for alert in alerts if alert.get("status") == "active"]

    for alert in alerts:
        ALERT_COUNT.labels(
            severity=alert.get("severity", "unknown"),
            service=alert.get("service") or "unknown",
        ).inc(0)  # ensure labels are registered without incrementing

    return AlertResponse(
        alerts=alerts,
        total_count=len(alerts),
        active_count=len(active_alerts),
        timestamp=time.time(),
    )


@router.get("/alerts/active", tags=["Alerts"])
async def get_active_alerts(
    alert_manager: AlertManager = Depends(get_alert_manager),
) -> Dict[str, Any]:
    """Get only active alerts."""
    active_alerts = await alert_manager.get_active_alerts()
    return {"alerts": active_alerts, "count": len(active_alerts), "timestamp": time.time()}


@router.post("/alerts/{alert_id}/acknowledge", tags=["Alerts"])
async def acknowledge_alert(
    alert_id: str, alert_manager: AlertManager = Depends(get_alert_manager)
) -> Dict[str, Any]:
    """Acknowledge an alert."""
    success = await alert_manager.acknowledge_alert(alert_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Alert {alert_id} not found"
        )
    return {"message": f"Alert {alert_id} acknowledged", "timestamp": time.time()}


@router.get("/performance", response_model=PerformanceReport, tags=["Performance"])
async def get_performance_report(
    performance_tracker: PerformanceTracker = Depends(get_performance_tracker),
) -> PerformanceReport:
    """Get comprehensive performance report with optimization recommendations."""
    performance_report = await performance_tracker.generate_performance_report()

    return PerformanceReport(
        service_performance=performance_report.get("services", {}),
        system_performance=performance_report.get("system", {}),
        recommendations=performance_report.get("recommendations", []),
        timestamp=time.time(),
    )


@router.get("/performance/benchmarks", tags=["Performance"])
async def run_performance_benchmarks(
    performance_tracker: PerformanceTracker = Depends(get_performance_tracker),
) -> Dict[str, Any]:
    """Run performance benchmarks on all services."""
    benchmark_results = await performance_tracker.run_benchmarks()
    return {"benchmarks": benchmark_results, "timestamp": time.time()}


@router.get("/traces", response_model=TraceResponse, tags=["Tracing"])
async def get_traces(
    tracer: DistributedTracer = Depends(get_tracer),
) -> TraceResponse:
    """Get distributed traces for request flow monitoring."""
    traces = await tracer.get_traces()
    return TraceResponse(traces=traces, total_count=len(traces), timestamp=time.time())


@router.get("/traces/{trace_id}", tags=["Tracing"])
async def get_trace_details(
    trace_id: str, tracer: DistributedTracer = Depends(get_tracer)
) -> Dict[str, Any]:
    """Get detailed information for a specific trace."""
    trace_details = await tracer.get_trace_details(trace_id)
    if not trace_details:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Trace {trace_id} not found"
        )
    return {"trace": trace_details, "timestamp": time.time()}


@router.get("/traces/{trace_id}/spans", tags=["Tracing"])
async def get_trace_spans(
    trace_id: str, tracer: DistributedTracer = Depends(get_tracer)
) -> Dict[str, Any]:
    """Get spans for a specific trace."""
    spans = await tracer.get_trace_spans(trace_id)
    if spans is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Trace {trace_id} not found"
        )
    return {"spans": spans, "count": len(spans), "timestamp": time.time()}


@router.get("/logs/{service_name}", tags=["Logs"])
async def get_service_logs(service_name: str) -> Dict[str, Any]:
    """Get logs for a specific service."""
    return {
        "service": service_name,
        "message": "Service log endpoint - integrate with your logging system",
        "timestamp": time.time(),
    }


@router.get("/config", tags=["Configuration"])
async def get_monitoring_config() -> Dict[str, Any]:
    """Get current monitoring configuration."""
    settings = MonitoringSettings()
    return {
        "config": {
            "collection_interval": settings.collection_interval,
            "alert_thresholds": settings.alert_thresholds,
            "performance_tracking": settings.performance_tracking_enabled,
            "distributed_tracing": settings.distributed_tracing_enabled,
            "log_level": settings.log_level,
        },
        "timestamp": time.time(),
    }


@router.post("/config/reload", tags=["Configuration"])
async def reload_monitoring_config(
    components: MonitoringComponents = Depends(get_components),
) -> Dict[str, Any]:
    """Reload monitoring configuration."""
    await components.reload()
    return {"message": "Configuration reloaded successfully", "timestamp": time.time()}


@router.get("/", tags=["Metadata"])
async def root(components: MonitoringComponents = Depends(get_components)) -> Dict[str, Any]:
    """Root endpoint with monitoring service information."""
    return {
        "name": "Web Scraping Tool - Monitoring Service",
        "version": "1.0.0",
        "status": "running",
        "timestamp": time.time(),
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "alerts": "/alerts",
            "performance": "/performance",
            "traces": "/traces",
            "logs": "/logs",
            "config": "/config",
        },
        "features": {
            "metrics_collection": components.metrics is not None,
            "health_monitoring": components.health is not None,
            "alerting": components.alerts is not None,
            "performance_tracking": components.performance is not None,
            "distributed_tracing": components.tracing is not None,
        },
    }
