"""
Monitoring Service - Comprehensive monitoring, logging, and performance optimization.
Provides centralized monitoring for all containerized services with metrics collection,
alerting, distributed tracing, and performance benchmarks.
"""

import os
import time
import asyncio
import json
import psutil
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import httpx
import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, 
    CONTENT_TYPE_LATEST, CollectorRegistry, REGISTRY
)

from config import MonitoringSettings
from logging_config import setup_monitoring_logging
from metrics_collector import MetricsCollector
from health_monitor import HealthMonitor
from alert_manager import AlertManager
from performance_tracker import PerformanceTracker
from distributed_tracer import DistributedTracer

# Initialize logging
setup_monitoring_logging()
logger = structlog.get_logger()

# Prometheus metrics
MONITORING_REQUESTS = Counter('monitoring_requests_total', 'Total monitoring requests', ['endpoint', 'status'])
MONITORING_DURATION = Histogram('monitoring_request_duration_seconds', 'Monitoring request duration')
SERVICE_HEALTH_STATUS = Gauge('service_health_status', 'Service health status (1=healthy, 0=unhealthy)', ['service'])
CONTAINER_CPU_USAGE = Gauge('container_cpu_usage_percent', 'Container CPU usage percentage', ['container'])
CONTAINER_MEMORY_USAGE = Gauge('container_memory_usage_bytes', 'Container memory usage in bytes', ['container'])
ALERT_COUNT = Counter('alerts_total', 'Total alerts generated', ['severity', 'service'])

# Global components
metrics_collector: Optional[MetricsCollector] = None
health_monitor: Optional[HealthMonitor] = None
alert_manager: Optional[AlertManager] = None
performance_tracker: Optional[PerformanceTracker] = None
distributed_tracer: Optional[DistributedTracer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global metrics_collector, health_monitor, alert_manager, performance_tracker, distributed_tracer
    
    logger.info("Starting Monitoring Service...")
    
    # Initialize settings
    settings = MonitoringSettings()
    
    # Initialize components
    metrics_collector = MetricsCollector(settings)
    health_monitor = HealthMonitor(settings)
    alert_manager = AlertManager(settings)
    performance_tracker = PerformanceTracker(settings)
    distributed_tracer = DistributedTracer(settings)
    
    # Start background tasks
    monitoring_task = asyncio.create_task(_start_monitoring_tasks())
    
    logger.info("Monitoring Service started successfully")
    
    yield
    
    # Cleanup
    logger.info("Shutting down Monitoring Service...")
    monitoring_task.cancel()
    
    try:
        await monitoring_task
    except asyncio.CancelledError:
        pass
    
    if metrics_collector:
        await metrics_collector.close()
    if health_monitor:
        await health_monitor.close()
    if alert_manager:
        await alert_manager.close()
    if performance_tracker:
        await performance_tracker.close()
    if distributed_tracer:
        await distributed_tracer.close()
    
    logger.info("Monitoring Service shutdown complete")


async def _start_monitoring_tasks():
    """Start all background monitoring tasks."""
    tasks = []
    
    if metrics_collector:
        tasks.append(asyncio.create_task(metrics_collector.start_collection()))
    
    if health_monitor:
        tasks.append(asyncio.create_task(health_monitor.start_monitoring()))
    
    if alert_manager:
        tasks.append(asyncio.create_task(alert_manager.start_alert_processing()))
    
    if performance_tracker:
        tasks.append(asyncio.create_task(performance_tracker.start_tracking()))
    
    if distributed_tracer:
        tasks.append(asyncio.create_task(distributed_tracer.start_tracing()))
    
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        raise


# Create FastAPI app
app = FastAPI(
    title="Web Scraping Tool - Monitoring Service",
    description="Comprehensive monitoring, logging, and performance optimization service",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
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


# Middleware for request tracking
@app.middleware("http")
async def track_requests(request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    
    # Update metrics
    MONITORING_REQUESTS.labels(
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    MONITORING_DURATION.observe(duration)
    
    # Log request
    logger.info(
        "Monitoring request processed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration=duration
    )
    
    return response


# Health endpoints
@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Get comprehensive health status of all services and system."""
    if not health_monitor:
        raise HTTPException(status_code=503, detail="Health monitor not initialized")
    
    try:
        health_status = await health_monitor.get_comprehensive_health()
        system_metrics = await _get_system_metrics()
        
        return HealthStatus(
            status=health_status["status"],
            timestamp=time.time(),
            services=health_status["services"],
            system_metrics=system_metrics
        )
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


@app.get("/health/services")
async def services_health():
    """Get health status of all monitored services."""
    if not health_monitor:
        raise HTTPException(status_code=503, detail="Health monitor not initialized")
    
    try:
        services_status = await health_monitor.check_all_services()
        
        # Update Prometheus metrics
        for service_name, status in services_status.items():
            SERVICE_HEALTH_STATUS.labels(service=service_name).set(
                1 if status.get("healthy", False) else 0
            )
        
        return {"services": services_status, "timestamp": time.time()}
    except Exception as e:
        logger.error("Services health check failed", error=str(e))
        raise HTTPException(status_code=503, detail=f"Services health check failed: {str(e)}")


# Metrics endpoints
@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.get("/metrics/detailed", response_model=MetricsResponse)
async def detailed_metrics():
    """Get detailed metrics from all services."""
    if not metrics_collector:
        raise HTTPException(status_code=503, detail="Metrics collector not initialized")
    
    try:
        metrics = await metrics_collector.collect_all_metrics()
        
        return MetricsResponse(
            metrics=metrics,
            timestamp=time.time(),
            collection_interval=metrics_collector.collection_interval
        )
    except Exception as e:
        logger.error("Failed to collect detailed metrics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Metrics collection failed: {str(e)}")


@app.get("/metrics/system")
async def system_metrics():
    """Get system-level metrics (CPU, memory, disk, network)."""
    try:
        system_metrics = await _get_system_metrics()
        return {"system_metrics": system_metrics, "timestamp": time.time()}
    except Exception as e:
        logger.error("Failed to collect system metrics", error=str(e))
        raise HTTPException(status_code=500, detail=f"System metrics collection failed: {str(e)}")


@app.get("/metrics/containers")
async def container_metrics():
    """Get Docker container metrics."""
    if not metrics_collector:
        raise HTTPException(status_code=503, detail="Metrics collector not initialized")
    
    try:
        container_metrics = await metrics_collector.collect_container_metrics()
        
        # Update Prometheus metrics
        for container_name, metrics in container_metrics.items():
            if "cpu_percent" in metrics:
                CONTAINER_CPU_USAGE.labels(container=container_name).set(metrics["cpu_percent"])
            if "memory_usage" in metrics:
                CONTAINER_MEMORY_USAGE.labels(container=container_name).set(metrics["memory_usage"])
        
        return {"container_metrics": container_metrics, "timestamp": time.time()}
    except Exception as e:
        logger.error("Failed to collect container metrics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Container metrics collection failed: {str(e)}")


# Alerting endpoints
@app.get("/alerts", response_model=AlertResponse)
async def get_alerts():
    """Get all alerts with filtering options."""
    if not alert_manager:
        raise HTTPException(status_code=503, detail="Alert manager not initialized")
    
    try:
        alerts = await alert_manager.get_all_alerts()
        active_alerts = [alert for alert in alerts if alert.get("status") == "active"]
        
        return AlertResponse(
            alerts=alerts,
            total_count=len(alerts),
            active_count=len(active_alerts),
            timestamp=time.time()
        )
    except Exception as e:
        logger.error("Failed to get alerts", error=str(e))
        raise HTTPException(status_code=500, detail=f"Alert retrieval failed: {str(e)}")


@app.get("/alerts/active")
async def get_active_alerts():
    """Get only active alerts."""
    if not alert_manager:
        raise HTTPException(status_code=503, detail="Alert manager not initialized")
    
    try:
        active_alerts = await alert_manager.get_active_alerts()
        return {"alerts": active_alerts, "count": len(active_alerts), "timestamp": time.time()}
    except Exception as e:
        logger.error("Failed to get active alerts", error=str(e))
        raise HTTPException(status_code=500, detail=f"Active alerts retrieval failed: {str(e)}")


@app.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert."""
    if not alert_manager:
        raise HTTPException(status_code=503, detail="Alert manager not initialized")
    
    try:
        success = await alert_manager.acknowledge_alert(alert_id)
        if success:
            return {"message": f"Alert {alert_id} acknowledged", "timestamp": time.time()}
        else:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    except Exception as e:
        logger.error("Failed to acknowledge alert", alert_id=alert_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Alert acknowledgment failed: {str(e)}")


# Performance endpoints
@app.get("/performance", response_model=PerformanceReport)
async def get_performance_report():
    """Get comprehensive performance report with optimization recommendations."""
    if not performance_tracker:
        raise HTTPException(status_code=503, detail="Performance tracker not initialized")
    
    try:
        performance_report = await performance_tracker.generate_performance_report()
        
        return PerformanceReport(
            service_performance=performance_report["services"],
            system_performance=performance_report["system"],
            recommendations=performance_report["recommendations"],
            timestamp=time.time()
        )
    except Exception as e:
        logger.error("Failed to generate performance report", error=str(e))
        raise HTTPException(status_code=500, detail=f"Performance report generation failed: {str(e)}")


@app.get("/performance/benchmarks")
async def run_performance_benchmarks():
    """Run performance benchmarks on all services."""
    if not performance_tracker:
        raise HTTPException(status_code=503, detail="Performance tracker not initialized")
    
    try:
        benchmark_results = await performance_tracker.run_benchmarks()
        return {"benchmarks": benchmark_results, "timestamp": time.time()}
    except Exception as e:
        logger.error("Failed to run performance benchmarks", error=str(e))
        raise HTTPException(status_code=500, detail=f"Performance benchmarks failed: {str(e)}")


# Distributed tracing endpoints
@app.get("/traces", response_model=TraceResponse)
async def get_traces():
    """Get distributed traces for request flow monitoring."""
    if not distributed_tracer:
        raise HTTPException(status_code=503, detail="Distributed tracer not initialized")
    
    try:
        traces = await distributed_tracer.get_traces()
        
        return TraceResponse(
            traces=traces,
            total_count=len(traces),
            timestamp=time.time()
        )
    except Exception as e:
        logger.error("Failed to get traces", error=str(e))
        raise HTTPException(status_code=500, detail=f"Trace retrieval failed: {str(e)}")


@app.get("/traces/{trace_id}")
async def get_trace_details(trace_id: str):
    """Get detailed information for a specific trace."""
    if not distributed_tracer:
        raise HTTPException(status_code=503, detail="Distributed tracer not initialized")
    
    try:
        trace_details = await distributed_tracer.get_trace_details(trace_id)
        if trace_details:
            return {"trace": trace_details, "timestamp": time.time()}
        else:
            raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
    except Exception as e:
        logger.error("Failed to get trace details", trace_id=trace_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Trace details retrieval failed: {str(e)}")


# Logging endpoints
@app.get("/logs")
async def get_logs():
    """Get aggregated logs from all services."""
    try:
        # This would integrate with your log aggregation system
        # For now, return a placeholder response
        return {
            "message": "Log aggregation endpoint - integrate with your logging system",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error("Failed to get logs", error=str(e))
        raise HTTPException(status_code=500, detail=f"Log retrieval failed: {str(e)}")


@app.get("/logs/{service_name}")
async def get_service_logs(service_name: str):
    """Get logs for a specific service."""
    try:
        # This would integrate with your log aggregation system
        # For now, return a placeholder response
        return {
            "service": service_name,
            "message": "Service log endpoint - integrate with your logging system",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error("Failed to get service logs", service=service_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Service log retrieval failed: {str(e)}")


# Configuration endpoints
@app.get("/config")
async def get_monitoring_config():
    """Get current monitoring configuration."""
    settings = MonitoringSettings()
    return {
        "config": {
            "collection_interval": settings.collection_interval,
            "alert_thresholds": settings.alert_thresholds,
            "performance_tracking": settings.performance_tracking_enabled,
            "distributed_tracing": settings.distributed_tracing_enabled,
            "log_level": settings.log_level
        },
        "timestamp": time.time()
    }


@app.post("/config/reload")
async def reload_monitoring_config():
    """Reload monitoring configuration."""
    try:
        # Reload configuration for all components
        if metrics_collector:
            await metrics_collector.reload_config()
        if health_monitor:
            await health_monitor.reload_config()
        if alert_manager:
            await alert_manager.reload_config()
        if performance_tracker:
            await performance_tracker.reload_config()
        if distributed_tracer:
            await distributed_tracer.reload_config()
        
        return {"message": "Configuration reloaded successfully", "timestamp": time.time()}
    except Exception as e:
        logger.error("Failed to reload configuration", error=str(e))
        raise HTTPException(status_code=500, detail=f"Configuration reload failed: {str(e)}")


# Utility functions
async def _get_system_metrics() -> Dict[str, Any]:
    """Get system-level metrics."""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        return {
            "cpu": {
                "percent": cpu_percent,
                "count": psutil.cpu_count(),
                "count_logical": psutil.cpu_count(logical=True)
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used,
                "free": memory.free
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100
            },
            "network": {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
        }
    except Exception as e:
        logger.error("Failed to collect system metrics", error=str(e))
        return {}


# Root endpoint
@app.get("/")
async def root():
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
            "config": "/config"
        },
        "features": {
            "metrics_collection": metrics_collector is not None,
            "health_monitoring": health_monitor is not None,
            "alerting": alert_manager is not None,
            "performance_tracking": performance_tracker is not None,
            "distributed_tracing": distributed_tracer is not None
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8091)