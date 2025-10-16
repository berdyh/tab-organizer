"""Health and monitoring endpoints."""

import time
from fastapi import APIRouter, Depends, HTTPException, Response

from ..dependencies import get_health_checker
from ..metrics import CONTENT_TYPE_LATEST, latest_metrics
from ..schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(health_checker=Depends(get_health_checker)):
    """
    Comprehensive health check for the API Gateway and all services.
    Returns detailed status information for monitoring and debugging.
    """
    try:
        health_status = await health_checker.get_comprehensive_health()
        return HealthResponse(
            status=health_status["status"],
            timestamp=time.time(),
            services=health_status["services"],
            uptime=health_status["uptime"],
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        raise HTTPException(
            status_code=503, detail=f"Health check failed: {exc}"
        ) from exc


@router.get("/health/simple")
async def simple_health_check():
    """Simple health check that returns OK if the gateway is running."""
    return {"status": "ok", "timestamp": time.time()}


@router.get("/health/services")
async def services_health(health_checker=Depends(get_health_checker)):
    """Get health status of all registered services."""
    try:
        services_status = await health_checker.check_all_services()
        return {"services": services_status, "timestamp": time.time()}
    except Exception as exc:  # pragma: no cover - defensive logging
        raise HTTPException(
            status_code=503, detail=f"Services health check failed: {exc}"
        ) from exc


@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=latest_metrics(), media_type=CONTENT_TYPE_LATEST)

