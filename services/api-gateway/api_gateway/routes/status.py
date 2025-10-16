"""Service status and root endpoint."""

import time
from fastapi import APIRouter, Depends

from ..dependencies import get_gateway_state
from ..state import GatewayState

router = APIRouter()


@router.get("/")
async def root(state: GatewayState = Depends(get_gateway_state)):
    """Root endpoint with API information."""
    return {
        "name": "Web Scraping & Clustering Tool - API Gateway",
        "version": "1.0.0",
        "status": "running",
        "timestamp": time.time(),
        "endpoints": {
            "health": "/health",
            "services": "/services",
            "models": "/models",
            "metrics": "/metrics",
            "auth": "/auth/login",
            "rate_limit": "/rate-limit/info",
            "api_proxy": "/api/{service_name}/{path}",
        },
        "features": {
            "rate_limiting": state.rate_limiter is not None,
            "authentication": state.auth_middleware is not None,
            "service_discovery": state.service_registry is not None,
            "health_monitoring": state.health_checker is not None,
        },
    }

