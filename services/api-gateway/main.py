"""Compatibility layer exposing the API Gateway application."""

from fastapi import FastAPI

from api_gateway import app, create_app
from api_gateway.core import (
    AuthMiddleware,
    HealthChecker,
    RateLimiter,
    ServiceRegistry,
    Settings,
)
from api_gateway.state import GatewayState

__all__ = [
    "app",
    "create_app",
    "Settings",
    "ServiceRegistry",
    "HealthChecker",
    "RateLimiter",
    "AuthMiddleware",
    "GatewayState",
    "get_app_state",
]


def get_app_state(app_instance: FastAPI = app) -> GatewayState:
    """Return the current gateway state for the provided FastAPI app."""
    state = getattr(app_instance.state, "gateway_state", None)
    if not state:
        raise RuntimeError("Gateway state has not been initialized")
    return state


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
