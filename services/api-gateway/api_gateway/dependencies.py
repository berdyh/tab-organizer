"""FastAPI dependency helpers."""

from typing import Any, Dict

from fastapi import Depends, HTTPException, Request

from .core import AuthMiddleware, HealthChecker, RateLimiter, ServiceRegistry, Settings
from .state import GatewayState


def get_gateway_state(request: Request) -> GatewayState:
    """Return the gateway state stored on the FastAPI application."""
    state = getattr(request.app.state, "gateway_state", None)
    if not state:
        raise HTTPException(status_code=503, detail="Gateway not initialized")
    return state


def get_settings(state: GatewayState = Depends(get_gateway_state)) -> Settings:
    return state.settings


def get_service_registry(
    state: GatewayState = Depends(get_gateway_state),
) -> ServiceRegistry:
    return state.service_registry


def get_health_checker(
    state: GatewayState = Depends(get_gateway_state),
) -> HealthChecker:
    return state.health_checker


def get_rate_limiter(
    state: GatewayState = Depends(get_gateway_state),
) -> RateLimiter:
    return state.rate_limiter


def get_auth_middleware(
    state: GatewayState = Depends(get_gateway_state),
) -> AuthMiddleware:
    return state.auth_middleware


async def authenticate(
    request: Request, auth: AuthMiddleware = Depends(get_auth_middleware)
) -> Dict[str, Any]:
    """Authenticate and authorize the incoming request."""
    auth_info = await auth.authenticate_request(request)
    await auth.authorize_request(request, auth_info)
    return auth_info

