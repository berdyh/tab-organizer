"""Application state helpers for the API Gateway."""

from dataclasses import dataclass, field
from typing import List, Optional

from .core import (
    AuthMiddleware,
    HealthChecker,
    RateLimiter,
    ServiceRegistry,
    Settings,
)


@dataclass
class GatewayState:
    """Container for runtime components attached to the FastAPI app."""

    settings: Settings
    service_registry: ServiceRegistry
    health_checker: HealthChecker
    rate_limiter: RateLimiter
    auth_middleware: AuthMiddleware
    background_tasks: List = field(default_factory=list)


def initialize_state(settings: Optional[Settings] = None) -> GatewayState:
    """Construct the gateway state using provided or default settings."""
    resolved_settings = settings or Settings()

    service_registry = ServiceRegistry(resolved_settings)
    health_checker = HealthChecker(service_registry)
    rate_limiter = RateLimiter()
    auth_middleware = AuthMiddleware()

    return GatewayState(
        settings=resolved_settings,
        service_registry=service_registry,
        health_checker=health_checker,
        rate_limiter=rate_limiter,
        auth_middleware=auth_middleware,
    )

