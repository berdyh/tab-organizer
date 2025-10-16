"""Collection of APIRouters that make up the API Gateway."""

from . import auth, health, models, proxy, rate_limit, services, status

ROUTERS = [
    status.router,
    health.router,
    services.router,
    models.router,
    auth.router,
    rate_limit.router,
    proxy.router,
]

__all__ = ["ROUTERS"]

