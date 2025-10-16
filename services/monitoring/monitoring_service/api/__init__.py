"""
API subpackages for the monitoring service.

Routers defined here are included in the main FastAPI application to keep
monitoring capabilities modular and easy to extend.
"""

from fastapi import APIRouter

# Expose a factory that child modules can use to register routers
api_router = APIRouter()

__all__ = ["api_router"]
