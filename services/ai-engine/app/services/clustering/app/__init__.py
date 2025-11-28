"""
Clustering service package initialization.
"""

from __future__ import annotations

from fastapi import FastAPI

from .logging import logger  # noqa: F401 - configure logging on import
from .routes import router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="Clustering Service",
        description="Groups similar content using UMAP and HDBSCAN algorithms with model-aware optimization",
        version="1.0.0",
    )
    app.include_router(router)
    return app


app = create_app()

__all__ = ["app", "create_app", "router", "logger"]
