"""
Application factory for the monitoring service.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import core as core_routes
from .api import visualization
from .config import MonitoringSettings
from .core.components import MonitoringComponents
from .instrumentation import register_request_middleware
from .logging import get_logger, setup_monitoring_logging

logger = get_logger("app")


def create_app(settings: Optional[MonitoringSettings] = None) -> FastAPI:
    """Return a fully configured FastAPI application."""
    setup_monitoring_logging()

    components = MonitoringComponents(settings=settings or MonitoringSettings())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Starting Monitoring Service...")
        await components.startup()
        logger.info("Monitoring Service started successfully")
        try:
            yield
        finally:
            logger.info("Shutting down Monitoring Service...")
            await components.shutdown()
            logger.info("Monitoring Service shutdown complete")

    app = FastAPI(
        title="Web Scraping Tool - Monitoring Service",
        description="Comprehensive monitoring, logging, and performance optimization service",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.state.components = components

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_request_middleware(app)
    app.include_router(core_routes.router)
    app.include_router(visualization.router)

    return app


def get_app() -> FastAPI:
    """Provide an application instance for ASGI servers."""
    return create_app()
