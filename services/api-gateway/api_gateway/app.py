"""Application factory for the API Gateway."""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core import Settings, setup_logging
from .dependencies import get_gateway_state  # noqa: F401  # Re-export for compatibility
from .middleware import request_metrics_middleware
from .routes import ROUTERS
from .state import GatewayState, initialize_state

logger = structlog.get_logger()


async def _background_cleanup(state: GatewayState) -> None:
    """Background task for periodic cleanup."""
    auth_middleware = state.auth_middleware
    while True:
        try:
            if auth_middleware:
                auth_middleware.cleanup_expired_tokens()
            await asyncio.sleep(300)  # Cleanup every 5 minutes
        except asyncio.CancelledError:
            break
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Background cleanup error", error=str(exc))
            await asyncio.sleep(60)


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Create a configured FastAPI application instance."""

    setup_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Starting API Gateway...")
        state = initialize_state(settings)
        app.state.gateway_state = state

        # Start background tasks
        if state.settings.enable_background_tasks:
            health_task = asyncio.create_task(state.health_checker.start_monitoring())
            cleanup_task = asyncio.create_task(_background_cleanup(state))
            state.background_tasks.extend([health_task, cleanup_task])

        logger.info("API Gateway started successfully")

        try:
            yield
        finally:
            logger.info("Shutting down API Gateway...")

            # Cancel background tasks
            for task in state.background_tasks:
                task.cancel()
            for task in state.background_tasks:
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Close resources
            close_rate_limiter = getattr(state.rate_limiter, "close", None)
            if close_rate_limiter:
                await close_rate_limiter()

            close_registry = getattr(state.service_registry, "close", None)
            if close_registry:
                await close_registry()

            logger.info("API Gateway shutdown complete")

    app = FastAPI(
        title="Web Scraping & Clustering Tool - API Gateway",
        description="Central orchestration layer for web scraping, analysis, and clustering services",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request middleware
    app.middleware("http")(request_metrics_middleware)

    # Include routers
    for router in ROUTERS:
        app.include_router(router)

    return app


# Default application instance for ASGI tools (e.g. uvicorn)
app = create_app()
