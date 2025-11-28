"""Application factory for the scraper service."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI

from twisted.internet import asyncioreactor

from .api.routes import router
from .dependencies import parallel_engine, scraping_engine
from .logging import get_logger, setup_logging
from .state import state

try:  # Ensure Scrapy uses asyncio reactor once
    asyncioreactor.install()
except Exception:
    pass

logger = get_logger()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    setup_logging()

    app = FastAPI(
        title="Web Scraper Service",
        description="Extracts and cleans content from web URLs with rate limiting and duplicate detection",
        version="1.0.0",
    )
    app.include_router(router)

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        """Cleanup resources on shutdown."""
        logger.info("Shutting down scraper service")

        for websocket in list(state.websocket_connections.values()):
            try:
                await websocket.close()
            except Exception:
                pass
        state.websocket_connections.clear()

        await parallel_engine.shutdown()
        await scraping_engine.close()

        logger.info("Scraper service shutdown complete")

    return app


app = create_app()

__all__ = ["app", "create_app"]
