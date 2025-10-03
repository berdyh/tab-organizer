"""Export Service - Formats and exports results to various platforms."""

import time
from fastapi import FastAPI
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

app = FastAPI(
    title="Export Service",
    description="Formats and exports results to Notion, Obsidian, Word, and Markdown",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "export",
        "timestamp": time.time()
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Export Service",
        "version": "1.0.0",
        "status": "running"
    }