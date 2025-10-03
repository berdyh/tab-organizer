"""Web Scraper Service - Extracts content from web URLs."""

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
    title="Web Scraper Service",
    description="Extracts and cleans content from web URLs",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "scraper",
        "timestamp": time.time()
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Web Scraper Service",
        "version": "1.0.0",
        "status": "running"
    }