"""Content Analyzer Service - Generates embeddings and summaries."""

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
    title="Content Analyzer Service",
    description="Generates embeddings and summaries for scraped content",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "analyzer",
        "timestamp": time.time()
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Content Analyzer Service",
        "version": "1.0.0",
        "status": "running"
    }