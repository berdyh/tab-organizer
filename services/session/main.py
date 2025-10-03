"""Session Management Service - Handles persistent storage and incremental processing."""

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
    title="Session Management Service",
    description="Handles persistent storage and incremental processing",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "session",
        "timestamp": time.time()
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Session Management Service",
        "version": "1.0.0",
        "status": "running"
    }