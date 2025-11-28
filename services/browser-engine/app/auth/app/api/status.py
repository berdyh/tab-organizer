"""Service status endpoints."""

import time

from fastapi import APIRouter


router = APIRouter(tags=["status"])


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "service": "auth", "timestamp": time.time()}


@router.get("/")
async def root() -> dict:
    """Root endpoint."""
    return {"service": "Authentication Service", "version": "1.0.0", "status": "running"}


__all__ = ["router"]
