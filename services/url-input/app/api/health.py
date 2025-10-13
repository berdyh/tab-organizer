"""System endpoints for health and root status."""

from __future__ import annotations

import time

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "url-input", "timestamp": time.time()}


@router.get("/")
async def root():
    """Root endpoint."""
    return {"service": "URL Input Service", "version": "1.0.0", "status": "running"}


__all__ = ["router"]
