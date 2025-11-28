"""API router assembly for the URL Input service."""

from __future__ import annotations

from fastapi import APIRouter

from . import health, ingest, inputs, processing

router = APIRouter()
router.include_router(health.router)
router.include_router(inputs.router)
router.include_router(ingest.router)
router.include_router(processing.router)

__all__ = ["router"]
