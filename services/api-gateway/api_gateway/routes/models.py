"""Model management endpoints."""

import time

import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_service_registry, get_settings

logger = structlog.get_logger()
router = APIRouter()


@router.get("/models")
async def list_models(service_registry=Depends(get_service_registry)):
    """List available and installed Ollama models."""
    ollama_service = service_registry.get_service("ollama")
    if not ollama_service or not ollama_service.get("healthy", False):
        raise HTTPException(status_code=503, detail="Ollama service is not available")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{ollama_service['url']}/api/tags")
            if response.status_code == 200:
                models_data = response.json()
                return {
                    "installed_models": models_data.get("models", []),
                    "timestamp": time.time(),
                }

            raise HTTPException(status_code=502, detail="Failed to fetch models from Ollama")
    except Exception as exc:
        logger.error("Failed to list models", error=str(exc))
        raise HTTPException(status_code=502, detail=f"Model listing error: {exc}") from exc


@router.get("/models/config")
async def get_model_config(settings=Depends(get_settings)):
    """Get current model configuration."""
    return {
        "llm_model": settings.ollama_model,
        "embedding_model": settings.ollama_embedding_model,
        "ollama_url": settings.ollama_url,
        "timestamp": time.time(),
    }

