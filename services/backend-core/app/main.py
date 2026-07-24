"""Backend Core Service - Main Application."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.config_loader import get_ai_config
from services.observability import RequestIDMiddleware, configure_logging, log_event

from .api.routes import router

configure_logging("backend-core")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Refuse to start on malformed shared AI config (finding 27).

    Backend Core does not read this config itself, but it ships the same
    ai_models.yaml the AI Engine depends on; failing fast here surfaces a
    broken provider catalog before any request is served, not on first use.
    """
    errors = get_ai_config().validate_config()
    if errors:
        log_event("config.invalid", level=logging.CRITICAL, errors=errors)
        raise RuntimeError("AI model configuration is invalid: " + "; ".join(errors))
    yield


app = FastAPI(
    title="Tab Organizer - Backend Core",
    description="Backend API and session management for Tab Organizer",
    version="1.0.0",
    lifespan=lifespan,
)

# Bind X-Request-ID for every request before other middleware runs.
app.add_middleware(RequestIDMiddleware, service="backend-core")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"service": "backend-core", "version": "1.0.0", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
