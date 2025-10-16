"""Logging helpers."""

from __future__ import annotations

from functools import lru_cache

import structlog


@lru_cache(maxsize=1)
def configure_logging() -> None:
    """Configure structlog once for the service."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

