"""Logging utilities for the export service."""

from __future__ import annotations

from functools import lru_cache

import structlog


@lru_cache(maxsize=1)
def configure_logging() -> structlog.stdlib.BoundLogger:
    """Configure structlog and return the service logger.

    The configuration mirrors the previous behaviour from the monolithic module
    while ensuring we only configure structlog once even if this function is
    imported multiple times (e.g. during test runs).
    """
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
    return structlog.get_logger()


__all__ = ["configure_logging"]
