"""Structlog configuration helpers for the Session service."""

from __future__ import annotations

import structlog


def configure_logging() -> None:
    """Configure structlog with JSON rendering."""
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


__all__ = ["configure_logging"]
