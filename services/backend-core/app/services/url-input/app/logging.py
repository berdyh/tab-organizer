"""Logging configuration helpers for the URL Input service."""

from __future__ import annotations

import structlog


_CONFIGURED = False


def configure_logging() -> None:
    """Configure structlog for the application if it hasn't been configured yet."""
    global _CONFIGURED
    if _CONFIGURED:
        return

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
    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Return a configured structlog logger."""
    if name:
        return structlog.get_logger(name)
    return structlog.get_logger()


__all__ = ["configure_logging", "get_logger"]
