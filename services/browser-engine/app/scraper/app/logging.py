"""Logging utilities for the scraper service."""

from __future__ import annotations

import structlog

_IS_CONFIGURED = False


def setup_logging() -> None:
    """Configure structlog once for the entire service."""
    global _IS_CONFIGURED
    if _IS_CONFIGURED:
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
    _IS_CONFIGURED = True


def get_logger() -> structlog.stdlib.BoundLogger:
    """Return the configured logger, configuring it on first use."""
    setup_logging()
    return structlog.get_logger()
