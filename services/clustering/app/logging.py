"""
Structlog configuration helpers for the clustering service.

Importing this module configures structlog once and provides a shared logger
instance that the rest of the package can use.
"""

from __future__ import annotations

import structlog


def configure_logging() -> structlog.typing.WrappedLogger:
    """Configure structlog for the clustering service."""
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


logger = configure_logging()

__all__ = ["logger", "configure_logging"]
