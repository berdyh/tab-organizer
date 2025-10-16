"""Logging utilities for the analyzer service."""

import structlog

_configured = False


def configure_logging() -> None:
    """Configure structlog for the analyzer service."""
    global _configured
    if _configured:
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
    _configured = True


__all__ = ["configure_logging"]
