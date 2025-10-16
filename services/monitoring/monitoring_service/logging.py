"""
Centralized logging configuration for the monitoring service.
Provides structured logging with multiple output formats and handlers.
"""

import os
import sys
import logging
import logging.handlers
from typing import Any, Dict
from pathlib import Path

import structlog
from pythonjsonlogger import jsonlogger
import colorlog

from .config import MonitoringSettings


def setup_monitoring_logging():
    """Set up comprehensive logging for the monitoring service."""
    settings = MonitoringSettings()
    
    # Create logs directory if it doesn't exist
    log_path = Path(settings.log_file_path)
    log_dir = log_path.parent
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        fallback_dir = Path("logs")
        fallback_dir.mkdir(parents=True, exist_ok=True)
        log_path = fallback_dir / log_path.name
        log_dir = fallback_dir
        settings = settings.model_copy(update={"log_file_path": str(log_path)})
        structlog.get_logger("monitoring").warning(
            "Falling back to local log directory",
            configured_path=str(Path(settings.log_file_path)),
            fallback_path=str(log_path),
        )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if settings.log_format == "json" else structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    if settings.log_format == "json":
        console_formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s'
        )
    else:
        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        settings.log_file_path,
        maxBytes=settings.log_max_size_mb * 1024 * 1024,
        backupCount=settings.log_backup_count
    )
    
    if settings.log_format == "json":
        file_formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(pathname)s %(lineno)d %(message)s'
        )
    else:
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("docker").setLevel(logging.WARNING)
    
    # Create monitoring-specific logger
    monitoring_logger = structlog.get_logger("monitoring")
    monitoring_logger.info("Monitoring logging configured", 
                          log_level=settings.log_level,
                          log_format=settings.log_format,
                          log_file=settings.log_file_path)


class StructuredLogger:
    """Structured logger wrapper for consistent logging across the monitoring service."""
    
    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with structured data."""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with structured data."""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with structured data."""
        self.logger.critical(message, **kwargs)
    
    def log_metric(self, metric_name: str, value: Any, **tags):
        """Log a metric with structured tags."""
        self.logger.info("Metric recorded",
                        metric_name=metric_name,
                        metric_value=value,
                        **tags)
    
    def log_alert(self, alert_type: str, severity: str, message: str, **context):
        """Log an alert with structured context."""
        self.logger.warning("Alert generated",
                           alert_type=alert_type,
                           severity=severity,
                           alert_message=message,
                           **context)
    
    def log_performance(self, operation: str, duration: float, **context):
        """Log performance metrics."""
        self.logger.info("Performance metric",
                        operation=operation,
                        duration_seconds=duration,
                        **context)
    
    def log_health_check(self, service: str, status: str, response_time: float = None, **details):
        """Log health check results."""
        self.logger.info("Health check completed",
                        service=service,
                        health_status=status,
                        response_time_ms=response_time,
                        **details)
    
    def log_trace(self, trace_id: str, span_id: str, operation: str, **context):
        """Log distributed trace information."""
        self.logger.info("Trace span",
                        trace_id=trace_id,
                        span_id=span_id,
                        operation=operation,
                        **context)


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(name)
