"""
Configuration settings for the monitoring service.
"""

import os
from typing import Dict, Any, List
from pydantic import BaseSettings, Field


class MonitoringSettings(BaseSettings):
    """Monitoring service configuration settings."""
    
    # Basic settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Service discovery
    services: Dict[str, str] = Field(default_factory=lambda: {
        "api-gateway": "http://api-gateway:8080",
        "url-input-service": "http://url-input-service:8081",
        "auth-service": "http://auth-service:8082",
        "scraper-service": "http://scraper-service:8083",
        "analyzer-service": "http://analyzer-service:8084",
        "clustering-service": "http://clustering-service:8085",
        "export-service": "http://export-service:8086",
        "session-service": "http://session-service:8087",
        "web-ui": "http://web-ui:8089",
        "visualization-service": "http://visualization-service:8090",
        "qdrant": "http://qdrant:6333",
        "ollama": "http://ollama:11434"
    })
    
    # Metrics collection
    collection_interval: int = Field(default=30, env="METRICS_COLLECTION_INTERVAL")
    metrics_retention_days: int = Field(default=7, env="METRICS_RETENTION_DAYS")
    
    # Health monitoring
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    health_check_timeout: int = Field(default=10, env="HEALTH_CHECK_TIMEOUT")
    health_check_retries: int = Field(default=3, env="HEALTH_CHECK_RETRIES")
    
    # Alert thresholds
    alert_thresholds: Dict[str, Any] = Field(default_factory=lambda: {
        "cpu_percent": 80.0,
        "memory_percent": 85.0,
        "disk_percent": 90.0,
        "response_time_ms": 5000,
        "error_rate_percent": 5.0,
        "service_down_minutes": 2
    })
    
    # Performance tracking
    performance_tracking_enabled: bool = Field(default=True, env="PERFORMANCE_TRACKING_ENABLED")
    benchmark_interval_minutes: int = Field(default=60, env="BENCHMARK_INTERVAL_MINUTES")
    
    # Distributed tracing
    distributed_tracing_enabled: bool = Field(default=True, env="DISTRIBUTED_TRACING_ENABLED")
    trace_retention_hours: int = Field(default=24, env="TRACE_RETENTION_HOURS")
    
    # Docker integration
    docker_socket: str = Field(default="unix://var/run/docker.sock", env="DOCKER_SOCKET")
    
    # Redis for caching and pub/sub
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # Notification settings
    notification_channels: List[str] = Field(default_factory=lambda: ["console", "webhook"])
    webhook_url: str = Field(default="", env="WEBHOOK_URL")
    slack_webhook_url: str = Field(default="", env="SLACK_WEBHOOK_URL")
    email_smtp_server: str = Field(default="", env="EMAIL_SMTP_SERVER")
    email_smtp_port: int = Field(default=587, env="EMAIL_SMTP_PORT")
    email_username: str = Field(default="", env="EMAIL_USERNAME")
    email_password: str = Field(default="", env="EMAIL_PASSWORD")
    email_recipients: List[str] = Field(default_factory=list, env="EMAIL_RECIPIENTS")
    
    # Logging configuration
    log_format: str = Field(default="json", env="LOG_FORMAT")  # json or text
    log_file_path: str = Field(default="/app/logs/monitoring.log", env="LOG_FILE_PATH")
    log_max_size_mb: int = Field(default=100, env="LOG_MAX_SIZE_MB")
    log_backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False