"""Configuration management for the API Gateway."""

import os
from typing import Dict, Any
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Basic settings
    environment: str = "development"
    log_level: str = "INFO"
    
    # Service URLs
    qdrant_url: str = "http://qdrant:6333"
    ollama_url: str = "http://ollama:11434"
    
    # Service registry configuration
    services: Dict[str, Dict[str, Any]] = {
        "url-input": {
            "url": "http://url-input-service:8081",
            "health_endpoint": "/health",
            "timeout": 10.0
        },
        "auth": {
            "url": "http://auth-service:8082",
            "health_endpoint": "/health",
            "timeout": 10.0
        },
        "scraper": {
            "url": "http://scraper-service:8083",
            "health_endpoint": "/health",
            "timeout": 10.0
        },
        "analyzer": {
            "url": "http://analyzer-service:8084",
            "health_endpoint": "/health",
            "timeout": 10.0
        },
        "clustering": {
            "url": "http://clustering-service:8085",
            "health_endpoint": "/health",
            "timeout": 10.0
        },
        "export": {
            "url": "http://export-service:8086",
            "health_endpoint": "/health",
            "timeout": 10.0
        },
        "session": {
            "url": "http://session-service:8087",
            "health_endpoint": "/health",
            "timeout": 10.0
        }
    }
    
    # Model configuration
    ollama_model: str = "llama3.2:3b"
    ollama_embedding_model: str = "nomic-embed-text"
    
    # External services
    external_services: Dict[str, Dict[str, Any]] = {
        "qdrant": {
            "url": qdrant_url,
            "health_endpoint": "/health",
            "timeout": 5.0
        },
        "ollama": {
            "url": ollama_url,
            "health_endpoint": "/api/tags",
            "timeout": 10.0
        }
    }
    
    # Health check settings
    health_check_interval: int = 30  # seconds
    health_check_timeout: int = 10   # seconds
    max_consecutive_failures: int = 3
    
    # Monitoring settings
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    # Logging settings
    log_format: str = "json"
    log_file: str = "/app/logs/api-gateway.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = False