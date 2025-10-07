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
        "url-input-service": {
            "url": "http://url-input-service:8081",
            "health_endpoint": "/health",
            "timeout": 10.0,
            "base_path": "api"
        },
        "auth-service": {
            "url": "http://auth-service:8082",
            "health_endpoint": "/health",
            "timeout": 10.0
        },
        "scraper-service": {
            "url": "http://scraper-service:8083",
            "health_endpoint": "/health",
            "timeout": 10.0
        },
        "analyzer-service": {
            "url": "http://analyzer-service:8084",
            "health_endpoint": "/health",
            "timeout": 10.0
        },
        "clustering-service": {
            "url": "http://clustering-service:8085",
            "health_endpoint": "/health",
            "timeout": 10.0
        },
        "export-service": {
            "url": "http://export-service:8086",
            "health_endpoint": "/health",
            "timeout": 10.0
        },
        "session-service": {
            "url": "http://session-service:8087",
            "health_endpoint": "/health",
            "timeout": 10.0
        },
        "chatbot-service": {
            "url": "http://chatbot-service:8092",
            "health_endpoint": "/health",
            "timeout": 10.0
        },
        "visualization-service": {
            "url": "http://visualization-service:8090",
            "health_endpoint": "/health",
            "timeout": 10.0
        }
    }
    
    # Model configuration
    ollama_model: str = "llama3.2:3b"
    ollama_embedding_model: str = "nomic-embed-text"
    
    @property
    def external_services(self) -> Dict[str, Dict[str, Any]]:
        """External services configuration with dynamic URLs."""
        return {
            "qdrant": {
                "url": self.qdrant_url,
                "health_endpoint": "/",
                "timeout": 5.0
            },
            "ollama": {
                "url": self.ollama_url,
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
