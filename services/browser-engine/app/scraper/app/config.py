"""Configuration handling for the scraper service."""

from __future__ import annotations

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    auth_service_url: AnyHttpUrl = Field(
        "http://auth-service:8082", alias="AUTH_SERVICE_URL"
    )
    api_gateway_url: AnyHttpUrl = Field(
        "http://api-gateway:8080", alias="API_GATEWAY_URL"
    )
    default_user_agent: str = Field(
        "WebScrapingTool/1.0 (+https://example.com/bot)", alias="SCRAPER_USER_AGENT"
    )
    max_parallel_workers: int = Field(5, ge=1, le=20)
    respect_robots: bool = Field(True)
    enable_pdf_extraction: bool = Field(True)
    content_quality_threshold: float = Field(0.5, ge=0.0, le=1.0)

    class Config:
        env_prefix = "SCRAPER_"
        case_sensitive = False
