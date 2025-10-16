"""Configuration helpers for the chatbot service."""

from __future__ import annotations

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    qdrant_host: str = Field(default="qdrant", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    ollama_base_url: AnyHttpUrl = Field(
        default="http://ollama:11434", alias="OLLAMA_BASE_URL"
    )
    default_llm_model: str = Field(default="phi4:3.8b", alias="DEFAULT_LLM_MODEL")
    default_embedding_model: str = Field(
        default="mxbai-embed-large", alias="DEFAULT_EMBEDDING_MODEL"
    )
    ollama_timeout: float = Field(default=60.0, alias="OLLAMA_TIMEOUT_SECONDS")
    qdrant_collection_prefix: str = Field(
        default="session_", alias="QDRANT_COLLECTION_PREFIX"
    )
    max_scroll_limit: int = Field(default=1000, alias="QDRANT_SCROLL_LIMIT")
    default_search_limit: int = Field(default=5, alias="QDRANT_DEFAULT_SEARCH_LIMIT")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
