"""Multi-provider LLM client with unified interface."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, AsyncIterator

import httpx

# Import configuration loader
from config.config_loader import get_ai_config


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096


@dataclass
class EmbeddingConfig:
    """Configuration for embedding provider."""
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    dimensions: int = 1536


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""
    
    @abstractmethod
    async def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    async def generate_stream(
        self, prompt: str, system: Optional[str] = None
    ) -> AsyncIterator[str]:
        """Stream generated text."""
        pass


class BaseEmbeddingProvider(ABC):
    """Base class for embedding providers."""
    
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts."""
        pass
    
    @abstractmethod
    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for single text."""
        pass


class LLMClient:
    """Unified LLM client supporting multiple providers."""
    
    PROVIDERS = {
        "ollama": {"llm": True, "embeddings": True, "local": True},
        "openai": {"llm": True, "embeddings": True, "local": False},
        "anthropic": {"llm": True, "embeddings": False, "local": False},
        "deepseek": {"llm": True, "embeddings": True, "local": False},
        "gemini": {"llm": True, "embeddings": True, "local": False},
    }
    
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
    ):
        self.llm_config = llm_config or self._default_llm_config()
        self.embedding_config = embedding_config or self._default_embedding_config()
        self._llm_provider: Optional[BaseLLMProvider] = None
        self._embedding_provider: Optional[BaseEmbeddingProvider] = None
        self._ai_config = get_ai_config()
    
    def _default_llm_config(self) -> LLMConfig:
        """Get default LLM config from environment."""
        provider = os.getenv("AI_PROVIDER", "ollama")
        ai_config = get_ai_config()
        
        # Get provider config
        provider_config = ai_config.get_provider_config(provider)
        
        # Get default model for provider
        default_model = ai_config.get_default_model(provider, "llm")
        model = os.getenv("LLM_MODEL", default_model)
        
        # Get API key from environment if cloud provider
        api_key = None
        if provider_config.get("type") == "cloud":
            api_key_env = ai_config.get_api_key_env(provider)
            if api_key_env:
                api_key = os.getenv(api_key_env)
        
        return LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL") or ai_config.get_base_url(provider),
        )
    
    def _default_embedding_config(self) -> EmbeddingConfig:
        """Get default embedding config from environment."""
        provider = os.getenv("EMBEDDING_PROVIDER", "ollama")
        ai_config = get_ai_config()
        
        # Get provider config
        provider_config = ai_config.get_provider_config(provider)
        
        # Check if provider supports embeddings
        if not ai_config.is_provider_supported(provider, "embeddings"):
            # Fallback to default provider
            provider = ai_config.get_defaults().get("provider", "ollama")
            provider_config = ai_config.get_provider_config(provider)
        
        # Get default model for provider
        default_model = ai_config.get_default_model(provider, "embedding")
        model = os.getenv("EMBEDDING_MODEL", default_model)
        
        # Get model dimensions from config
        model_config = ai_config.get_model_config(model)
        dimensions = model_config.get("dimensions", 1536)
        
        # Get API key from environment if cloud provider
        api_key = None
        if provider_config.get("type") == "cloud":
            api_key_env = ai_config.get_api_key_env(provider)
            if api_key_env:
                api_key = os.getenv(api_key_env)
        
        return EmbeddingConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=os.getenv("EMBEDDING_BASE_URL") or ai_config.get_base_url(provider),
            dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", dimensions)),
        )
    
    @property
    def llm(self) -> BaseLLMProvider:
        """Get or create LLM provider."""
        if self._llm_provider is None:
            self._llm_provider = self._create_llm_provider()
        return self._llm_provider
    
    @property
    def embeddings(self) -> BaseEmbeddingProvider:
        """Get or create embedding provider."""
        if self._embedding_provider is None:
            self._embedding_provider = self._create_embedding_provider()
        return self._embedding_provider
    
    def _create_llm_provider(self) -> BaseLLMProvider:
        """Create LLM provider based on config."""
        from ..providers import (
            OllamaLLMProvider,
            OpenAILLMProvider,
            AnthropicLLMProvider,
            DeepSeekLLMProvider,
            GeminiLLMProvider,
        )
        
        providers = {
            "ollama": OllamaLLMProvider,
            "openai": OpenAILLMProvider,
            "anthropic": AnthropicLLMProvider,
            "deepseek": DeepSeekLLMProvider,
            "gemini": GeminiLLMProvider,
        }
        
        provider_class = providers.get(self.llm_config.provider)
        if not provider_class:
            raise ValueError(f"Unknown LLM provider: {self.llm_config.provider}")
        
        return provider_class(self.llm_config)
    
    def _create_embedding_provider(self) -> BaseEmbeddingProvider:
        """Create embedding provider based on config."""
        from ..providers import (
            OllamaEmbeddingProvider,
            OpenAIEmbeddingProvider,
            DeepSeekEmbeddingProvider,
            GeminiEmbeddingProvider,
        )
        
        providers = {
            "ollama": OllamaEmbeddingProvider,
            "openai": OpenAIEmbeddingProvider,
            "deepseek": DeepSeekEmbeddingProvider,
            "gemini": GeminiEmbeddingProvider,
        }
        
        provider_class = providers.get(self.embedding_config.provider)
        if not provider_class:
            raise ValueError(
                f"Unknown embedding provider: {self.embedding_config.provider}"
            )
        
        return provider_class(self.embedding_config)
    
    async def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Generate text using configured LLM."""
        return await self.llm.generate(prompt, system)
    
    async def generate_stream(
        self, prompt: str, system: Optional[str] = None
    ) -> AsyncIterator[str]:
        """Stream generated text."""
        async for chunk in self.llm.generate_stream(prompt, system):
            yield chunk
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts."""
        return await self.embeddings.embed(texts)
    
    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for single text."""
        return await self.embeddings.embed_single(text)
    
    def switch_provider(
        self,
        llm_provider: Optional[str] = None,
        embedding_provider: Optional[str] = None,
    ) -> None:
        """Switch providers without restart."""
        if llm_provider:
            self.llm_config.provider = llm_provider
            self._llm_provider = None  # Force recreation
        
        if embedding_provider:
            self.embedding_config.provider = embedding_provider
            self._embedding_provider = None  # Force recreation
    
    def get_provider_info(self) -> dict:
        """Get current provider information."""
        return {
            "llm": {
                "provider": self.llm_config.provider,
                "model": self.llm_config.model,
                "capabilities": self.PROVIDERS.get(self.llm_config.provider, {}),
            },
            "embeddings": {
                "provider": self.embedding_config.provider,
                "model": self.embedding_config.model,
                "dimensions": self.embedding_config.dimensions,
            },
        }
    
    def get_available_models(self, provider: Optional[str] = None, model_type: Optional[str] = None) -> dict:
        """Get available models for UI selection.
        
        Args:
            provider: Optional provider filter
            model_type: Optional type filter ('llm' or 'embedding')
            
        Returns:
            Dictionary with provider names as keys and model lists as values
        """
        if provider:
            # Get models for specific provider
            models = self._ai_config.get_provider_models(provider, model_type)
            return {provider: models}
        else:
            # Get models for all providers
            result = {}
            for p in self._ai_config.get_all_providers():
                models = self._ai_config.get_provider_models(p, model_type)
                if models:
                    result[p] = models
            return result
    
    def get_model_info(self, model: str) -> dict:
        """Get detailed information about a model."""
        return self._ai_config.get_model_info(model)
    
    def get_use_case_recommendation(self, use_case: str) -> dict:
        """Get recommended provider and model for a use case."""
        return self._ai_config.get_use_case_config(use_case)
