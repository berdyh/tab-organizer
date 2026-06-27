"""Multi-provider LLM client with unified interface."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional

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
        "claude_code": {
            "llm": True,
            "embeddings": False,
            "local": True,
            "subscription": True,
        },
        "codex_cli": {
            "llm": True,
            "embeddings": False,
            "local": True,
            "subscription": True,
        },
        "codex_acp": {
            "llm": True,
            "embeddings": False,
            "local": True,
            "subscription": True,
            "acp": True,
        },
        "deepseek": {"llm": True, "embeddings": False, "local": False},
        "gemini": {"llm": True, "embeddings": True, "local": False},
        "openrouter": {"llm": True, "embeddings": True, "local": False},
    }
    CLI_PROVIDER_COMMANDS = {
        "claude_code": ("CLAUDE_CODE_COMMAND", "claude"),
        "codex_cli": ("CODEX_CLI_COMMAND", "codex"),
        "codex_acp": ("CODEX_ACP_COMMAND", "acpx"),
    }

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
    ):
        self._ai_config = get_ai_config()
        self.llm_config = llm_config or self._default_llm_config()
        self.embedding_config = embedding_config or self._default_embedding_config()
        self._llm_provider: Optional[BaseLLMProvider] = None
        self._embedding_provider: Optional[BaseEmbeddingProvider] = None

    def _default_llm_config(self) -> LLMConfig:
        """Get default LLM config from environment."""
        provider = os.getenv("AI_PROVIDER") or "openrouter"
        ai_config = get_ai_config()

        # Get provider config
        provider_config = ai_config.get_provider_config(provider)

        # Get default model for provider
        default_model = ai_config.get_default_model(provider, "llm")
        model = os.getenv("LLM_MODEL") or default_model

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
            base_url=self._base_url_for(provider, "LLM_BASE_URL"),
        )

    def _default_embedding_config(self) -> EmbeddingConfig:
        """Get default embedding config from environment."""
        provider = os.getenv("EMBEDDING_PROVIDER") or "openrouter"
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
        model = os.getenv("EMBEDDING_MODEL") or default_model

        # Get model dimensions from config
        model_config = ai_config.get_model_config(model)
        dimensions = model_config.get("dimensions", 1536)

        # Get API key from environment if cloud provider
        api_key = None
        if provider_config.get("type") == "cloud":
            api_key_env = ai_config.get_api_key_env(provider)
            if api_key_env:
                api_key = os.getenv(api_key_env)

        dimensions_override = os.getenv("EMBEDDING_DIMENSIONS")

        return EmbeddingConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=self._base_url_for(provider, "EMBEDDING_BASE_URL"),
            dimensions=int(dimensions_override or dimensions),
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
        if not self._ai_config.is_provider_supported(self.llm_config.provider, "llm"):
            raise ValueError(
                f"Provider {self.llm_config.provider} does not support LLMs"
            )

        from ..providers import (
            AnthropicLLMProvider,
            ClaudeCodeLLMProvider,
            CodexAcpLLMProvider,
            CodexCliLLMProvider,
            DeepSeekLLMProvider,
            GeminiLLMProvider,
            OllamaLLMProvider,
            OpenAILLMProvider,
        )

        providers = {
            "ollama": OllamaLLMProvider,
            "openai": OpenAILLMProvider,
            "anthropic": AnthropicLLMProvider,
            "claude_code": ClaudeCodeLLMProvider,
            "codex_cli": CodexCliLLMProvider,
            "codex_acp": CodexAcpLLMProvider,
            "deepseek": DeepSeekLLMProvider,
            "gemini": GeminiLLMProvider,
            "openrouter": OpenAILLMProvider,
        }

        provider_class = providers.get(self.llm_config.provider)
        if not provider_class:
            raise ValueError(f"Unknown LLM provider: {self.llm_config.provider}")

        return provider_class(self.llm_config)

    def _create_embedding_provider(self) -> BaseEmbeddingProvider:
        """Create embedding provider based on config."""
        if not self._ai_config.is_provider_supported(
            self.embedding_config.provider, "embeddings"
        ):
            raise ValueError(
                f"Provider {self.embedding_config.provider} does not support embeddings"
            )

        from ..providers import (
            DeepSeekEmbeddingProvider,
            GeminiEmbeddingProvider,
            OllamaEmbeddingProvider,
            OpenAIEmbeddingProvider,
        )

        providers = {
            "ollama": OllamaEmbeddingProvider,
            "openai": OpenAIEmbeddingProvider,
            "deepseek": DeepSeekEmbeddingProvider,
            "gemini": GeminiEmbeddingProvider,
            "openrouter": OpenAIEmbeddingProvider,
        }

        provider_class = providers.get(self.embedding_config.provider)
        if not provider_class:
            raise ValueError(
                f"Unknown embedding provider: {self.embedding_config.provider}"
            )

        return provider_class(self.embedding_config)

    async def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Generate text using configured LLM."""
        self._ensure_current_provider_ready("llm")
        return await self.llm.generate(prompt, system)

    async def generate_stream(
        self, prompt: str, system: Optional[str] = None
    ) -> AsyncIterator[str]:
        """Stream generated text."""
        self._ensure_current_provider_ready("llm")
        async for chunk in self.llm.generate_stream(prompt, system):
            yield chunk

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts."""
        self._ensure_current_provider_ready("embeddings")
        return await self.embeddings.embed(texts)

    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for single text."""
        self._ensure_current_provider_ready("embeddings")
        return await self.embeddings.embed_single(text)

    def switch_provider(
        self,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        """Switch providers without restart."""
        if llm_provider or llm_model:
            target_provider = llm_provider or self.llm_config.provider
            if not self._ai_config.is_provider_supported(target_provider, "llm"):
                raise ValueError(f"Provider {target_provider} does not support LLMs")
            state = self.get_provider_runtime_state(target_provider, "llm")
            if not state["available"]:
                raise ValueError(
                    state.get("reason")
                    or f"Provider {target_provider} is not available in this runtime"
                )

            target_model = llm_model
            if not target_model and llm_provider:
                target_model = self._ai_config.get_default_model(target_provider, "llm")
            if target_model:
                self._validate_model(target_provider, target_model, "llm")
            else:
                target_model = self.llm_config.model

            self.llm_config.provider = target_provider
            self.llm_config.model = target_model
            self.llm_config.api_key = self._api_key_for(target_provider)
            self.llm_config.base_url = self._base_url_for(
                target_provider, "LLM_BASE_URL"
            )
            self._llm_provider = None  # Force recreation

        if embedding_provider or embedding_model:
            target_provider = embedding_provider or self.embedding_config.provider
            if not self._ai_config.is_provider_supported(target_provider, "embeddings"):
                raise ValueError(
                    f"Provider {target_provider} does not support embeddings"
                )
            state = self.get_provider_runtime_state(target_provider, "embeddings")
            if not state["available"]:
                raise ValueError(
                    state.get("reason")
                    or f"Provider {target_provider} is not available in this runtime"
                )

            target_model = embedding_model
            if not target_model and embedding_provider:
                target_model = self._ai_config.get_default_model(
                    target_provider, "embedding"
                )
            if target_model:
                self._validate_model(target_provider, target_model, "embedding")
                model_config = self._ai_config.get_model_config(target_model)
                dimensions = int(
                    model_config.get("dimensions", self.embedding_config.dimensions)
                )
            else:
                target_model = self.embedding_config.model
                dimensions = self.embedding_config.dimensions

            self.embedding_config.provider = target_provider
            self.embedding_config.model = target_model
            self.embedding_config.dimensions = dimensions
            self.embedding_config.api_key = self._api_key_for(target_provider)
            self.embedding_config.base_url = self._base_url_for(
                target_provider, "EMBEDDING_BASE_URL"
            )
            self._embedding_provider = None  # Force recreation

    def refresh_runtime_credentials(self) -> None:
        """Reload credentials from the current process environment."""
        self.llm_config.api_key = self._api_key_for(self.llm_config.provider)
        self.embedding_config.api_key = self._api_key_for(
            self.embedding_config.provider
        )
        self._llm_provider = None
        self._embedding_provider = None

    def is_provider_runtime_available(
        self, provider: str, capability: str = "llm"
    ) -> bool:
        """Check runtime availability in addition to static config support."""
        return self.get_provider_runtime_state(provider, capability)["available"]

    def get_provider_runtime_state(
        self, provider: str, capability: str = "llm"
    ) -> dict:
        """Return runtime availability details for a configured provider."""
        state = {
            "capabilities": self.PROVIDERS.get(provider, {}),
            "available": False,
            "requires_api_key": False,
            "api_key_env": None,
            "api_key_configured": None,
            "reason": None,
        }

        try:
            provider_config = self._ai_config.get_provider_config(provider)
            supported = self._ai_config.is_provider_supported(provider, capability)
        except ValueError as exc:
            state["reason"] = str(exc)
            return state

        if not supported:
            state["reason"] = f"Provider {provider} does not support {capability}"
            return state

        provider_type = provider_config.get("type")
        if provider_type == "cloud":
            api_key_env = self._ai_config.get_api_key_env(provider)
            configured = bool(api_key_env and os.getenv(api_key_env, "").strip())
            state.update(
                {
                    "requires_api_key": True,
                    "api_key_env": api_key_env,
                    "api_key_configured": configured,
                    "available": configured,
                    "reason": (
                        None if configured else f"{api_key_env} is not configured"
                    ),
                }
            )
            return state

        if capability == "llm" and provider in self.CLI_PROVIDER_COMMANDS:
            command_env, default_command = self.CLI_PROVIDER_COMMANDS[provider]
            available = self._cli_provider_available(provider)
            command = os.getenv(command_env, default_command)
            state.update(
                {
                    "available": available,
                    "command_env": command_env,
                    "command": command,
                    "reason": (
                        None
                        if available
                        else f"{command} is not available or failed its preflight check"
                    ),
                }
            )
            return state

        state["available"] = True
        return state

    def _cli_provider_available(self, provider: str) -> bool:
        from ..providers import (
            ClaudeCodeLLMProvider,
            CodexAcpLLMProvider,
            CodexCliLLMProvider,
        )

        providers = {
            "claude_code": ClaudeCodeLLMProvider,
            "codex_cli": CodexCliLLMProvider,
            "codex_acp": CodexAcpLLMProvider,
        }
        provider_class = providers.get(provider)
        return bool(provider_class and provider_class.is_available())

    def _validate_runtime_provider(self, provider: str, capability: str) -> None:
        if not self.is_provider_runtime_available(provider, capability):
            raise ValueError(f"Provider {provider} is not available in this runtime")

    def _validate_model(self, provider: str, model: str, model_type: str) -> None:
        model_config = self._ai_config.get_model_config(model)
        if model_config.get("provider") != provider:
            raise ValueError(f"Model {model} does not belong to provider {provider}")
        if model_config.get("type") != model_type:
            raise ValueError(f"Model {model} is not a {model_type} model")

    def _current_runtime_state(self, capability: str) -> dict:
        if capability == "llm":
            provider = self.llm_config.provider
            model = self.llm_config.model
            model_type = "llm"
        else:
            provider = self.embedding_config.provider
            model = self.embedding_config.model
            model_type = "embedding"

        state = self.get_provider_runtime_state(provider, capability)
        state.update({"provider": provider, "model": model})

        if state["available"]:
            try:
                self._validate_model(provider, model, model_type)
            except ValueError as exc:
                state["available"] = False
                state["reason"] = str(exc)
            else:
                if provider == "ollama":
                    ollama_state = self._ollama_current_model_state(
                        capability,
                        model,
                    )
                    state.update(ollama_state)
        return state

    def _ensure_current_provider_ready(self, capability: str) -> None:
        state = self._current_runtime_state(capability)
        if not state["available"]:
            raise ValueError(state.get("reason") or "AI provider is not configured")

    def _api_key_for(self, provider: str) -> Optional[str]:
        """Resolve an API key for cloud providers; local providers return None."""
        provider_config = self._ai_config.get_provider_config(provider)
        if provider_config.get("type") != "cloud":
            return None

        api_key_env = self._ai_config.get_api_key_env(provider)
        return os.getenv(api_key_env) if api_key_env else None

    def _base_url_for(self, provider: str, override_env: str) -> str:
        """Resolve provider base URL, keeping Docker Ollama env authoritative."""
        if provider == "ollama":
            return (
                os.getenv(override_env)
                or os.getenv("OLLAMA_HOST")
                or self._ai_config.get_base_url(provider)
            )
        return os.getenv(override_env) or self._ai_config.get_base_url(provider)

    def _ollama_current_model_state(self, capability: str, model: str) -> dict:
        """Check the selected Ollama server and model for health reporting."""
        if capability == "llm":
            base_url = self.llm_config.base_url or self._base_url_for(
                "ollama", "LLM_BASE_URL"
            )
        else:
            base_url = self.embedding_config.base_url or self._base_url_for(
                "ollama", "EMBEDDING_BASE_URL"
            )
        base_url = base_url.rstrip("/")

        try:
            response = httpx.get(
                f"{base_url}/api/tags",
                timeout=self._ollama_health_timeout(),
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            return {
                "available": False,
                "server_url": base_url,
                "reason": f"Ollama is not reachable at {base_url}: {exc}",
            }

        installed_models = self._ollama_installed_models(payload)
        if model not in installed_models:
            return {
                "available": False,
                "server_url": base_url,
                "installed_models": sorted(installed_models),
                "reason": f"Ollama model {model} is not installed at {base_url}",
            }

        return {
            "available": True,
            "server_url": base_url,
            "installed_models": sorted(installed_models),
            "reason": None,
        }

    def _ollama_installed_models(self, payload: dict) -> set[str]:
        models = payload.get("models", [])
        installed: set[str] = set()
        if not isinstance(models, list):
            return installed

        for item in models:
            if not isinstance(item, dict):
                continue
            for key in ("name", "model"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    model_name = value.strip()
                    installed.add(model_name)
                    if model_name.endswith(":latest"):
                        installed.add(model_name.removesuffix(":latest"))

        return installed

    def _ollama_health_timeout(self) -> float:
        raw = os.getenv("OLLAMA_HEALTH_TIMEOUT", "1.0").strip()
        try:
            timeout = float(raw)
        except ValueError:
            return 1.0
        return max(timeout, 0.1)

    def _provider_catalog(self, capability: str) -> dict:
        result = {}
        for provider in self._ai_config.get_all_providers():
            if not self._ai_config.is_provider_supported(provider, capability):
                continue
            result[provider] = self.get_provider_runtime_state(provider, capability)
        return result

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
            "available": {
                "llm": self._provider_catalog("llm"),
                "embeddings": self._provider_catalog("embeddings"),
            },
            "models": {
                "llm": self.get_available_models(model_type="llm"),
                "embeddings": self.get_available_models(model_type="embedding"),
            },
        }

    def get_runtime_health(self) -> dict:
        """Return health details for the currently selected providers."""
        llm_state = self._current_runtime_state("llm")
        embedding_state = self._current_runtime_state("embeddings")
        return {
            "ready": llm_state["available"] and embedding_state["available"],
            "llm": llm_state,
            "embeddings": embedding_state,
        }

    def get_available_models(
        self, provider: Optional[str] = None, model_type: Optional[str] = None
    ) -> dict:
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
