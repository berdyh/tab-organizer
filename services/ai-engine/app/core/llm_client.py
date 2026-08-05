"""Multi-provider LLM client with unified interface."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import httpx

# Import configuration loader
from config.config_loader import get_ai_config


class ProviderSelectionError(RuntimeError):
    """Raised when no provider was chosen, or the chosen one cannot serve.

    Fail-closed: this service never substitutes a provider for the user. The
    error carries a structured ``{code, cause, fix}`` so the reason survives
    into ``GET /health`` and the logs instead of being flattened into a
    string, matching ``CDPConnectionError``
    (``services/browser-engine/app/tabs/cdp.py``) and ``CredentialStoreError``
    (``services/browser-engine/app/auth/queue.py``).
    """

    def __init__(self, code: str, cause: str, fix: str):
        self.code = code
        self.cause = cause
        self.fix = fix
        super().__init__(f"{code}: {cause} Fix: {fix}")

    def to_dict(self) -> dict:
        return {"code": self.code, "cause": self.cause, "fix": self.fix}


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
        # openrouter serves NO embedding models (all 338 catalog entries
        # scanned, verified 2026-08-04). This mirror of the catalog is what
        # `/providers` displays, so claiming True here advertised a capability
        # `ai_models.yaml` denies and `is_provider_supported()` refuses.
        "openrouter": {"llm": True, "embeddings": False, "local": False},
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
        """Resolve the selected providers, or record why none is selected.

        R3 invariant (`routing.explicit_opt_in_required` in
        `config/ai_models.yaml`): the *only* ways a provider becomes active are
        an explicit env var (`AI_PROVIDER` / `EMBEDDING_PROVIDER`), a config
        object passed in by a caller, or an explicit `switch_provider()` call.
        Each of those is a record of a deliberate user decision. There is no
        fourth way, and there must never be one: any future "try the next
        provider" router has to re-open this constructor and confront the fact
        that `ollama`, `openrouter`, `openai`, `gemini`, `anthropic` and
        `deepseek` all require a deliberate choice, even when one of them is
        the only thing that would work.

        A selection failure is *stored*, not raised. Raising here would kill
        the process at import (`app/main.py` builds this client at module
        scope) and the service would go dark; the contract is that it starts,
        reports `degraded`, names the fix, and refuses to answer requests.
        """
        self._ai_config = get_ai_config()
        self.llm_config: Optional[LLMConfig] = llm_config
        self.llm_config_error: Optional[ProviderSelectionError] = None
        self.embedding_config: Optional[EmbeddingConfig] = embedding_config
        self.embedding_config_error: Optional[ProviderSelectionError] = None

        if self.llm_config is None:
            try:
                self.llm_config = self._default_llm_config()
            except ProviderSelectionError as exc:
                self.llm_config_error = exc
        if self.embedding_config is None:
            try:
                self.embedding_config = self._default_embedding_config()
            except ProviderSelectionError as exc:
                self.embedding_config_error = exc

        self._llm_provider: Optional[BaseLLMProvider] = None
        self._embedding_provider: Optional[BaseEmbeddingProvider] = None

    def _routing_config(self) -> dict:
        return self._ai_config.config.get("routing") or {}

    def _providers_supporting(self, capability: str) -> list[str]:
        """Catalog-derived list of providers that can serve ``capability``.

        Never hardcode this. `scripts/init.py` self-corrected when the catalog
        was fixed (openrouter's phantom embedding models) precisely because it
        asked the catalog instead of carrying its own copy of the answer.
        """
        return [
            provider
            for provider in self._ai_config.get_all_providers()
            if self._ai_config.is_provider_supported(provider, capability)
        ]

    def _provider_choices_by_cost(self, capability: str) -> dict[str, list[str]]:
        """Group the capable providers into preferred / metered / local."""
        preference_order = self._routing_config().get("llm_preference_order") or []
        capable = self._providers_supporting(capability)
        preferred = [p for p in preference_order if p in capable]
        metered: list[str] = []
        local: list[str] = []
        for provider in capable:
            if provider in preferred:
                continue
            cost_model = self._ai_config.get_provider_config(provider).get("cost_model")
            if cost_model == "free_local":
                local.append(provider)
            else:
                metered.append(provider)
        return {"preferred": preferred, "metered": metered, "local": local}

    def _not_selected_error(
        self, env_var: str, capability: str
    ) -> ProviderSelectionError:
        choices = self._provider_choices_by_cost(capability)
        lines = [
            f"Run ./scripts/cli.py configure-provider, or set {env_var} explicitly."
        ]
        if choices["preferred"]:
            lines.append(
                f"Preferred: {', '.join(choices['preferred'])} "
                "(uses your subscription)."
            )
        if choices["metered"]:
            lines.append(
                f"Metered: {', '.join(choices['metered'])} "
                "(requires an API key and consent)."
            )
        if choices["local"]:
            lines.append(
                f"Local: {', '.join(choices['local'])} "
                "(free, requires models pulled first)."
            )
        return ProviderSelectionError(
            code="provider_not_selected",
            cause=(
                f"{env_var} is not set. This service does not pick a provider "
                "for you."
            ),
            fix=" ".join(lines),
        )

    def _known_provider_config(
        self, provider: str, env_var: str, capability: str
    ) -> dict:
        """Look the provider up, turning a typo into a structured failure.

        Without this a misspelled `AI_PROVIDER` raises a bare `ValueError` out
        of the constructor, which kills the process at import instead of
        reporting `degraded`.
        """
        try:
            return self._ai_config.get_provider_config(provider)
        except ValueError:
            known = ", ".join(self._providers_supporting(capability))
            raise ProviderSelectionError(
                code="provider_unknown",
                cause=f"{env_var}={provider!r} is not a provider in the catalog.",
                fix=f"Set {env_var} to one of: {known}.",
            ) from None

    def _default_llm_config(self) -> LLMConfig:
        """Resolve the LLM config from the environment, or fail closed.

        There is deliberately no default provider here. An unset `AI_PROVIDER`
        used to mean "openrouter", which silently spent metered credit nobody
        had agreed to spend.
        """
        provider = (os.getenv("AI_PROVIDER") or "").strip()
        if not provider:
            raise self._not_selected_error("AI_PROVIDER", "llm")
        ai_config = get_ai_config()

        provider_config = self._known_provider_config(provider, "AI_PROVIDER", "llm")

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
        """Resolve the embedding config from the environment, or fail closed.

        This method used to silently rewrite `provider` to the catalog default
        when the chosen one could not embed. That fallback is deleted on
        purpose (`routing.silent_fallback: false`): it substituted a provider
        the user never chose, logged nothing, and surfaced nothing, which is
        the same defect class as WI0-B1 — a broken path reporting success.
        """
        provider = (os.getenv("EMBEDDING_PROVIDER") or "").strip()
        if not provider:
            raise self._not_selected_error("EMBEDDING_PROVIDER", "embeddings")
        ai_config = get_ai_config()

        provider_config = self._known_provider_config(
            provider, "EMBEDDING_PROVIDER", "embeddings"
        )

        if not ai_config.is_provider_supported(provider, "embeddings"):
            capable = self._providers_supporting("embeddings")
            raise ProviderSelectionError(
                code="embedding_provider_cannot_embed",
                cause=f"EMBEDDING_PROVIDER={provider!r} serves no embedding models.",
                fix=(
                    f"Choose one of: {', '.join(capable) or '(none in the catalog)'}. "
                    "Note openrouter serves NO embedding models -- verified "
                    "2026-08-04."
                ),
            )

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
        if self.llm_config is None:
            raise self.llm_config_error or self._not_selected_error(
                "AI_PROVIDER", "llm"
            )
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
        if self.embedding_config is None:
            raise self.embedding_config_error or self._not_selected_error(
                "EMBEDDING_PROVIDER", "embeddings"
            )
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
            # No openrouter entry: it serves no embedding models, so the
            # capability check above already rejects it. An adapter here would
            # only be reachable if that check were removed, and would then
            # issue requests against endpoints that do not exist.
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
        """Switch providers without restart.

        This is the second lawful way a provider becomes active (see
        ``__init__``): an authenticated caller naming one is a deliberate
        choice, exactly like setting the env var. It still cannot *infer* one —
        asking to change only the model while no provider is selected fails
        closed rather than guessing which provider the model belongs to.
        """
        if llm_provider or llm_model:
            target_provider = llm_provider or (
                self.llm_config.provider if self.llm_config else None
            )
            if not target_provider:
                raise self.llm_config_error or self._not_selected_error(
                    "AI_PROVIDER", "llm"
                )
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
            elif self.llm_config is not None:
                target_model = self.llm_config.model

            if self.llm_config is None:
                self.llm_config = LLMConfig(
                    provider=target_provider, model=target_model or ""
                )
            self.llm_config.provider = target_provider
            self.llm_config.model = target_model
            self.llm_config.api_key = self._api_key_for(target_provider)
            self.llm_config.base_url = self._base_url_for(
                target_provider, "LLM_BASE_URL"
            )
            self.llm_config_error = None
            self._llm_provider = None  # Force recreation

        if embedding_provider or embedding_model:
            target_provider = embedding_provider or (
                self.embedding_config.provider if self.embedding_config else None
            )
            if not target_provider:
                raise self.embedding_config_error or self._not_selected_error(
                    "EMBEDDING_PROVIDER", "embeddings"
                )
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
            current = self.embedding_config
            current_dimensions = (
                current.dimensions if current else EmbeddingConfig.dimensions
            )
            if target_model:
                self._validate_model(target_provider, target_model, "embedding")
                model_config = self._ai_config.get_model_config(target_model)
                dimensions = int(model_config.get("dimensions", current_dimensions))
            elif current is not None:
                target_model = current.model
                dimensions = current_dimensions
            else:
                raise ProviderSelectionError(
                    code="embedding_model_not_selected",
                    cause=(
                        f"Provider {target_provider!r} declares no default embedding "
                        "model and none was given."
                    ),
                    fix=(
                        "Pass embedding_model explicitly, or set EMBEDDING_MODEL to a "
                        f"model belonging to {target_provider!r}."
                    ),
                )

            if current is None:
                current = EmbeddingConfig(provider=target_provider, model=target_model)
                self.embedding_config = current
            current.provider = target_provider
            current.model = target_model
            current.dimensions = dimensions
            current.api_key = self._api_key_for(target_provider)
            current.base_url = self._base_url_for(target_provider, "EMBEDDING_BASE_URL")
            self.embedding_config_error = None
            self._embedding_provider = None  # Force recreation

    def refresh_runtime_credentials(self) -> None:
        """Reload credentials from the current process environment.

        A role with no selected provider stays unselected — a new API key is
        not a provider choice.
        """
        if self.llm_config is not None:
            self.llm_config.api_key = self._api_key_for(self.llm_config.provider)
        if self.embedding_config is not None:
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

    def _selection_error(self, capability: str) -> Optional[ProviderSelectionError]:
        if capability == "llm":
            if self.llm_config is not None:
                return None
            return self.llm_config_error or self._not_selected_error(
                "AI_PROVIDER", "llm"
            )
        if self.embedding_config is not None:
            return None
        return self.embedding_config_error or self._not_selected_error(
            "EMBEDDING_PROVIDER", "embeddings"
        )

    def _unselected_runtime_state(self, error: ProviderSelectionError) -> dict:
        """Runtime state for a role nobody chose: unavailable, and says why."""
        return {
            "capabilities": {},
            "available": False,
            "requires_api_key": False,
            "api_key_env": None,
            "api_key_configured": None,
            "provider": None,
            "model": None,
            "reason": str(error),
            "error": error.to_dict(),
        }

    def _current_runtime_state(self, capability: str) -> dict:
        error = self._selection_error(capability)
        if error is not None:
            return self._unselected_runtime_state(error)

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
        error = self._selection_error(capability)
        if error is not None:
            raise error
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

    def active_provider_summary(self, capability: str) -> dict:
        """Announce-shaped description of the provider serving ``capability``.

        This is the R4 attribution contract: what is answering, and what it
        costs the user. Deliberately small and stable — it is what the startup
        log line, `GET /health`, and (later) the UI badge all read. Never put a
        credential in here; `/health` is the one unauthenticated endpoint.
        """
        error = self._selection_error(capability)
        if error is not None:
            return {
                "provider": None,
                "model": None,
                "cost_model": None,
                "error": error.to_dict(),
            }

        config = self.llm_config if capability == "llm" else self.embedding_config
        provider = config.provider
        model = config.model
        try:
            cost_model = self._ai_config.get_provider_config(provider).get("cost_model")
        except ValueError:
            cost_model = None
        summary = {
            "provider": provider,
            "model": model,
            "cost_model": cost_model,
        }
        try:
            model_config = self._ai_config.get_model_config(model)
        except ValueError:
            model_config = {}
        if capability == "llm":
            summary["tier"] = model_config.get("tier")
        else:
            summary["dimensions"] = config.dimensions
        return summary

    def get_active_providers(self) -> dict:
        """Active provider per role, keyed as the spec's `/health` block."""
        return {
            "llm": self.active_provider_summary("llm"),
            "embedding": self.active_provider_summary("embeddings"),
        }

    def get_provider_info(self) -> dict:
        """Get current provider information."""
        llm_provider = self.llm_config.provider if self.llm_config else None
        embedding_config = self.embedding_config
        return {
            "llm": {
                "provider": llm_provider,
                "model": self.llm_config.model if self.llm_config else None,
                "capabilities": self.PROVIDERS.get(llm_provider, {}),
                "error": (
                    self._selection_error("llm").to_dict()
                    if self.llm_config is None
                    else None
                ),
            },
            "embeddings": {
                "provider": embedding_config.provider if embedding_config else None,
                "model": embedding_config.model if embedding_config else None,
                "dimensions": (
                    embedding_config.dimensions if embedding_config else None
                ),
                "error": (
                    self._selection_error("embeddings").to_dict()
                    if embedding_config is None
                    else None
                ),
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
