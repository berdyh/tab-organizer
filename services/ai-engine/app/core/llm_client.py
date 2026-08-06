"""Multi-provider LLM client with unified interface."""

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import httpx

# Import configuration loader
from config.config_loader import get_ai_config

# `scheme://user:pass@host` anywhere inside a string. Matched on the whole
# message rather than on the URL alone because the credential can also arrive
# second-hand: httpx puts the request URL into its own exception text, so
# f"...: {exc}" re-imports the userinfo the caller just stripped.
_URL_USERINFO_RE = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.\-]*://)[^/\s@]+@")


def redact_url_userinfo(text: Optional[str]) -> Optional[str]:
    """Strip `user:pass@` from every URL in ``text``.

    `GET /health` is unauthenticated and every `reason` string on it is built
    from a base URL. `OLLAMA_HOST=http://admin:s3cret@host:11434` is a
    perfectly ordinary way to reach a proxied Ollama, and it turned the
    unauthenticated health endpoint into a credential disclosure. The invariant
    in `MODULE.md` is that *nothing* in the announce or runtime surface carries
    a key or a token -- URL userinfo is a credential like any other.
    """
    if not text:
        return text
    return _URL_USERINFO_RE.sub(r"\g<scheme>***@", text)


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


class ProviderUnavailableError(ProviderSelectionError):
    """The provider *was* chosen, but it cannot serve right now.

    A subclass rather than a sibling on purpose: "nobody chose a provider" and
    "the provider you chose has no `claude` binary / no API key / no pulled
    model" are the same thing to every caller that must not paper over it. Any
    best-effort ``except Exception`` that would swallow one must swallow
    neither, so they share a base and a single ``except ProviderSelectionError:
    raise`` guard covers both.

    This is the condition `MODULE.md` used to carry as a stub -- "unavailable
    local subscription CLI providers should report unhealthy rather than
    silently falling back". The plain ``ValueError`` it used to raise was
    indistinguishable from a transient provider error and got absorbed by
    exactly such a handler in the clustering pipeline.
    """

    def __init__(self, code: str = "provider_unavailable", *, cause: str, fix: str):
        super().__init__(code=code, cause=cause, fix=fix)


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
    # True when the catalog marks the model `dimensions_configurable` -- i.e.
    # the model accepts a `dimensions` request parameter and will emit exactly
    # that width (Matryoshka truncation, as on OpenAI's text-embedding-3-*).
    #
    # This is what makes `dimensions` an INSTRUCTION to the provider rather
    # than only a declaration about it. Without it, an adapter that silently
    # dropped the parameter would return the model's native width while
    # `/health` announced the configured one, and ai-engine would refuse every
    # write to a table built at the announced width -- the exact silent-drift
    # class `_resolve_embedding_dimensions` exists to prevent. Adapters must
    # send the parameter when this is True; a model without it is a fixed-width
    # model and `EMBEDDING_DIMENSIONS` may only restate its catalog width.
    dimensions_configurable: bool = False


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
        # embeddings was set False here on 2026-08-04 to mirror a catalog entry
        # that was itself wrong (inferred from openrouter's /v1/models chat
        # listing instead of from a call to /v1/embeddings -- see the
        # correction note in config/ai_models.yaml). Restored 2026-08-05 after
        # the endpoint was called directly and answered 200.
        #
        # This dict is a MIRROR, never a second source of truth: it exists so
        # `GET /providers` can answer without a catalog round-trip, and
        # `test_provider_capability_mirror_agrees_with_the_catalog` fails the
        # build on any divergence. It did exactly that when the catalog was
        # corrected and this line was not -- which is the only reason the
        # falsehood could not survive here quietly.
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

        That contract is about the *class* of misconfiguration, not about the
        two fields someone remembered. Any exception raised while resolving a
        role -- a typo'd model name, a non-numeric `EMBEDDING_DIMENSIONS`, a
        catalog entry missing a field -- is caught here and converted into a
        structured `{code, cause, fix}`. A bare `except ProviderSelectionError`
        would only cover the paths already taught to raise one, and the next
        env var nobody thought about would blackhole the service at import
        again.
        """
        self._ai_config = get_ai_config()
        self.llm_config: Optional[LLMConfig] = llm_config
        self.llm_config_error: Optional[ProviderSelectionError] = None
        self.embedding_config: Optional[EmbeddingConfig] = embedding_config
        self.embedding_config_error: Optional[ProviderSelectionError] = None

        if self.llm_config is None:
            self.llm_config, self.llm_config_error = self._resolve_role(
                self._default_llm_config, "llm"
            )
        if self.embedding_config is None:
            self.embedding_config, self.embedding_config_error = self._resolve_role(
                self._default_embedding_config, "embeddings"
            )

        self._llm_provider: Optional[BaseLLMProvider] = None
        self._embedding_provider: Optional[BaseEmbeddingProvider] = None

    def _resolve_role(self, resolve, capability: str):
        """Run a role resolver, degrading on *any* failure instead of dying."""
        try:
            return resolve(), None
        except ProviderSelectionError as exc:
            return None, exc
        except Exception as exc:  # noqa: BLE001 -- deliberate: see __init__
            env_vars = (
                "AI_PROVIDER / LLM_MODEL"
                if capability == "llm"
                else "EMBEDDING_PROVIDER / EMBEDDING_MODEL / EMBEDDING_DIMENSIONS"
            )
            return None, ProviderSelectionError(
                code="provider_config_invalid",
                cause=(
                    f"The {capability} configuration could not be resolved: "
                    f"{type(exc).__name__}: {redact_url_userinfo(str(exc))}"
                ),
                fix=(
                    f"Check {env_vars} against config/ai_models.yaml, or run "
                    "./scripts/cli.py configure-provider to rewrite them."
                ),
            )

    def _routing_config(self) -> dict:
        return self._ai_config.config.get("routing") or {}

    def _providers_supporting(self, capability: str) -> list[str]:
        """Catalog-derived list of providers that can serve ``capability``.

        Never hardcode this. Demonstrated twice, in both directions: when the
        catalog wrongly dropped openrouter's embedding support (2026-08-04) and
        again when that was corrected (2026-08-05), `scripts/init.py` tracked
        the change with no edit, purely because it asks the catalog instead of
        carrying its own copy of the answer. Every hardcoded copy of the same
        fact -- the `PROVIDERS` mirror, the embedding adapter map, the prose in
        half a dozen docs -- had to be repaired by hand both times.
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

    def _known_model_config(self, model: Optional[str], env_var: str) -> dict:
        """Look the model up, turning a typo into a structured failure.

        Same reasoning as `_known_provider_config`, applied to the field the
        provider fix missed: `get_model_config()` raises a bare `ValueError`,
        and this is reached from `__init__`.
        """
        if not model:
            raise ProviderSelectionError(
                code="model_not_selected",
                cause=(
                    "No model is selected and the provider declares no default "
                    "for this role."
                ),
                fix=f"Set {env_var} to a model listed in config/ai_models.yaml.",
            )
        try:
            return self._ai_config.get_model_config(model)
        except ValueError:
            raise ProviderSelectionError(
                code="model_unknown",
                cause=f"{env_var}={model!r} is not a model in the catalog.",
                fix=(
                    f"Set {env_var} to a model listed in config/ai_models.yaml, "
                    "or leave it blank to use the provider's default."
                ),
            ) from None

    def _resolve_embedding_dimensions(self, model: str, model_config: dict) -> int:
        """Resolve the vector width from the catalog, never by inference.

        The catalog is the only source of truth for dimensions (`R1`/`R2`: no
        silent substitution). Two ways this used to drift, both silent:

        * a catalog entry with no `dimensions` key fell back to a hardcoded
          1536, so an embedding model of any width announced 1536 and the
          LanceDB table was created at a width nothing would ever produce;
        * `EMBEDDING_DIMENSIONS` overrode the catalog with no comparison, so
          `/health` and the `provider.active` line announced a number that
          contradicted the model actually being called.

        Both now fail closed: the role degrades and `/health` names the fix.
        """
        catalog_dimensions = model_config.get("dimensions")
        if catalog_dimensions is None:
            raise ProviderSelectionError(
                code="embedding_dimensions_unknown",
                cause=(
                    f"The catalog entry for embedding model {model!r} declares no "
                    "'dimensions', and this service does not guess a vector width."
                ),
                fix=(
                    f"Add 'dimensions' to the {model!r} entry in "
                    "config/ai_models.yaml (it must match what the model emits)."
                ),
            )

        override = (os.getenv("EMBEDDING_DIMENSIONS") or "").strip()
        if not override:
            return int(catalog_dimensions)

        try:
            requested = int(override)
        except ValueError:
            raise ProviderSelectionError(
                code="embedding_dimensions_invalid",
                cause=f"EMBEDDING_DIMENSIONS={override!r} is not an integer.",
                fix=(
                    "Leave EMBEDDING_DIMENSIONS blank so it resolves from the "
                    f"catalog ({model} is {catalog_dimensions}), or set it to that "
                    "number."
                ),
            ) from None

        if requested == int(catalog_dimensions):
            return requested

        # A `dimensions_configurable` model is told what width to emit, so an
        # override is not drift -- it is the request. The invariant is
        # unchanged and still enforced: the announced width must equal what the
        # model will actually produce. Truncation only goes down from the
        # native width, so an over-ask is still refused.
        if model_config.get("dimensions_configurable"):
            if 0 < requested < int(catalog_dimensions):
                return requested
            raise ProviderSelectionError(
                code="embedding_dimensions_out_of_range",
                cause=(
                    f"EMBEDDING_DIMENSIONS={requested} is not a width {model} can "
                    f"emit: it is configurable but cannot exceed its native "
                    f"{catalog_dimensions} (and must be positive)."
                ),
                fix=(
                    f"Choose a value between 1 and {catalog_dimensions}, or leave "
                    "EMBEDDING_DIMENSIONS blank to use the native width. 768 "
                    "matches a table built by a 768-d model (e.g. "
                    "nomic-embed-text) and needs no reindex."
                ),
            )

        raise ProviderSelectionError(
            code="embedding_dimensions_mismatch",
            cause=(
                f"EMBEDDING_DIMENSIONS={requested} contradicts the catalog: "
                f"{model} emits {catalog_dimensions}-dimensional vectors."
            ),
            fix=(
                "Leave EMBEDDING_DIMENSIONS blank so it resolves from the "
                f"catalog, or set it to {catalog_dimensions}. A mismatch makes "
                "ai-engine refuse every write to the LanceDB table."
            ),
        )

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

        # Get default model for provider. A typo here used to survive selection
        # and only surface as a runtime-health `reason`; it is a misconfigured
        # field like any other, so it degrades with a structured code.
        default_model = ai_config.get_default_model(provider, "llm")
        model = os.getenv("LLM_MODEL") or default_model
        self._known_model_config(model, "LLM_MODEL")

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
                # The `fix` names no provider of its own, by design. It used to
                # append a hand-written note asserting that one specific
                # provider served no embeddings; that note was false, and
                # because it was a string literal rather than a catalog lookup,
                # correcting the catalog did not correct it. Everything this
                # message says about capability now comes from `capable`, which
                # is derived from config/ai_models.yaml on every call.
                cause=f"EMBEDDING_PROVIDER={provider!r} serves no embedding models.",
                fix=(
                    f"Choose one of: {', '.join(capable) or '(none in the catalog)'}. "
                    "That list comes from config/ai_models.yaml, so it is current "
                    "by construction."
                ),
            )

        # Get default model for provider
        default_model = ai_config.get_default_model(provider, "embedding")
        model = os.getenv("EMBEDDING_MODEL") or default_model
        model_config = self._known_model_config(model, "EMBEDDING_MODEL")

        # Vector width comes from the catalog and must agree with any override.
        dimensions = self._resolve_embedding_dimensions(model, model_config)

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
            base_url=self._base_url_for(provider, "EMBEDDING_BASE_URL"),
            dimensions=dimensions,
            dimensions_configurable=bool(model_config.get("dimensions_configurable")),
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
            # openrouter reuses the OpenAI adapter: /v1/embeddings is
            # OpenAI-compatible, and OpenAIEmbeddingProvider already keys off
            # `openrouter.ai` in the base_url for OPENROUTER_API_KEY and the
            # HTTP-Referer/X-Title headers. This entry was deleted on
            # 2026-08-04 as "unreachable, and would issue requests against
            # endpoints that do not exist" -- both halves wrong: it was
            # unreachable only because the capability check above was reading a
            # wrong catalog flag, and the endpoint exists and answers 200.
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
                # Same rule as `_resolve_embedding_dimensions`: no inference.
                # Carrying the *previous* model's width forward would index the
                # new model's vectors under the old model's dimension.
                if model_config.get("dimensions") is None:
                    raise ProviderSelectionError(
                        code="embedding_dimensions_unknown",
                        cause=(
                            f"The catalog entry for embedding model "
                            f"{target_model!r} declares no 'dimensions'."
                        ),
                        fix=(
                            f"Add 'dimensions' to the {target_model!r} entry in "
                            "config/ai_models.yaml before switching to it."
                        ),
                    )
                dimensions = int(model_config["dimensions"])
                configurable = bool(model_config.get("dimensions_configurable"))
            elif current is not None:
                target_model = current.model
                dimensions = current_dimensions
                configurable = current.dimensions_configurable
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
            # Must move with `model`: a stale flag would either drop the
            # `dimensions` parameter for a model that needs it or send it to
            # one that rejects it.
            current.dimensions_configurable = configurable
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

    @staticmethod
    def _redacted_state(state: dict) -> dict:
        """Last line of defence before a runtime state reaches a body or a log.

        Applied to the whole dict rather than at each construction site so a
        newly added `reason` string cannot reintroduce the leak by being
        written somewhere this file does not yet think about.
        """
        for key in ("reason", "server_url", "command"):
            if isinstance(state.get(key), str):
                state[key] = redact_url_userinfo(state[key])
        return state

    def get_provider_runtime_state(
        self, provider: str, capability: str = "llm"
    ) -> dict:
        """Return runtime availability details, safe to serve and to log."""
        return self._redacted_state(
            self._raw_provider_runtime_state(provider, capability)
        )

    def _raw_provider_runtime_state(
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
        return self._redacted_state(state)

    def _ensure_current_provider_ready(self, capability: str) -> None:
        """Gate every request path on a chosen *and* usable provider.

        The unavailable case raises `ProviderUnavailableError` rather than a
        bare `ValueError` so callers cannot confuse "the provider you chose is
        not there" with a transient provider-side error and fall back to a
        best-effort placeholder. `switch_provider` keeps raising `ValueError`
        for the same condition: that is a caller error on a mutation request,
        not the service answering with nothing.
        """
        error = self._selection_error(capability)
        if error is not None:
            raise error
        state = self._current_runtime_state(capability)
        if not state["available"]:
            config = self.llm_config if capability == "llm" else self.embedding_config
            provider = getattr(config, "provider", None)
            raise ProviderUnavailableError(
                cause=(
                    f"The selected {capability} provider {provider!r} is not usable: "
                    f"{redact_url_userinfo(state.get('reason')) or 'unknown reason'}"
                ),
                fix=(
                    "Make that provider usable (install/authenticate its CLI, set "
                    "its API key, or pull the model), or choose another one with "
                    "./scripts/cli.py configure-provider. This service does not "
                    "substitute a provider for you."
                ),
            )

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
        # Everything below this line is destined for `GET /health` (which is
        # unauthenticated) and for log lines, so it carries the redacted form.
        # `base_url` itself stays intact for the request.
        safe_url = redact_url_userinfo(base_url)

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
                "server_url": safe_url,
                # httpx embeds the request URL in several of its exception
                # messages, so redact the composed string, not just the prefix.
                "reason": redact_url_userinfo(
                    f"Ollama is not reachable at {base_url}: {exc}"
                ),
            }

        installed_models = self._ollama_installed_models(payload)
        if model not in installed_models:
            return {
                "available": False,
                "server_url": safe_url,
                "installed_models": sorted(installed_models),
                "reason": f"Ollama model {model} is not installed at {safe_url}",
            }

        return {
            "available": True,
            "server_url": safe_url,
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
