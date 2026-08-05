"""Provider routing: no implicit selection, no silent fallback, always announced.

Contract: `routing:` in `config/ai_models.yaml` (`silent_fallback: false`,
`explicit_opt_in_required`, `announce_active_provider: true`). Behaviour spec:
`docs/SPEC-provider-routing.md` R1, R2, R4.

The failure these tests exist to prevent is substitution: the service quietly
answering with a provider nobody chose. That is the WI0-B1 defect class one
layer up -- the system reports success while doing something the user did not
ask for. So every test here asserts on the *absence of a substitute*
(`llm_config is None`, `providers.llm.provider is None`), not merely on an
error being raised: an error can be raised for the wrong reason, but a
substitute cannot appear without the fallback being back.
"""

import io
import json
import logging

import httpx
import pytest

from config.config_loader import get_ai_config
from services.ai_engine.app import main as ai_main
from services.ai_engine.app.core.llm_client import (
    EmbeddingConfig,
    LLMClient,
    LLMConfig,
    ProviderSelectionError,
)

PROVIDER_ENV_VARS = (
    "AI_PROVIDER",
    "EMBEDDING_PROVIDER",
    "LLM_MODEL",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSIONS",
    "LLM_BASE_URL",
    "EMBEDDING_BASE_URL",
)


def _clear_provider_env(monkeypatch):
    """Simulate a fresh checkout: nothing chosen, nothing inherited."""
    for name in PROVIDER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _block_ollama_probe(monkeypatch):
    """Keep runtime-health checks off the network and deterministic."""

    def _refuse(*_args, **_kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("services.ai_engine.app.core.llm_client.httpx.get", _refuse)


def _catalog_embedding_providers() -> list[str]:
    ai_config = get_ai_config()
    return [
        provider
        for provider in ai_config.get_all_providers()
        if ai_config.is_provider_supported(provider, "embeddings")
    ]


def _named_choices(fix: str) -> str:
    """The 'Choose one of: ...' segment of an R2 fix string.

    Sliced out deliberately: the rest of the fix text mentions openrouter by
    name ("serves NO embedding models"), so a naive substring check over the
    whole string would pass whether or not openrouter was offered as a choice.
    """
    return fix.split("Choose one of: ", 1)[1].split(".", 1)[0]


class _FakeChatbot:
    """Stands in for LanceDB so /health exercises only the provider branch."""

    db_uri = "/tmp/lancedb"
    TABLE_NAME = "tab_organizer_docs"
    table = object()


def _attach_json_capture():
    """Capture ai-engine JSON log lines (the service logger does not propagate)."""
    from services.observability import _JsonFormatter

    logger = logging.getLogger("taborganizer.ai-engine")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(_JsonFormatter())
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return stream, handler


def _read_events(stream: io.StringIO):
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def _force_service_name(monkeypatch):
    from services import observability as obs

    monkeypatch.setattr(obs, "_service_name", "ai-engine")


# --------------------------------------------------------------------------- #
# R1 -- no implicit provider, ever
# --------------------------------------------------------------------------- #
def test_unset_ai_provider_selects_nothing(monkeypatch):
    """`AI_PROVIDER` unset must leave the LLM role empty, not 'openrouter'."""
    _clear_provider_env(monkeypatch)

    client = LLMClient()

    assert client.llm_config is None
    assert client.get_active_providers()["llm"]["provider"] is None

    error = client.get_provider_info()["llm"]["error"]
    assert error["code"] == "provider_not_selected"
    assert "AI_PROVIDER is not set" in error["cause"]
    assert "configure-provider" in error["fix"]


def test_unset_embedding_provider_selects_nothing(monkeypatch):
    _clear_provider_env(monkeypatch)

    client = LLMClient()

    assert client.embedding_config is None
    assert client.get_active_providers()["embedding"]["provider"] is None
    assert (
        client.get_provider_info()["embeddings"]["error"]["code"]
        == "provider_not_selected"
    )


def test_provider_not_selected_fix_names_the_preference_order(monkeypatch):
    """The fix text is catalog-derived, not a hand-maintained copy."""
    _clear_provider_env(monkeypatch)
    routing = get_ai_config().config["routing"]

    fix = LLMClient().get_provider_info()["llm"]["error"]["fix"]

    for provider in routing["llm_preference_order"]:
        assert provider in fix
    assert "subscription" in fix


def test_no_llm_provider_means_no_provider_is_constructed(monkeypatch):
    """The request path fails closed instead of instantiating a substitute."""
    _clear_provider_env(monkeypatch)
    client = LLMClient()

    with pytest.raises(ProviderSelectionError) as exc:
        _ = client.llm

    assert exc.value.code == "provider_not_selected"


@pytest.mark.asyncio
async def test_embed_refuses_when_no_embedding_provider_is_selected(monkeypatch):
    _clear_provider_env(monkeypatch)
    _block_ollama_probe(monkeypatch)
    client = LLMClient()

    with pytest.raises(ProviderSelectionError) as exc:
        await client.embed(["some captured page text"])

    assert exc.value.code == "provider_not_selected"


def test_unknown_provider_is_a_structured_error_not_an_import_crash(monkeypatch):
    """A typo must degrade the service, not kill it before it can report why."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("AI_PROVIDER", "opnerouter")

    client = LLMClient()

    assert client.llm_config is None
    error = client.get_provider_info()["llm"]["error"]
    assert error["code"] == "provider_unknown"
    assert "opnerouter" in error["cause"]


# --------------------------------------------------------------------------- #
# R2 -- the silent embedding fallback is deleted
# --------------------------------------------------------------------------- #
def test_embedding_provider_that_cannot_embed_is_an_error_not_a_swap(monkeypatch):
    """This is the deleted fallback. It used to rewrite the provider to ollama."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("AI_PROVIDER", "claude_code")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "claude_code")

    client = LLMClient()

    assert client.embedding_config is None
    assert client.get_active_providers()["embedding"]["provider"] is None

    error = client.get_provider_info()["embeddings"]["error"]
    assert error["code"] == "embedding_provider_cannot_embed"
    assert "claude_code" in error["cause"]


def test_embedding_error_names_every_catalog_provider_that_can_embed(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "claude_code")
    capable = _catalog_embedding_providers()
    assert (
        capable
    ), "catalog declares no embedding provider; the assert below is vacuous"

    error = LLMClient().get_provider_info()["embeddings"]["error"]
    choices = _named_choices(error["fix"])

    for provider in capable:
        assert provider in choices
    # openrouter serves no embedding models (verified 2026-08-04), so it must
    # not be offered as a choice even though the note below mentions it.
    assert "openrouter" not in capable
    assert "openrouter" not in choices
    assert "openrouter serves NO embedding models" in error["fix"]


def test_embedding_error_list_follows_the_catalog(monkeypatch):
    """Flip the catalog and the named list moves with it -- no hardcoded copy."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "claude_code")
    supports = get_ai_config().config["providers"]["ollama"]["supports"]

    assert "ollama" in _named_choices(
        LLMClient().get_provider_info()["embeddings"]["error"]["fix"]
    )

    monkeypatch.setitem(supports, "embeddings", False)

    assert "ollama" not in _named_choices(
        LLMClient().get_provider_info()["embeddings"]["error"]["fix"]
    )


def test_embedding_provider_that_can_embed_is_used_as_given(monkeypatch):
    """The removal must not break the happy path it was pretending to protect."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("AI_PROVIDER", "claude_code")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")

    client = LLMClient()

    assert client.embedding_config is not None
    assert client.embedding_config.provider == "ollama"
    assert client.embedding_config.model == "nomic-embed-text"
    assert client.embedding_config_error is None


# --------------------------------------------------------------------------- #
# R4 -- announce the active provider
# --------------------------------------------------------------------------- #
def test_startup_announces_one_provider_active_line_per_role(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("AI_PROVIDER", "claude_code")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setattr(ai_main, "llm_client", LLMClient())
    _force_service_name(monkeypatch)

    stream, handler = _attach_json_capture()
    try:
        ai_main._announce_active_providers()
    finally:
        logging.getLogger("taborganizer.ai-engine").removeHandler(handler)

    active = [e for e in _read_events(stream) if e["event"] == "provider.active"]
    assert {event["role"] for event in active} == {"llm", "embedding"}

    llm = next(event for event in active if event["role"] == "llm")
    assert llm["provider"] == "claude_code"
    assert llm["model"] == "sonnet"
    assert llm["cost_model"] == "subscription"
    assert llm["tier"] == "mid"

    embedding = next(event for event in active if event["role"] == "embedding")
    assert embedding["provider"] == "ollama"
    assert embedding["model"] == "nomic-embed-text"
    assert embedding["cost_model"] == "free_local"
    assert embedding["dimensions"] == 768


@pytest.mark.asyncio
async def test_lifespan_announces_the_active_providers(monkeypatch):
    """The announcement must be *wired into startup*, not merely available.

    Asserting on `_announce_active_providers()` alone would pass even if
    nothing ever called it -- which is precisely how a surface goes silent
    without a single test noticing.
    """
    _clear_provider_env(monkeypatch)
    _block_ollama_probe(monkeypatch)
    monkeypatch.setenv("AI_PROVIDER", "claude_code")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setattr(ai_main, "llm_client", LLMClient())
    _force_service_name(monkeypatch)

    stream, handler = _attach_json_capture()
    try:
        async with ai_main.lifespan(ai_main.app):
            pass
    finally:
        logging.getLogger("taborganizer.ai-engine").removeHandler(handler)

    active = [e for e in _read_events(stream) if e["event"] == "provider.active"]
    assert {event["role"] for event in active} == {"llm", "embedding"}
    assert next(e for e in active if e["role"] == "llm")["provider"] == "claude_code"


def test_provider_active_announcement_never_carries_an_api_key(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("AI_PROVIDER", "openrouter")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-canary-must-not-be-logged")
    monkeypatch.setattr(ai_main, "llm_client", LLMClient())
    _force_service_name(monkeypatch)

    stream, handler = _attach_json_capture()
    try:
        ai_main._announce_active_providers()
    finally:
        logging.getLogger("taborganizer.ai-engine").removeHandler(handler)

    raw = stream.getvalue()
    assert "provider.active" in raw
    assert "sk-or-canary-must-not-be-logged" not in raw


def test_unselected_role_is_still_announced_with_its_fix(monkeypatch):
    """Silence would be the failure mode: nothing chosen must still be said."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setattr(ai_main, "llm_client", LLMClient())
    _force_service_name(monkeypatch)

    stream, handler = _attach_json_capture()
    try:
        ai_main._announce_active_providers()
    finally:
        logging.getLogger("taborganizer.ai-engine").removeHandler(handler)

    active = [e for e in _read_events(stream) if e["event"] == "provider.active"]
    assert {event["role"] for event in active} == {"llm", "embedding"}
    for event in active:
        assert event["provider"] is None
        assert event["level"] == "error"
        assert event["error"]["code"] == "provider_not_selected"


@pytest.mark.asyncio
async def test_health_reports_the_active_provider_for_both_roles(monkeypatch):
    _block_ollama_probe(monkeypatch)
    monkeypatch.setattr(ai_main, "chatbot", _FakeChatbot())
    monkeypatch.setattr(
        ai_main,
        "llm_client",
        LLMClient(
            LLMConfig(provider="claude_code", model="sonnet"),
            EmbeddingConfig(
                provider="ollama", model="nomic-embed-text", dimensions=768
            ),
        ),
    )

    response = await ai_main.health()

    assert response["providers"]["llm"]["provider"] == "claude_code"
    assert response["providers"]["llm"]["model"] == "sonnet"
    assert response["providers"]["llm"]["cost_model"] == "subscription"
    assert response["providers"]["embedding"]["provider"] == "ollama"
    assert response["providers"]["embedding"]["model"] == "nomic-embed-text"
    assert response["providers"]["embedding"]["cost_model"] == "free_local"


@pytest.mark.asyncio
async def test_health_is_degraded_and_names_the_fix_when_nothing_is_selected(
    monkeypatch,
):
    _clear_provider_env(monkeypatch)
    _block_ollama_probe(monkeypatch)
    monkeypatch.setattr(ai_main, "chatbot", _FakeChatbot())
    monkeypatch.setattr(ai_main, "llm_client", LLMClient())

    response = await ai_main.health()

    assert response["status"] == "degraded"
    for role in ("llm", "embedding"):
        block = response["providers"][role]
        assert block["provider"] is None
        assert block["error"]["code"] == "provider_not_selected"
        assert "configure-provider" in block["error"]["fix"]


@pytest.mark.asyncio
async def test_health_never_leaks_a_key_or_a_token(monkeypatch):
    """/health is the one unauthenticated endpoint on this service."""
    _clear_provider_env(monkeypatch)
    _block_ollama_probe(monkeypatch)
    monkeypatch.setenv("AI_PROVIDER", "openrouter")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-canary-must-not-be-served")
    monkeypatch.setenv("AI_ENGINE_API_TOKEN", "token-canary-must-not-be-served")
    monkeypatch.setattr(ai_main, "chatbot", _FakeChatbot())
    monkeypatch.setattr(ai_main, "llm_client", LLMClient())

    body = json.dumps(await ai_main.health())

    assert "openrouter" in body
    assert "sk-or-canary-must-not-be-served" not in body
    assert "token-canary-must-not-be-served" not in body


# --------------------------------------------------------------------------- #
# R3 -- the invariant, stated rather than emergent
# --------------------------------------------------------------------------- #
def test_opt_in_providers_are_never_reached_without_an_explicit_choice(monkeypatch):
    """Every `explicit_opt_in_required` provider stays unreachable by default.

    R3 is guaranteed by R1 on the Python side rather than by a mechanism of its
    own. This test is the statement of that guarantee: if a future router ever
    starts picking a provider when the env var is empty, it fails here.
    """
    _clear_provider_env(monkeypatch)
    opt_in = get_ai_config().config["routing"]["explicit_opt_in_required"]
    assert opt_in, "catalog declares no opt-in providers; the assert below is vacuous"

    client = LLMClient()

    active = client.get_active_providers()
    assert active["llm"]["provider"] not in opt_in
    assert active["embedding"]["provider"] not in opt_in
    assert active["llm"]["provider"] is None
    assert active["embedding"]["provider"] is None


def test_switching_to_an_opt_in_provider_is_an_explicit_choice(monkeypatch):
    """Naming a provider is consent; the env var is not the only lawful record."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    client = LLMClient()

    client.switch_provider(llm_provider="openrouter", llm_model="openrouter/auto")

    assert client.llm_config.provider == "openrouter"
    assert client.llm_config_error is None
    assert client.get_active_providers()["llm"]["cost_model"] == "metered"
