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
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException

from config.config_loader import get_ai_config
from services.ai_engine.app import main as ai_main
from services.ai_engine.app.clustering.pipeline import (
    UNLABELED_CLUSTER_NAME,
    Cluster,
    Tab,
    TabClusterer,
)
from services.ai_engine.app.core.llm_client import (
    EmbeddingConfig,
    LLMClient,
    LLMConfig,
    ProviderSelectionError,
    ProviderUnavailableError,
    redact_url_userinfo,
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


# --------------------------------------------------------------------------- #
# R1 -- a selection failure is never absorbed by a best-effort handler
#
# The acceptance criterion is "it does not answer requests using a provider
# nobody chose". Refusing to *select* one is only half of that: a caller that
# catches Exception around the call site re-creates the whole defect, because
# the request then completes with a placeholder that reads like a result. These
# tests assert on what the caller sees, not on what the client raises.
# --------------------------------------------------------------------------- #
def _tabs(count: int):
    import numpy as np

    return [
        Tab(
            url=f"https://example{i}.test/page",
            title=f"Tab {i}",
            content="captured text",
            embedding=np.full(8, float(i)),
        )
        for i in range(count)
    ]


class _ExplodingLLMClient:
    """LLM client whose generation fails for a provider-side reason."""

    def __init__(self, error):
        self.error = error

    async def generate(self, *_args, **_kwargs):
        raise self.error

    async def embed(self, texts):
        return [[0.0] * 8 for _ in texts]


@pytest.mark.asyncio
async def test_cluster_label_does_not_absorb_an_unselected_provider(monkeypatch):
    """The exact reported defect: labels became "Cluster N" and nothing said so."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")  # embeddings work...
    clusterer = TabClusterer()
    clusterer.set_llm_client(LLMClient())  # ...but no LLM was chosen

    with pytest.raises(ProviderSelectionError) as exc:
        await clusterer.generate_cluster_label(Cluster(id=0, tabs=_tabs(3)))

    assert exc.value.code == "provider_not_selected"


@pytest.mark.asyncio
async def test_cluster_label_does_not_absorb_an_unavailable_provider(monkeypatch):
    """A *selected* provider that is merely unusable must not be papered over.

    This is the condition `services/ai-engine/MODULE.md` carried as a stub
    ("unavailable local subscription CLI providers should report unhealthy
    rather than silently falling back"). It reached the same `except
    Exception` as the unselected case and produced the same fake label.
    """
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("AI_PROVIDER", "claude_code")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setattr(
        "services.ai_engine.app.providers.agent_cli.shutil.which", lambda _c: None
    )
    clusterer = TabClusterer()
    clusterer.set_llm_client(LLMClient())

    with pytest.raises(ProviderUnavailableError) as exc:
        await clusterer.generate_cluster_label(Cluster(id=0, tabs=_tabs(3)))

    assert exc.value.code == "provider_unavailable"
    assert "claude_code" in exc.value.cause


@pytest.mark.asyncio
async def test_cluster_endpoint_refuses_instead_of_returning_placeholder_names(
    monkeypatch,
):
    """`POST /cluster` answered 200 with `["Cluster 0", "Cluster 1"]`."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    client = LLMClient()

    async def _embed(texts):
        """Embeddings succeed -- that is what made the failure invisible."""
        return [[float(i), 0.0, 0.0, 0.0] for i, _ in enumerate(texts)]

    monkeypatch.setattr(client, "embed", _embed)
    clusterer = TabClusterer(min_cluster_corpus=2)
    clusterer.set_llm_client(client)
    monkeypatch.setattr(ai_main, "llm_client", client)
    monkeypatch.setattr(ai_main, "clusterer", clusterer)

    request = ai_main.ClusterRequest(
        session_id="s",
        urls=[
            {"url": f"https://example{i}.test/p", "title": f"t{i}", "content": "c"}
            for i in range(6)
        ],
    )

    with pytest.raises(HTTPException) as exc:
        await ai_main.cluster_urls(request)

    assert exc.value.status_code == 503
    assert exc.value.detail["code"] == "provider_not_selected"
    assert "configure-provider" in exc.value.detail["fix"]


@pytest.mark.asyncio
async def test_transient_label_failure_is_counted_logged_and_surfaced(monkeypatch):
    """Best-effort is allowed -- silent is not (CLAUDE.md's batch convention)."""
    _force_service_name(monkeypatch)
    clusterer = TabClusterer(min_cluster_corpus=2)
    clusterer.set_llm_client(_ExplodingLLMClient(TimeoutError("provider timed out")))
    cluster = Cluster(id=0, tabs=_tabs(3))

    stream, handler = _attach_json_capture()
    try:
        name = await clusterer.generate_cluster_label(cluster)
    finally:
        logging.getLogger("taborganizer.ai-engine").removeHandler(handler)

    assert name == UNLABELED_CLUSTER_NAME.format(id=0)
    assert "provider timed out" in cluster.metadata["label_error"]
    assert clusterer.count_label_failures([cluster]) == 1
    assert clusterer.to_dict([cluster])[0]["label_error"]

    events = [e for e in _read_events(stream) if e["event"] == "cluster.label_failed"]
    assert len(events) == 1
    assert events[0]["cluster_id"] == 0


# --------------------------------------------------------------------------- #
# F2 -- no misconfigured field may blackhole the service at import
#
# `app/main.py` builds `LLMClient()` at module scope, so *any* exception out of
# the constructor kills the process before it can report why. The fix has to
# hold for the class of misconfiguration, not for the two fields someone
# happened to name: the matrix below is the statement of that.
# --------------------------------------------------------------------------- #
HOSTILE_ENV = {
    "unknown_embedding_model": (
        {"EMBEDDING_PROVIDER": "ollama", "EMBEDDING_MODEL": "nomic-embed-txt"},
        "embeddings",
        "model_unknown",
    ),
    "unknown_llm_model": (
        {"AI_PROVIDER": "ollama", "LLM_MODEL": "qwen-3"},
        "llm",
        "model_unknown",
    ),
    "non_numeric_dimensions": (
        {"EMBEDDING_PROVIDER": "ollama", "EMBEDDING_DIMENSIONS": "auto"},
        "embeddings",
        "embedding_dimensions_invalid",
    ),
    "mismatched_dimensions": (
        {
            "EMBEDDING_PROVIDER": "ollama",
            "EMBEDDING_MODEL": "nomic-embed-text",
            "EMBEDDING_DIMENSIONS": "1536",
        },
        "embeddings",
        "embedding_dimensions_mismatch",
    ),
    "unknown_embedding_provider": (
        {"EMBEDDING_PROVIDER": "olama"},
        "embeddings",
        "provider_unknown",
    ),
    "unknown_llm_provider": (
        {"AI_PROVIDER": "claude-code"},
        "llm",
        "provider_unknown",
    ),
}


@pytest.mark.parametrize("case", list(HOSTILE_ENV))
def test_misconfigured_field_degrades_with_a_structured_error(monkeypatch, case):
    env, role, expected_code = HOSTILE_ENV[case]
    _clear_provider_env(monkeypatch)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    client = LLMClient()  # must not raise -- this runs at module scope

    key = "llm" if role == "llm" else "embeddings"
    config = client.llm_config if role == "llm" else client.embedding_config
    assert config is None, "a misconfigured role must not resolve to a substitute"
    error = client.get_provider_info()[key]["error"]
    assert error["code"] == expected_code
    assert error["cause"] and error["fix"]


def test_an_unforeseen_config_failure_still_degrades(monkeypatch):
    """The catch-all, so the *next* unhandled field is not another blackhole."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")

    def _explode(*_args, **_kwargs):
        raise KeyError("some field nobody thought about")

    monkeypatch.setattr(LLMClient, "_base_url_for", _explode)

    client = LLMClient()

    assert client.embedding_config is None
    error = client.get_provider_info()["embeddings"]["error"]
    assert error["code"] == "provider_config_invalid"
    assert "KeyError" in error["cause"]


def test_ai_engine_imports_under_a_typod_embedding_model():
    """Drives the real module-scope construction, in a real fresh interpreter.

    Asserting on `LLMClient()` alone would not prove the service survives: the
    reported crash was at *import* of `services.ai_engine.app.main`, and only a
    fresh interpreter can show that.
    """
    repo_root = Path(__file__).resolve().parents[2]
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": str(repo_root),
        "EMBEDDING_PROVIDER": "ollama",
        "EMBEDDING_MODEL": "nomic-embed-txt",
        "AI_ENGINE_ALLOW_UNAUTHENTICATED": "true",
        "VECTOR_DB_PATH": "/tmp/provider-routing-import-probe",
    }
    probe = (
        "import services.ai_engine.app.main as m;"
        "print(m.llm_client.embedding_config_error.code)"
    )

    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stderr
    assert "model_unknown" in result.stdout


# --------------------------------------------------------------------------- #
# F4 -- the catalog is the only source of truth for embedding dimensions
# --------------------------------------------------------------------------- #
def test_matching_dimensions_override_is_accepted(monkeypatch):
    """The happy path the fail-closed checks must not break."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_MODEL", "nomic-embed-text")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "768")

    client = LLMClient()

    assert client.embedding_config.dimensions == 768
    assert client.embedding_config_error is None


def test_dimensions_missing_from_the_catalog_are_never_inferred(monkeypatch):
    """`model_config.get("dimensions", 1536)` invented a width for any model."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_MODEL", "nomic-embed-text")
    entry = dict(get_ai_config().config["models"]["nomic-embed-text"])
    entry.pop("dimensions", None)
    monkeypatch.setitem(get_ai_config().config["models"], "nomic-embed-text", entry)

    client = LLMClient()

    assert client.embedding_config is None
    error = client.get_provider_info()["embeddings"]["error"]
    assert error["code"] == "embedding_dimensions_unknown"
    assert "1536" not in json.dumps(error)


def test_announce_surface_never_reports_a_width_the_model_contradicts(monkeypatch):
    """`/health` announced dimensions: 1536 for a 768-dimensional model."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_MODEL", "nomic-embed-text")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1536")

    summary = LLMClient().get_active_providers()["embedding"]

    assert summary["provider"] is None
    assert summary.get("dimensions") != 1536
    assert summary["error"]["code"] == "embedding_dimensions_mismatch"


def test_switching_to_a_model_without_catalog_dimensions_is_refused(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    client = LLMClient()
    entry = dict(get_ai_config().config["models"]["nomic-embed-text"])
    entry.pop("dimensions", None)
    monkeypatch.setitem(get_ai_config().config["models"], "nomic-embed-text", entry)

    with pytest.raises(ProviderSelectionError) as exc:
        client.switch_provider(
            embedding_provider="ollama", embedding_model="nomic-embed-text"
        )

    assert exc.value.code == "embedding_dimensions_unknown"


# --------------------------------------------------------------------------- #
# F3 -- /health is unauthenticated, so nothing on it may carry a credential
#
# The previous canaries were an API key and a service token, both of which live
# in their own env vars and never reach these strings. URL userinfo is the
# credential shape that does: it is embedded in a value (`OLLAMA_HOST`) whose
# whole purpose is to be echoed back as a diagnostic.
# --------------------------------------------------------------------------- #
CREDENTIALED_OLLAMA_HOST = "http://admin:sup3rsecret@ollama.internal:11434"


@pytest.mark.parametrize(
    "text, expected",
    [
        (CREDENTIALED_OLLAMA_HOST, "http://***@ollama.internal:11434"),
        ("https://token@host/x", "https://***@host/x"),
        ("reached http://u:p@a/ and http://v:q@b/", "reached http://***@a/ and http://***@b/"),
        ("http://ollama:11434", "http://ollama:11434"),
        ("no url here", "no url here"),
    ],
)
def test_redact_url_userinfo(text, expected):
    assert redact_url_userinfo(text) == expected


@pytest.mark.asyncio
async def test_health_never_leaks_credentials_embedded_in_a_url(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_HOST", CREDENTIALED_OLLAMA_HOST)
    _block_ollama_probe(monkeypatch)
    monkeypatch.setattr(ai_main, "chatbot", _FakeChatbot())
    monkeypatch.setattr(ai_main, "llm_client", LLMClient())

    body = json.dumps(await ai_main.health())

    assert "sup3rsecret" not in body
    assert "admin" not in body
    # still diagnosable: the host survives, only the credential is gone
    assert "ollama.internal" in body


def test_startup_log_never_leaks_credentials_embedded_in_a_url(monkeypatch):
    """`provider.unusable_at_startup` echoes the same reason strings."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_HOST", CREDENTIALED_OLLAMA_HOST)
    _block_ollama_probe(monkeypatch)
    client = LLMClient()
    monkeypatch.setattr(ai_main, "llm_client", client)
    _force_service_name(monkeypatch)

    stream, handler = _attach_json_capture()
    try:
        runtime = client.get_runtime_health()
        ai_main._announce_active_providers()
        from services.observability import log_event

        log_event(
            "provider.unusable_at_startup",
            level=logging.ERROR,
            llm_reason=runtime["llm"].get("reason"),
        )
    finally:
        logging.getLogger("taborganizer.ai-engine").removeHandler(handler)

    assert "sup3rsecret" not in stream.getvalue()


@pytest.mark.asyncio
async def test_request_refusal_never_leaks_credentials_embedded_in_a_url(monkeypatch):
    """The refusal reason is built from the same base URL."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_HOST", CREDENTIALED_OLLAMA_HOST)
    _block_ollama_probe(monkeypatch)
    client = LLMClient()

    with pytest.raises(ProviderUnavailableError) as exc:
        await client.embed(["captured page text"])

    assert "sup3rsecret" not in json.dumps(exc.value.to_dict())


# --------------------------------------------------------------------------- #
# F5 -- the capability mirror must agree with the catalog, entry by entry
# --------------------------------------------------------------------------- #
def test_provider_capability_mirror_agrees_with_the_catalog():
    """`LLMClient.PROVIDERS` backs `GET /providers`; drift is a false advertisement.

    Asserted for every provider and every capability rather than for the entry
    that happened to be wrong (openrouter/embeddings), so the next divergence
    is caught wherever it appears.
    """
    ai_config = get_ai_config()
    catalog = set(ai_config.get_all_providers())

    assert set(LLMClient.PROVIDERS) == catalog, (
        "the mirror and the catalog disagree about which providers exist"
    )

    for provider in sorted(catalog):
        mirror = LLMClient.PROVIDERS[provider]
        provider_config = ai_config.get_provider_config(provider)
        for capability in ("llm", "embeddings"):
            assert mirror.get(capability, False) == ai_config.is_provider_supported(
                provider, capability
            ), f"{provider}.{capability} disagrees with config/ai_models.yaml"
        assert mirror.get("local", False) == str(
            provider_config.get("type", "")
        ).startswith("local"), f"{provider}.local disagrees with its catalog type"
        assert mirror.get("subscription", False) == (
            provider_config.get("cost_model") == "subscription"
        ), f"{provider}.subscription disagrees with its catalog cost_model"


def test_a_refused_embedding_switch_leaves_the_selection_untouched(monkeypatch):
    """Refusing must not be a half-apply: the old provider stays selected.

    `switch_provider`'s explicit `is_provider_supported(..., "embeddings")`
    guard is shadowed by `get_provider_runtime_state()`, which performs the
    same catalog check and raises the same message -- so no test can
    distinguish the guard's presence. What *is* observable, and what the spec
    actually cares about, is that a refused switch substitutes nothing.
    """
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    client = LLMClient()
    before = (client.embedding_config.provider, client.embedding_config.model)

    for provider in ("openrouter", "claude_code", "anthropic", "deepseek"):
        with pytest.raises(ValueError, match="does not support embeddings"):
            client.switch_provider(embedding_provider=provider)
        assert (
            client.embedding_config.provider,
            client.embedding_config.model,
        ) == before
