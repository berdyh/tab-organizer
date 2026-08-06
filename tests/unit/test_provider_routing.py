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

import inspect
import io
import json
import logging
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
import yaml
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

    Sliced out deliberately, so an assertion about which providers are OFFERED
    cannot be satisfied by a provider name appearing elsewhere in the message.
    That was not hypothetical: the fix text used to end with a hand-written
    "Note openrouter serves NO embedding models", and a naive substring check
    over the whole string would have passed whether or not openrouter was
    actually offered as a choice. That note is gone (it was false), but the
    slicing stays -- the next hardcoded name in an error string should not be
    able to fool this either.
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

    # The complement matters as much as the list: a provider the catalog says
    # cannot embed must never appear as a choice. Derived from the catalog
    # rather than naming a provider, because the version of this assertion that
    # hardcoded `openrouter` outlived the (false) catalog entry it was written
    # against and had to be deleted here.
    incapable = [
        provider
        for provider in get_ai_config().get_all_providers()
        if provider not in capable
    ]
    assert incapable, "every provider can embed; the assert below is vacuous"
    for provider in incapable:
        assert provider not in choices

    # The fix text must name no provider of its own. It used to append
    # "openrouter serves NO embedding models -- verified 2026-08-04", which was
    # false; a hand-written provider name in an error string is a second source
    # of truth that nothing keeps current.
    assert "serves NO embedding models" not in error["fix"]


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

    # openrouter was in this list until 2026-08-05 on the strength of a false
    # catalog entry; it embeds, so a refusal is no longer the correct outcome
    # for it. The remaining three are LLM-only providers with no embedding
    # adapter at all.
    for provider in ("claude_code", "anthropic", "deepseek"):
        with pytest.raises(ValueError, match="does not support embeddings"):
            client.switch_provider(embedding_provider=provider)
        assert (
            client.embedding_config.provider,
            client.embedding_config.model,
        ) == before


# --------------------------------------------------------------------------- #
# F6 -- `supports.embeddings` must agree with what the PROVIDER PATH does
#
# Written after the 2026-08-04/05 correction, in which the catalog carried
# `openrouter.supports.embeddings: false` annotated "verified", and every
# downstream artifact -- the `PROVIDERS` mirror, the adapter map, six documents
# and two tests -- faithfully repeated it for a day. Nothing failed, because
# every check compared the claim against another copy of the claim.
#
# The two tests below close the two halves of that blind spot:
#
#   * the hermetic one asserts the flag agrees with the CODE that would have to
#     serve the capability. It catches the b1fe195 half of the regression -- an
#     adapter deleted as "unreachable" on the strength of the flag -- and it
#     runs everywhere, on every change.
#   * the live one asserts the flag agrees with the REMOTE API. It is the only
#     test that could have caught the original error, because the original
#     error was a false belief about a remote service, and no amount of local
#     cross-checking can refute that. It is deliberately a network call and is
#     marked so it skips without credentials rather than passing vacuously.
# --------------------------------------------------------------------------- #
def _embedding_adapter_map() -> dict:
    """The provider -> embedding-adapter mapping `_create_embedding_provider` uses.

    Read out of the function's own source rather than duplicated here: a second
    copy of the map is exactly the kind of mirror this test exists to police.
    """
    import inspect
    import re

    source = inspect.getsource(LLMClient._create_embedding_provider)
    body = source.split("providers = {", 1)[1].split("}", 1)[0]
    return dict(re.findall(r'"(\w+)":\s*(\w+)', body))


# Providers where the catalog flag and the adapter map are KNOWN to disagree
# and the disagreement has not been resolved by the only method that can
# resolve it -- calling the provider's embedding endpoint with a real key.
#
# Each entry is a debt, not a dispensation. The staleness check below fails the
# moment an entry stops diverging, so a resolved case cannot linger here and
# turn the exception list into a rubber stamp.
UNRESOLVED_EMBEDDING_CAPABILITY = {
    # Found by this test on 2026-08-05, pre-existing and unrelated to the
    # openrouter correction that prompted it: `supports.embeddings: false` in
    # the catalog, but `DeepSeekEmbeddingProvider` is wired into the adapter
    # map and exported from `providers/__init__.py`.
    #
    # NOT resolved here, deliberately. No DEEPSEEK_API_KEY is configured, and
    # an unauthenticated probe cannot settle it: api.deepseek.com returns 401
    # "Authentication Fails" for /v1/embeddings and /v1/chat/completions
    # alike, so the gateway authenticates before routing and a 401 is not
    # evidence that the surface exists. Guessing from that -- or from the
    # adapter's mere existence -- would repeat exactly the error this test was
    # written to catch: inferring a remote capability from a local signal that
    # cannot see it.
    #
    # To resolve: set DEEPSEEK_API_KEY, POST /v1/embeddings, and make the
    # catalog and the adapter map agree with whatever it answers.
    "deepseek",
}


def test_embedding_capability_flag_agrees_with_the_adapter_map():
    """A `supports.embeddings` flag with no adapter behind it is a lie, and so
    is an adapter for a provider the catalog says cannot embed.

    The catalog is the source of truth for capability, but a truth nothing
    implements is not serveable. `b1fe195` deleted the openrouter adapter as
    "unreachable" -- true only because the capability check above it was
    reading a wrong flag -- and no test noticed the map and the catalog had
    diverged.
    """
    ai_config = get_ai_config()
    adapters = _embedding_adapter_map()
    assert adapters, "could not parse the embedding adapter map"

    for provider in sorted(ai_config.get_all_providers()):
        if provider in UNRESOLVED_EMBEDDING_CAPABILITY:
            continue
        supported = ai_config.is_provider_supported(provider, "embeddings")
        has_adapter = provider in adapters
        assert supported == has_adapter, (
            f"{provider}: config/ai_models.yaml says supports.embeddings="
            f"{supported} but _create_embedding_provider "
            f"{'has' if has_adapter else 'has no'} adapter for it. "
            "One of the two is wrong -- decide which by calling the provider's "
            "embedding endpoint, not by reading a catalog listing."
        )


def test_the_unresolved_capability_list_is_not_stale():
    """An exception that no longer applies must be deleted, not inherited.

    Without this, `UNRESOLVED_EMBEDDING_CAPABILITY` would silently become a
    permanent hole: someone fixes deepseek, the entry stays, and the provider
    is exempt from the agreement check forever.
    """
    ai_config = get_ai_config()
    adapters = _embedding_adapter_map()

    for provider in sorted(UNRESOLVED_EMBEDDING_CAPABILITY):
        assert provider in ai_config.get_all_providers(), (
            f"{provider} is no longer a catalog provider; remove it from "
            "UNRESOLVED_EMBEDDING_CAPABILITY"
        )
        supported = ai_config.is_provider_supported(provider, "embeddings")
        assert supported != (provider in adapters), (
            f"{provider} no longer diverges: the catalog and the adapter map "
            "agree. Remove it from UNRESOLVED_EMBEDDING_CAPABILITY so the "
            "agreement check covers it again."
        )


@pytest.mark.requires_provider_credentials
@pytest.mark.asyncio
async def test_catalog_embedding_claim_matches_the_live_endpoint(monkeypatch):
    """Ask the endpoint, not a listing. THE test that would have caught this.

    Skipped without `OPENROUTER_API_KEY`, so it never passes vacuously -- an
    absent key yields a skip naming the reason, not a green tick.

    Deliberately a live network call. The original error was produced by
    substituting a cheaper local check (scan the /v1/models catalog listing for
    an embedding modality) for the real one (call /v1/embeddings). That listing
    covers the chat-completions surface only; the embedding ids asserted below
    do not appear in it and serve 200 anyway. Any hermetic version of this test
    would re-import the same blind spot.

    Uses the catalog's DEFAULT openrouter embedding model, so a default changed
    to something the provider does not actually serve fails here.
    """
    import os

    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set; cannot verify the live claim")

    ai_config = get_ai_config()
    assert ai_config.is_provider_supported("openrouter", "embeddings"), (
        "catalog says openrouter cannot embed; this test asserts the opposite "
        "and one of them is out of date -- rerun it and trust the endpoint"
    )

    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openrouter")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "768")

    client = LLMClient()
    assert client.embedding_config_error is None, client.embedding_config_error
    assert client.embedding_config.dimensions == 768

    vectors = await client.embed(["live catalog-agreement probe"])

    # The capability claim, and the width claim, both against the wire.
    assert len(vectors) == 1
    assert len(vectors[0]) == 768, (
        "openrouter honours a `dimensions` override; a native-width vector "
        "here means the adapter dropped the parameter, and every write to a "
        "768-d LanceDB table would be refused"
    )


@pytest.mark.asyncio
async def test_adapter_sends_dimensions_for_a_configurable_model():
    """Hermetic guard on the `dimensions` passthrough.

    The live test above proves the same thing against the real API, but it
    SKIPS without a key -- i.e. in CI, which is exactly where a silent
    regression would land. This one runs everywhere: it intercepts the request
    and asserts the parameter is on the wire.

    Why the parameter is load-bearing rather than cosmetic: the catalog's
    default openrouter embedding model is natively 4096-d. Dropped here, a
    `EMBEDDING_DIMENSIONS=768` deployment announces 768, receives 4096, and
    ai-engine refuses every write to the 768-wide table -- a failure that looks
    like a LanceDB problem and is actually a missing JSON key.
    """
    from services.ai_engine.app.core.llm_client import EmbeddingConfig
    from services.ai_engine.app.providers.openai import OpenAIEmbeddingProvider

    sent = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        width = sent.get("dimensions", 4096)
        return httpx.Response(200, json={"data": [{"embedding": [0.0] * width}]})

    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient

    def patched(*args, **kwargs):
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    provider = OpenAIEmbeddingProvider(
        EmbeddingConfig(
            provider="openrouter",
            model="qwen/qwen3-embedding-8b",
            api_key="sk-or-test",
            base_url="https://openrouter.ai/api/v1",
            dimensions=768,
            dimensions_configurable=True,
        )
    )
    httpx.AsyncClient = patched
    try:
        vectors = await provider.embed(["hermetic passthrough probe"])
    finally:
        httpx.AsyncClient = original

    assert sent.get("dimensions") == 768, (
        "the adapter dropped `dimensions`; a configurable model would return "
        f"its native width instead of the configured one. Sent: {sent}"
    )
    assert len(vectors[0]) == 768


@pytest.mark.asyncio
async def test_adapter_omits_dimensions_for_a_fixed_width_model():
    """The converse: a model that does not accept the parameter must not get it.

    Fixed-width embedding endpoints reject an unexpected `dimensions` key
    outright, so sending it unconditionally would break every provider that is
    not Matryoshka-capable.
    """
    from services.ai_engine.app.core.llm_client import EmbeddingConfig
    from services.ai_engine.app.providers.openai import OpenAIEmbeddingProvider

    sent = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(200, json={"data": [{"embedding": [0.0] * 1536}]})

    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient

    def patched(*args, **kwargs):
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    provider = OpenAIEmbeddingProvider(
        EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            api_key="sk-test",
            dimensions=1536,
            dimensions_configurable=False,
        )
    )
    httpx.AsyncClient = patched
    try:
        await provider.embed(["hermetic omission probe"])
    finally:
        httpx.AsyncClient = original

    assert "dimensions" not in sent, (
        "the adapter sent `dimensions` for a model the catalog does not mark "
        f"configurable; fixed-width endpoints reject it. Sent: {sent}"
    )


@pytest.mark.requires_provider_credentials
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model", sorted(get_ai_config().get_provider_models("openrouter", "embedding"))
)
async def test_every_registered_openrouter_embedding_model_serves_and_truncates(model):
    """Each registered id must actually exist AND honour `dimensions`.

    Parametrized off the catalog, so registering a model that openrouter does
    not serve -- the failure mode that started all this, in the other
    direction -- fails here rather than at a user's first index run. One
    small request per model.

    Truncation is asserted, not just a 200: every one of these is registered
    with `dimensions_configurable: true`, and that flag is what tells the
    adapter it may retarget the width to match an existing LanceDB table.
    """
    import os

    from services.ai_engine.app.core.llm_client import EmbeddingConfig
    from services.ai_engine.app.providers.openai import OpenAIEmbeddingProvider

    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set; cannot verify the live claim")

    model_config = get_ai_config().get_model_config(model)
    assert model_config.get("dimensions_configurable"), (
        f"{model} is registered without dimensions_configurable; this test "
        "assumes every openrouter embedding model was verified truncatable"
    )

    provider = OpenAIEmbeddingProvider(
        EmbeddingConfig(
            provider="openrouter",
            model=model,
            base_url="https://openrouter.ai/api/v1",
            dimensions=768,
            dimensions_configurable=True,
        )
    )
    vectors = await provider.embed(["registered-model truncation probe"])

    assert len(vectors[0]) == 768, (
        f"{model} returned {len(vectors[0])} floats for a 768 request; it "
        "cannot be truncated and must not carry dimensions_configurable"
    )


# --------------------------------------------------------------------------
# Catalog representation: a model is never shown, defaulted, or reasoned about
# without the provider route it travels on.
# --------------------------------------------------------------------------


def _catalog_models() -> dict:
    return get_ai_config().config.get("models", {})


def _subscription_families() -> set:
    """Families with at least one route on a subscription provider."""
    ai_config = get_ai_config()
    families = set()
    for model in _catalog_models():
        if ai_config.get_model_cost_model(model) == "subscription":
            families.add(ai_config.get_model_family(model))
    return families


def _metered_duplicate_routes() -> dict:
    """Routes a default must never land on: metered, and already paid for.

    Two ways a route qualifies, both read from the catalog:
      * its `model_family` also has a route on a subscription provider
        (`gpt-5.6-luna` on codex_cli vs `openai/gpt-5.6-luna` on openrouter);
      * it declares `superseded_by`, for the case where the subscription
        equivalent is a nearby model rather than the identical weights
        (`google/gemini-2.5-flash-lite` -> `gemini-3.1-flash-lite`).
    """
    ai_config = get_ai_config()
    subscription_families = _subscription_families()
    duplicates = {}
    for model, model_config in _catalog_models().items():
        if ai_config.get_model_cost_model(model) == "subscription":
            continue
        family = ai_config.get_model_family(model)
        reason = None
        if family in subscription_families:
            reason = f"family {family!r} is served by a subscription provider"
        elif model_config.get("superseded_by"):
            reason = f"superseded_by {model_config['superseded_by']!r}"
        if reason:
            duplicates[model] = reason
    return duplicates


def test_the_metered_duplicate_set_is_not_empty():
    """Non-vacuity guard for the invariant below.

    The rule test is a scan: if nothing in the catalog ever qualified as a
    metered duplicate it would pass forever while enforcing nothing. This
    repo has shipped four tests that could not fail; this is the cheap way
    not to ship a fifth.
    """
    duplicates = _metered_duplicate_routes()
    assert duplicates, (
        "no metered route is recognised as a duplicate of a subscription "
        "one, so the never-default rule below is vacuous"
    )
    assert "openai/gpt-5.6-luna" in duplicates, sorted(duplicates)
    assert "google/gemini-2.5-flash-lite" in duplicates, sorted(duplicates)


def test_no_metered_route_is_defaulted_when_the_family_has_a_subscription_route():
    """"Don't mix them up": registered is fine, DEFAULTED is not.

    The GPT-5.6 family is spent from the Codex subscription and the cheap
    Gemini tier from the Gemini CLI subscription. openrouter carries metered
    copies of both, and those copies stay registered on purpose -- they are
    the deliberate smoke-test path. What must never happen is arriving at one
    of them without asking: as a `recommended: true` badge, a provider's
    `default_models`, or a `defaults.use_cases` model. Each of those is a way
    the system spends money on something the user already bought.

    The rule is stated as data in `routing.metered_duplicate_policy`; this
    test is that policy executed.
    """
    ai_config = get_ai_config()
    assert (
        ai_config.config.get("routing", {}).get("metered_duplicate_policy")
        == "never_default"
    ), "the catalog no longer declares the policy this test enforces"

    duplicates = _metered_duplicate_routes()
    violations = []

    for model, reason in duplicates.items():
        if _catalog_models()[model].get("recommended"):
            violations.append(f"{model} is recommended: true ({reason})")

    for provider in ai_config.get_all_providers():
        defaults = ai_config.get_provider_config(provider).get("default_models") or {}
        for role, model in defaults.items():
            if model in duplicates:
                violations.append(
                    f"{provider}.default_models.{role} = {model} ({duplicates[model]})"
                )

    use_cases = ai_config.get_defaults().get("use_cases") or {}
    for use_case, entry in use_cases.items():
        model = (entry or {}).get("model")
        if model in duplicates:
            violations.append(
                f"defaults.use_cases.{use_case} = {model} ({duplicates[model]})"
            )

    assert not violations, (
        "metered copies of subscription models are being defaulted to:\n  "
        + "\n  ".join(sorted(violations))
    )


def test_the_same_weights_reachable_twice_are_linked_by_family_not_by_prose():
    """Requirement: the catalog must show provider and model together.

    Before this, `gpt-5.6-luna` and `openai/gpt-5.6-luna` were two unrelated
    entries and only a YAML comment said they were the same weights at
    different prices. Now the link is data, and cost is resolved per route.
    """
    ai_config = get_ai_config()

    llm_routes = ai_config.get_family_routes("gpt-5.6-luna")
    assert {r["model"] for r in llm_routes} == {
        "gpt-5.6-luna",
        "openai/gpt-5.6-luna",
    }, llm_routes
    assert {r["cost_model"] for r in llm_routes} == {"subscription", "metered"}
    assert {r["provider"] for r in llm_routes} == {"codex_cli", "openrouter"}

    # The embedding side of the same modelling question.
    embed_routes = ai_config.get_family_routes("bge-m3")
    assert {r["model"] for r in embed_routes} == {"bge-m3", "baai/bge-m3"}
    assert {r["cost_model"] for r in embed_routes} == {"free_local", "metered"}


def test_every_catalog_route_states_its_provider_and_its_cost():
    ai_config = get_ai_config()
    for model in _catalog_models():
        route = ai_config.describe_model(model)
        assert route["provider"], f"{model} names no provider"
        assert route["cost_model"] != "unknown", (
            f"{model}'s provider {route['provider']} declares no cost_model, so "
            "no listing can tell the user whether it costs money"
        )
        # The key is the wire id: this is what gets written to LLM_MODEL /
        # EMBEDDING_MODEL and handed to the adapter unchanged.
        assert route["provider_model_id"] == model
        assert model in ai_config.get_provider_models(route["provider"])


def test_every_model_menu_row_carries_its_provider_and_cost():
    """A human-facing list may never show a bare model name.

    `configure-provider` and `init` render every option through
    `format_model_description`, so this covers the surfaces the user actually
    reads.
    """
    ai_config = get_ai_config()
    for model in _catalog_models():
        rendered = ai_config.format_model_description(model)
        route = ai_config.describe_model(model)
        assert route["provider"] in rendered, rendered
        assert route["cost_model"] in rendered, rendered


def test_configure_provider_menu_rows_are_rendered_with_the_route():
    """The renderer is actually wired into the CLI, not merely available."""
    from scripts import cli

    source = inspect.getsource(cli.cmd_configure_provider)
    assert source.count("ai_config.format_model_description(m)") == 2, (
        "configure-provider stopped rendering model menus through the "
        "route-carrying formatter; a bare model name tells the user nothing "
        "about which provider serves it or what it costs"
    )


def test_llm_preference_order_contains_only_subscription_providers():
    """"Preferred" has one meaning: it spends a subscription you already have."""
    ai_config = get_ai_config()
    order = ai_config.config["routing"]["llm_preference_order"]
    assert order, "the preference order is empty"
    for provider in order:
        config = ai_config.get_provider_config(provider)
        assert config.get("cost_model") == "subscription", (
            f"{provider} is in llm_preference_order but is "
            f"{config.get('cost_model')!r}, so 'preferred' would mean "
            "'reached first' rather than 'already paid for'"
        )
        assert ai_config.is_provider_supported(provider, "llm")


def test_gemini_cli_is_a_preferred_subscription_llm_route():
    ai_config = get_ai_config()
    assert "gemini_cli" in ai_config.config["routing"]["llm_preference_order"]
    assert "gemini_cli" not in ai_config.config["routing"]["explicit_opt_in_required"]
    assert ai_config.is_provider_supported("gemini_cli", "llm")
    assert not ai_config.is_provider_supported("gemini_cli", "embeddings")
    assert ai_config.get_provider_models("gemini_cli", "llm"), (
        "gemini_cli has no models, so configure-provider would offer an "
        "empty menu after probing it as available"
    )


# Widths pinned INDEPENDENTLY of the catalog, so the check below is not a
# tautology (an earlier draft compared the catalog to itself and survived a
# mutation that changed a width -- it could not fail). Provenance: the six
# openrouter rows were measured by live calls on 2026-08-05 (recorded in the
# docs/MODULE_INDEX.md ledger row); the ollama and direct-cloud rows are the
# published widths of those models. Changing a catalog width now requires
# changing this table too, which is the point.
VERIFIED_EMBEDDING_DIMENSIONS = {
    "nomic-embed-text": 768,
    "embeddinggemma": 768,
    "bge-m3": 1024,
    "qwen3-embedding": 1024,
    "qwen/qwen3-embedding-8b": 4096,
    "qwen/qwen3-embedding-4b": 2560,
    "baai/bge-m3": 1024,
    "google/gemini-embedding-001": 3072,
    "openai/text-embedding-3-small": 1536,
    "openai/text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-004": 768,
}


def test_the_pinned_width_table_covers_every_registered_embedding_model():
    """Non-vacuity: a new embedding route must be width-verified, not skipped."""
    registered = {
        name
        for name, config in get_ai_config().config["models"].items()
        if config.get("type") == "embedding"
    }
    assert registered == set(VERIFIED_EMBEDDING_DIMENSIONS), (
        "VERIFIED_EMBEDDING_DIMENSIONS is out of step with the catalog; "
        f"symmetric diff: {registered ^ set(VERIFIED_EMBEDDING_DIMENSIONS)}"
    )


@pytest.mark.parametrize("model", sorted(VERIFIED_EMBEDDING_DIMENSIONS))
def test_dimension_resolution_is_per_route_after_the_family_restructure(
    monkeypatch, model
):
    """Requirement 3 regression guard, run against every embedding route.

    Every registered embedding model must still resolve, through the real
    `LLMClient` path, to the width it actually emits -- compared against the
    pinned table above rather than against the catalog entry the resolver
    itself reads, so a wrong catalog width fails here instead of agreeing
    with itself.

    `test_two_routes_to_one_family_keep_their_own_vector_widths` covers the
    separate hazard this one cannot see: resolving a width from the FAMILY
    rather than the route.
    """
    ai_config = get_ai_config()
    provider = ai_config.get_model_config(model)["provider"]

    _clear_provider_env(monkeypatch)
    _block_ollama_probe(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", provider)
    monkeypatch.setenv("EMBEDDING_MODEL", model)

    client = LLMClient()

    assert client.embedding_config_error is None, client.embedding_config_error
    assert client.embedding_config.provider == provider
    assert client.embedding_config.dimensions == VERIFIED_EMBEDDING_DIMENSIONS[model], (
        f"{model} resolves to {client.embedding_config.dimensions}-d but emits "
        f"{VERIFIED_EMBEDDING_DIMENSIONS[model]}-d; ai-engine would refuse "
        "every write to the table it announces"
    )


def test_two_routes_to_one_family_keep_their_own_vector_widths(tmp_path):
    """The hazard `model_family` introduces, gated before it can be built.

    Grouping routes by family invites hoisting shared attributes onto the
    family. `dimensions` must never be one of them: two routes to the same
    family can genuinely emit different widths (a local 1024-d pull and a
    hosted 4096-d sibling), and an announced width that differs from the
    emitted one makes ai-engine refuse every write to its LanceDB table --
    the exact failure `_resolve_embedding_dimensions` exists to prevent.

    The current catalog happens not to contain a same-family/different-width
    pair, so a family-level lookup would pass every other test in this file.
    This one constructs the case.
    """
    from config.config_loader import AIModelConfig

    catalog = tmp_path / "ai_models.yaml"
    catalog.write_text(
        yaml.safe_dump(
            {
                "providers": {
                    "ollama": {
                        "type": "local",
                        "cost_model": "free_local",
                        "supports": {"llm": True, "embeddings": True},
                        "default_models": {"llm": None, "embedding": "twin-small"},
                    },
                    "openrouter": {
                        "type": "cloud",
                        "cost_model": "metered",
                        "supports": {"llm": True, "embeddings": True},
                        "default_models": {"llm": None, "embedding": "vendor/twin-big"},
                    },
                },
                "models": {
                    "twin-small": {
                        "provider": "ollama",
                        "model_family": "twin",
                        "type": "embedding",
                        "dimensions": 1024,
                    },
                    "vendor/twin-big": {
                        "provider": "openrouter",
                        "model_family": "twin",
                        "type": "embedding",
                        "dimensions": 4096,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    config = AIModelConfig(config_file=str(catalog))

    assert {r["model"] for r in config.get_family_routes("twin")} == {
        "twin-small",
        "vendor/twin-big",
    }
    assert config.describe_model("twin-small")["dimensions"] == 1024
    assert config.describe_model("vendor/twin-big")["dimensions"] == 4096, (
        "two routes to one family collapsed onto a single vector width; "
        "every write to the other route's table would be refused"
    )
