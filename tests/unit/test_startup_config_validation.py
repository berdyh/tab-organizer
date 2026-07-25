"""Fail-fast config validation and provider-usability regressions.

Finding 27: `validate_config()` (config/config_loader.py) exists but was never
called — wire it into each service's FastAPI lifespan so a malformed
ai_models.yaml refuses to start instead of failing on first use.

WI0 B1: a selected-but-unusable provider (e.g. openrouter with no API key)
must not report healthy-and-silent. Startup logs it structurally; the service
still starts (degraded-visible beats dead) and `/health` reports the reason.
"""

import io
import json
import logging
import sys
import types

import pytest


def _install_playwright_stub():
    """Allow importing browser-engine code without the browser binaries."""
    playwright_module = types.ModuleType("playwright")
    async_api = types.ModuleType("playwright.async_api")
    async_api.Browser = object
    async_api.Page = object
    async_api.TimeoutError = TimeoutError
    async_api.async_playwright = lambda: None
    playwright_module.async_api = async_api
    sys.modules.setdefault("playwright", playwright_module)
    sys.modules.setdefault("playwright.async_api", async_api)


try:
    import playwright.async_api  # noqa: F401
except ModuleNotFoundError:
    _install_playwright_stub()

from services.ai_engine.app import main as ai_main  # noqa: E402
from services.backend_core.app import main as backend_main  # noqa: E402
from services.browser_engine.app import main as browser_main  # noqa: E402


class _FakeAIConfig:
    """Stand-in for AIModelConfig with a scriptable validate_config() result."""

    def __init__(self, errors):
        self._errors = errors

    def validate_config(self):
        return self._errors


def _attach_json_capture(service: str):
    """Capture emitted JSON lines from a service logger for assertions.

    `configure_logging()` sets `propagate = False` on each service logger, so
    pytest's `caplog` (a root-logger handler) never sees these records; attach
    directly to the service logger instead (mirrors test_observability.py).
    """
    from services.observability import _JsonFormatter

    logger = logging.getLogger(f"taborganizer.{service}")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(_JsonFormatter())
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return stream, handler


def _force_service_name(monkeypatch, service: str):
    """Pin observability's process-global active-service name for this test.

    `configure_logging(service)` runs once per module at *import* time and
    sets a shared global; whichever of ai-engine/backend-core/browser-engine
    was imported last in this test process "wins" for `log_event()`'s target
    logger. That's fine in production (one service per process) but makes
    log-content assertions import-order-dependent here — pin it explicitly.
    """
    from services import observability as obs

    monkeypatch.setattr(obs, "_service_name", service)


def _read_events(stream: io.StringIO):
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "module",
    [ai_main, backend_main, browser_main],
    ids=["ai-engine", "backend-core", "browser-engine"],
)
async def test_lifespan_refuses_to_start_on_invalid_config(monkeypatch, module):
    monkeypatch.setattr(
        module,
        "get_ai_config",
        lambda: _FakeAIConfig(["Provider ollama missing 'supports' field"]),
    )

    with pytest.raises(RuntimeError, match="AI model configuration is invalid"):
        async with module.lifespan(module.app):
            pass  # pragma: no cover - should never reach the yield


@pytest.mark.smoke
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "module",
    [ai_main, backend_main, browser_main],
    ids=["ai-engine", "backend-core", "browser-engine"],
)
async def test_lifespan_starts_cleanly_on_valid_config(module):
    """The real, shipped ai_models.yaml must pass validate_config() as-is."""
    entered = False
    async with module.lifespan(module.app):
        entered = True
    assert entered


@pytest.mark.asyncio
async def test_ai_engine_lifespan_logs_but_does_not_crash_on_unusable_provider(
    monkeypatch,
):
    """WI0 B1: an unusable selected provider degrades, it never takes the
    service down — the whole point is staying diagnosable via /health.
    """
    monkeypatch.setattr(
        ai_main.llm_client,
        "get_runtime_health",
        lambda: {
            "ready": False,
            "llm": {
                "available": False,
                "reason": "OPENROUTER_API_KEY is not configured",
            },
            "embeddings": {
                "available": False,
                "reason": "OPENROUTER_API_KEY is not configured",
            },
        },
    )

    _force_service_name(monkeypatch, "ai-engine")
    stream, handler = _attach_json_capture("ai-engine")
    try:
        entered = False
        async with ai_main.lifespan(ai_main.app):
            entered = True  # never crashes/raises even though the provider is unusable
    finally:
        logging.getLogger("taborganizer.ai-engine").removeHandler(handler)

    assert entered
    events = _read_events(stream)
    assert any(event["event"] == "provider.unusable_at_startup" for event in events)
    unusable = next(e for e in events if e["event"] == "provider.unusable_at_startup")
    assert unusable["level"] == "error"
    assert unusable["llm_reason"] == "OPENROUTER_API_KEY is not configured"


def test_ai_engine_health_endpoint_reports_degraded_reason(monkeypatch):
    """B1: /health must expose *why* the provider is unusable, not just a flag."""
    from fastapi.testclient import TestClient

    # Isolate from real LanceDB I/O; only the provider-runtime branch matters here.
    monkeypatch.setattr(type(ai_main.chatbot), "table", property(lambda self: object()))
    monkeypatch.setattr(
        ai_main.llm_client,
        "get_runtime_health",
        lambda: {
            "ready": False,
            "llm": {
                "available": False,
                "reason": "OPENROUTER_API_KEY is not configured",
            },
            "embeddings": {
                "available": False,
                "reason": "OPENROUTER_API_KEY is not configured",
            },
        },
    )

    client = TestClient(ai_main.app)
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["runtime"]["llm"]["reason"] == "OPENROUTER_API_KEY is not configured"
