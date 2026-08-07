"""Static checks for Streamlit UI copy."""

import importlib
import sys
import types
from pathlib import Path


def test_streamlit_ui_text_has_no_sticker_characters():
    """User-facing Streamlit copy should not include sticker-style emoji."""
    root = Path(__file__).resolve().parents[2]
    web_ui = root / "services" / "web-ui"
    paths = [web_ui / "app.py", *sorted((web_ui / "src").rglob("*.py"))]

    offenders = []
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            stickers = sorted({char for char in line if _is_sticker(char)})
            if stickers:
                offenders.append(
                    f"{path.relative_to(root)}:{line_number}: {''.join(stickers)}"
                )

    assert not offenders, "Sticker characters found in UI text:\n" + "\n".join(
        offenders
    )


def test_clustering_page_does_not_render_llm_labels_as_unsafe_html():
    root = Path(__file__).resolve().parents[2]
    clustering_text = (
        root / "services" / "web-ui" / "src" / "pages" / "clustering.py"
    ).read_text(encoding="utf-8")

    assert "unsafe_allow_html=True" not in clustering_text


def _is_sticker(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x1F300 <= codepoint <= 0x1FAFF
        or 0x2600 <= codepoint <= 0x27BF
        or codepoint == 0xFE0F
    )


def test_settings_provider_options_preserve_unknown_current_provider(monkeypatch):
    """Unknown current providers should not silently select the first CLI option."""
    monkeypatch.setitem(sys.modules, "streamlit", types.SimpleNamespace())
    settings = importlib.import_module("services.web_ui.src.pages.settings")

    options = settings._provider_options(
        available={"llm": {"ollama": {"available": True}}},
        fallback=["claude_code", "ollama"],
        current_provider="custom_provider",
        capability="llm",
    )

    assert options[0] == "custom_provider"
    assert settings._provider_index(options, "custom_provider") == 0


def test_settings_provider_options_show_unavailable_cli_providers(monkeypatch):
    monkeypatch.setitem(sys.modules, "streamlit", types.SimpleNamespace())
    settings = importlib.import_module("services.web_ui.src.pages.settings")

    options = settings._provider_options(
        available={
            "llm": {
                "claude_code": {"available": False},
                "codex_cli": {"available": False},
                "codex_acp": {"available": False},
                "ollama": {"available": True},
            }
        },
        fallback=["claude_code", "codex_cli", "codex_acp", "ollama"],
        current_provider="ollama",
        capability="llm",
    )

    assert options == ["claude_code", "codex_cli", "codex_acp", "ollama"]


def test_settings_provider_index_returns_none_when_nothing_active():
    """R1/R4: an unselected provider must render unselected, not fall through
    to index 0 and highlight the first catalog entry (the exact regression
    ec4cc0a introduced once GET /providers started sending provider=None
    instead of omitting the key)."""
    import services.web_ui.src.pages.settings as settings

    options = ["ollama", "openai", "claude_code"]

    assert settings._provider_index(options, "unknown") is None
    assert settings._provider_index(options, "") is None
    assert settings._provider_index(options, None) is None


def test_settings_model_index_handles_empty_options_without_crashing():
    """No active provider means no model list to look up; index must be
    None (Streamlit accepts that), never 0 against an empty options list."""
    import services.web_ui.src.pages.settings as settings

    assert settings._model_index([], "") is None
    assert settings._model_index(["llama3"], "") == 0
    assert settings._model_index(["llama3", "phi3"], "phi3") == 1


def _build_fake_streamlit():
    """Minimal Streamlit stand-in for driving render_settings_page() end to
    end, recording every selectbox call and every piece of rendered text."""
    rendered = []
    selectbox_calls = []

    class SessionState(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError:
                return None

        def __setattr__(self, key, value):
            self[key] = value

    class Context:
        def __enter__(self):
            return fake_st

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeStreamlit:
        session_state = SessionState({"current_session_id": None})

        def _record(self, *values):
            rendered.extend(str(value) for value in values if value is not None)

        def header(self, text):
            self._record(text)

        def subheader(self, text):
            self._record(text)

        def write(self, text):
            self._record(text)

        def info(self, text):
            self._record(text)

        def warning(self, text):
            self._record(text)

        def error(self, text):
            self._record(text)

        def success(self, text):
            self._record(text)

        def caption(self, text):
            self._record(text)

        def code(self, text):
            return None

        def divider(self):
            return None

        def columns(self, spec):
            count = spec if isinstance(spec, int) else len(spec)
            return [Context() for _ in range(count)]

        def expander(self, label, expanded=False):
            self._record(label)
            return Context()

        def button(self, label, **kwargs):
            self._record(label)
            return False

        def text_input(self, label, **kwargs):
            self._record(label)
            return ""

        def selectbox(
            self,
            label,
            options,
            index=None,
            key=None,
            format_func=None,
            placeholder=None,
            **kwargs,
        ):
            options = list(options)
            selectbox_calls.append(
                {
                    "label": label,
                    "options": options,
                    "index": index,
                    "key": key,
                    "placeholder": placeholder,
                }
            )
            if index is not None and options:
                return options[index]
            return None

        def download_button(self, *args, **kwargs):
            return False

        def metric(self, label, value):
            self._record(label, value)

        def rerun(self):
            return None

    fake_st = FakeStreamlit()
    return fake_st, rendered, selectbox_calls


def test_settings_page_shows_no_provider_selected_and_error_fix(monkeypatch):
    """End-to-end regression test for the settings-page bug: with no
    provider selected, GET /providers now sends provider=None (present, not
    omitted). The page must not pre-select the first catalog provider, must
    say plainly that nothing is selected, and must surface error.fix -- the
    actionable sentence telling the user to run
    ./scripts/cli.py configure-provider or set AI_PROVIDER."""
    fake_st, rendered, selectbox_calls = _build_fake_streamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)

    error = {
        "code": "provider_not_selected",
        "cause": "AI_PROVIDER is not set. This service does not pick a provider for you.",
        "fix": "Run ./scripts/cli.py configure-provider, or set AI_PROVIDER explicitly.",
    }

    class FakeAPI:
        def get_providers(self):
            return {
                "llm": {
                    "provider": None,
                    "model": None,
                    "capabilities": {},
                    "error": error,
                },
                "embeddings": {
                    "provider": None,
                    "model": None,
                    "dimensions": None,
                    "error": error,
                },
                "available": {
                    "llm": {"ollama": {"available": True}},
                    "embeddings": {"ollama": {"available": True}},
                },
                "models": {
                    "llm": {"ollama": ["llama3"]},
                    "embeddings": {"ollama": ["nomic-embed-text"]},
                },
            }

        def get_ai_engine_health(self):
            return {
                "status": "degraded",
                "providers": {
                    "llm": {"provider": None, "cost_model": None, "error": error},
                    "embedding": {
                        "provider": None,
                        "cost_model": None,
                        "error": error,
                    },
                },
            }

        def list_sessions(self):
            return []

    fake_st.session_state["api_client"] = FakeAPI()

    sys.modules.pop("services.web_ui.src.pages.settings", None)
    settings = importlib.import_module("services.web_ui.src.pages.settings")

    settings.render_settings_page()

    llm_select = next(c for c in selectbox_calls if c["key"] == "llm_provider_select")
    emb_select = next(c for c in selectbox_calls if c["key"] == "emb_provider_select")

    # The regression: index fell through to 0, highlighting whichever
    # provider happened to be first in the catalog, as though it were active.
    assert llm_select["index"] is None
    assert emb_select["index"] is None
    assert "ollama" not in llm_select["options"] or llm_select["index"] != 0

    joined = "\n".join(rendered)
    assert "None / None" not in joined
    assert "No LLM provider selected." in rendered
    assert "No embedding provider selected." in rendered
    assert error["fix"] in rendered


def test_settings_page_preselects_and_labels_active_provider(monkeypatch):
    """When a provider *is* active, the page must show it plainly, pre-select
    it in the dropdown, and (R4) surface its cost_model from GET /health."""
    fake_st, rendered, selectbox_calls = _build_fake_streamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)

    class FakeAPI:
        def get_providers(self):
            return {
                "llm": {
                    "provider": "claude_code",
                    "model": "sonnet",
                    "capabilities": {},
                    "error": None,
                },
                "embeddings": {
                    "provider": "ollama",
                    "model": "nomic-embed-text",
                    "dimensions": 768,
                    "error": None,
                },
                "available": {
                    "llm": {
                        "claude_code": {"available": True},
                        "ollama": {"available": True},
                    },
                    "embeddings": {"ollama": {"available": True}},
                },
                "models": {
                    "llm": {"claude_code": ["sonnet", "opus"]},
                    "embeddings": {"ollama": ["nomic-embed-text"]},
                },
            }

        def get_ai_engine_health(self):
            return {
                "status": "healthy",
                "providers": {
                    "llm": {
                        "provider": "claude_code",
                        "model": "sonnet",
                        "cost_model": "subscription",
                    },
                    "embedding": {
                        "provider": "ollama",
                        "model": "nomic-embed-text",
                        "cost_model": "free_local",
                    },
                },
            }

        def list_sessions(self):
            return []

    fake_st.session_state["api_client"] = FakeAPI()

    sys.modules.pop("services.web_ui.src.pages.settings", None)
    settings = importlib.import_module("services.web_ui.src.pages.settings")

    settings.render_settings_page()

    llm_select = next(c for c in selectbox_calls if c["key"] == "llm_provider_select")
    emb_select = next(c for c in selectbox_calls if c["key"] == "emb_provider_select")

    assert llm_select["options"][llm_select["index"]] == "claude_code"
    assert emb_select["options"][emb_select["index"]] == "ollama"

    joined = "\n".join(rendered)
    assert "Current: claude_code / sonnet" in joined
    assert "subscription" in joined
    assert "Current: ollama / nomic-embed-text" in joined
    assert "free_local" in joined


def test_api_client_start_scraping_sends_browser_mode(monkeypatch):
    from services.web_ui.src.api.client import SyncAPIClient

    calls = []

    def fake_request(method, url, timeout=None, **kwargs):
        calls.append({"method": method, "url": url, "kwargs": kwargs})

        class Response:
            content = b"{}"

            def raise_for_status(self):
                return None

            def json(self):
                return {"status": "started"}

        return Response()

    monkeypatch.setattr("services.web_ui.src.api.client.requests.request", fake_request)

    client = SyncAPIClient()
    client.start_scraping("session-1", use_browser=True)

    assert calls[0]["kwargs"]["json"] == {
        "session_id": "session-1",
        "use_browser": True,
    }


def test_api_client_chat_routes_through_backend_not_ai_engine(monkeypatch):
    """WI0 B7: chat must go through Backend Core, never straight to ai-engine."""
    from services.web_ui.src.api.client import SyncAPIClient

    calls = []

    def fake_request(method, url, timeout=None, **kwargs):
        calls.append({"method": method, "url": url, "kwargs": kwargs})

        class Response:
            content = b'{"answer": "ok"}'

            def raise_for_status(self):
                return None

            def json(self):
                return {"answer": "ok"}

        return Response()

    monkeypatch.setattr("services.web_ui.src.api.client.requests.request", fake_request)

    client = SyncAPIClient()
    result = client.chat("what changed?", session_id="session-1")

    assert calls[0]["url"] == f"{client.backend_url}/chat"
    assert client.ai_url not in calls[0]["url"]
    assert result == {"answer": "ok"}


def test_api_client_search_routes_through_backend_not_ai_engine(monkeypatch):
    """search() must go through Backend Core's /search, which merges SQLite
    FTS keyword hits with ai-engine semantic hits; calling ai_url directly
    silently dropped the keyword leg (same class of bug as WI0 B7's chat)."""
    from services.web_ui.src.api.client import SyncAPIClient

    calls = []

    def fake_request(method, url, timeout=None, **kwargs):
        calls.append({"method": method, "url": url, "kwargs": kwargs})

        class Response:
            content = b'{"results": [], "count": 0, "mode": "hybrid"}'

            def raise_for_status(self):
                return None

            def json(self):
                return {"results": [], "count": 0, "mode": "hybrid"}

        return Response()

    monkeypatch.setattr("services.web_ui.src.api.client.requests.request", fake_request)
    monkeypatch.setenv("BACKEND_AGENT_API_TOKEN", "agent-token")

    client = SyncAPIClient()
    result = client.search("cats", session_id="session-1")

    assert calls[0]["url"] == f"{client.backend_url}/search"
    assert client.ai_url not in calls[0]["url"]
    assert calls[0]["kwargs"]["json"] == {"query": "cats", "session_id": "session-1"}
    assert calls[0]["kwargs"]["headers"] == {"Authorization": "Bearer agent-token"}
    assert result == {"results": [], "count": 0, "mode": "hybrid"}


def test_api_client_scrape_status_reports_unavailable_on_backend_http_error(
    monkeypatch,
):
    """Finding 28: never fabricate not_started/completed on a failed status call."""
    import requests

    from services.web_ui.src.api.client import SyncAPIClient

    def fake_request(method, url, timeout=None, **kwargs):
        response = requests.Response()
        response.status_code = 404
        raise requests.HTTPError(response=response)

    monkeypatch.setattr("services.web_ui.src.api.client.requests.request", fake_request)

    client = SyncAPIClient()
    status = client.get_scrape_status("session-1")

    assert status["status"] == "unknown"
    assert "status unavailable" in status["detail"]
    assert "total" not in status
    assert "completed" not in status


def test_api_client_scrape_status_reports_unavailable_when_backend_unreachable(
    monkeypatch,
):
    """Connection failures must not propagate uncaught nor be fabricated."""
    import requests

    from services.web_ui.src.api.client import SyncAPIClient

    def fake_request(method, url, timeout=None, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr("services.web_ui.src.api.client.requests.request", fake_request)

    client = SyncAPIClient()
    status = client.get_scrape_status("session-1")

    assert status["status"] == "unknown"
    assert "backend unreachable" in status["detail"]


def test_api_client_check_health_passes_through_status_strings(monkeypatch):
    """Finding 33: health aggregation must not collapse to booleans only."""
    import requests

    from services.web_ui.src.api.client import SyncAPIClient

    def fake_request(method, url, timeout=None, headers=None, **kwargs):
        class Response:
            def __init__(self, status_code, payload):
                self.status_code = status_code
                self._payload = payload
                self.content = b"1"

            def json(self):
                return self._payload

        if "backend-core" in url:
            return Response(200, {"status": "healthy"})
        if "ai-engine" in url:
            return Response(200, {"status": "degraded", "runtime": {"ready": False}})
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr("services.web_ui.src.api.client.requests.request", fake_request)

    client = SyncAPIClient()
    health = client.check_health()

    assert health["backend"] is True
    assert health["backend_status"] == "healthy"
    assert health["ai_engine"] is False
    assert health["ai_engine_status"] == "degraded"
    assert health["browser_engine"] is False
    assert health["browser_engine_status"] == "unreachable"


def test_api_client_get_ai_engine_health_returns_providers_block(monkeypatch):
    """Settings page reads cost_model off this payload's `providers` block
    (R4); it must reach the caller as-is, and an unreachable ai-engine must
    degrade rather than raise so the rest of the settings page still renders."""
    import requests

    from services.web_ui.src.api.client import SyncAPIClient

    def fake_request(method, url, timeout=None, headers=None, **kwargs):
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr("services.web_ui.src.api.client.requests.request", fake_request)

    client = SyncAPIClient()
    health = client.get_ai_engine_health()

    assert health["status"] == "unreachable"


def test_scraping_page_renders_downstream_error_details_for_callbacks(monkeypatch):
    rendered = []

    class SessionState(dict):
        def __getattr__(self, key):
            return self[key]

        def __setattr__(self, key, value):
            self[key] = value

    class Context:
        def __enter__(self):
            return fake_st

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeAPI:
        def get_scrape_status(self, session_id):
            return {
                "session_id": session_id,
                "status": "completed_with_downstream_errors",
                "total": 1,
                "completed": 1,
                "success": 1,
                "failed": 0,
                "auth_required": 0,
                "backend_callback_failed": 1,
                "downstream_errors": [
                    {
                        "source": "backend_callback",
                        "message": "Session not found",
                        "url": "https://example.com/",
                    }
                ],
                "error": "1 downstream operation(s) failed",
            }

        def get_pending_auth(self):
            return {"pending": []}

    class FakeStreamlit:
        session_state = SessionState(
            {
                "current_session_id": "session-1",
                "api_client": FakeAPI(),
            }
        )

        def _record(self, *values):
            rendered.extend(str(value) for value in values if value is not None)

        def header(self, text):
            self._record(text)

        def subheader(self, text):
            self._record(text)

        def warning(self, text):
            self._record(text)

        def info(self, text):
            self._record(text)

        def success(self, text):
            self._record(text)

        def error(self, text):
            self._record(text)

        def caption(self, text):
            self._record(text)

        def metric(self, label, value):
            self._record(label, value)

        def progress(self, value, text=None):
            self._record(text)

        def divider(self):
            return None

        def columns(self, spec):
            count = spec if isinstance(spec, int) else len(spec)
            return [Context() for _ in range(count)]

        def button(self, label, **kwargs):
            self._record(label)
            return False

        def checkbox(self, label, value=False, **kwargs):
            self._record(label)
            return False

        def expander(self, label, expanded=False):
            self._record(label)
            return Context()

        def write(self, text):
            self._record(text)

        def text_input(self, label, **kwargs):
            self._record(label)
            return ""

        def text_area(self, label, **kwargs):
            self._record(label)
            return ""

    fake_st = FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    sys.modules.pop("services.web_ui.src.pages.scraping", None)
    scraping = importlib.import_module("services.web_ui.src.pages.scraping")

    scraping.render_scraping_page()

    assert any("backend_callback: Session not found" in item for item in rendered)
