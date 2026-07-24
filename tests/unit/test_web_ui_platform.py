"""Tests for Streamlit platform navigation and API client shapes."""

import importlib
import sys
import types
from pathlib import Path

import requests


class SessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


class _StreamlitContext:
    def __init__(self, streamlit):
        self.streamlit = streamlit

    def __enter__(self):
        return self.streamlit

    def __exit__(self, exc_type, exc, traceback):
        return False


class FakeStreamlit:
    """Small Streamlit stand-in for render-smoke tests."""

    def __init__(self, session_state=None):
        self.session_state = session_state or SessionState()
        self.rendered = []

    def _record(self, *values):
        text = " ".join(str(value) for value in values if value is not None)
        if text:
            self.rendered.append(text)

    def header(self, text):
        self._record(text)

    def subheader(self, text):
        self._record(text)

    def success(self, text):
        self._record(text)

    def info(self, text):
        self._record(text)

    def warning(self, text):
        self._record(text)

    def write(self, text):
        self._record(text)

    def caption(self, text):
        self._record(text)

    def markdown(self, text):
        self._record(text)

    def code(self, text, language=None):
        self._record(text, language)

    def metric(self, label, value):
        self._record(label, value)

    def table(self, rows):
        self._record(rows)

    def divider(self):
        return None

    def columns(self, spec):
        count = spec if isinstance(spec, int) else len(spec)
        return [_StreamlitContext(self) for _ in range(count)]

    def tabs(self, labels):
        for label in labels:
            self._record(label)
        return [_StreamlitContext(self) for _ in labels]

    def form(self, key):
        self._record(key)
        return _StreamlitContext(self)

    def expander(self, label, expanded=False):
        self._record(label)
        return _StreamlitContext(self)

    def text_input(self, label, value="", **kwargs):
        self._record(label)
        key = kwargs.get("key")
        if key and key in self.session_state:
            return self.session_state[key]
        return value

    def number_input(self, label, value=0, **kwargs):
        self._record(label)
        key = kwargs.get("key")
        if key and key in self.session_state:
            return self.session_state[key]
        return value

    def selectbox(self, label, options, index=0, **kwargs):
        self._record(label)
        key = kwargs.get("key")
        if key and key in self.session_state:
            return self.session_state[key]
        return options[index]

    def multiselect(self, label, options, default=None, **kwargs):
        self._record(label)
        key = kwargs.get("key")
        if key and key in self.session_state:
            return self.session_state[key]
        return default or []

    def button(self, label, **kwargs):
        self._record(label)
        return False

    def form_submit_button(self, label, **kwargs):
        self._record(label)
        return False

    def rerun(self):
        self._record("rerun")


def test_platform_page_is_registered_in_navigation():
    root = Path(__file__).resolve().parents[2]
    app_text = (root / "services" / "web-ui" / "app.py").read_text(encoding="utf-8")
    pages_text = (
        root / "services" / "web-ui" / "src" / "pages" / "__init__.py"
    ).read_text(encoding="utf-8")

    assert '"Platform"' in app_text
    assert "render_platform_page" in app_text
    assert "render_platform_page" in pages_text


def test_platform_page_has_no_sticker_text():
    root = Path(__file__).resolve().parents[2]
    paths = [
        root / "services" / "web-ui" / "app.py",
        root / "services" / "web-ui" / "src" / "pages" / "platform.py",
    ]

    offenders = []
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            1,
        ):
            stickers = sorted({char for char in line if _is_sticker(char)})
            if stickers:
                offenders.append(
                    f"{path.relative_to(root)}:{line_number}: {''.join(stickers)}"
                )

    assert not offenders, "Sticker characters found:\n" + "\n".join(offenders)


def test_platform_response_helpers_match_backend_shapes(monkeypatch):
    monkeypatch.setitem(sys.modules, "streamlit", types.SimpleNamespace())
    platform = importlib.import_module("services.web_ui.src.pages.platform")

    assert (
        platform._extract_token({"session_token": "tbs_local_session"})
        == "tbs_local_session"
    )
    assert platform._extract_profile({"user": {"role": "b2b"}}) == {"role": "b2b"}
    assert platform._is_b2b_user({"role": "b2b"}) is True
    assert platform._dashboard_metrics(
        {
            "counters": {
                "api_requests_total": 3,
                "api_tokens_active": 2,
                "companies_available": 9,
                "issues_created": 1,
            }
        }
    ) == {
        "api_calls": 3,
        "active_tokens": 2,
        "companies": 9,
        "issues": 1,
    }


def test_platform_auth_clears_account_scoped_state(monkeypatch):
    session_state = SessionState(
        {
            "platform_created_token": {"token": "tbo_previous_raw_token"},
            "platform_last_raw_api_token": "tbo_previous_raw_token",
            "platform_first_call_api_token": "tbo_previous_raw_token",
            "platform_dashboard": {"api_calls": 9},
            "platform_company_results": {"companies": []},
        }
    )
    monkeypatch.setitem(
        sys.modules,
        "streamlit",
        types.SimpleNamespace(session_state=session_state),
    )
    sys.modules.pop("services.web_ui.src.pages.platform", None)
    platform = importlib.import_module("services.web_ui.src.pages.platform")

    stored = platform._store_auth_response(
        {"access_token": "new-session", "user": {"email": "new@example.com"}}
    )

    assert stored is True
    assert session_state["platform_token"] == "new-session"
    assert session_state["platform_profile"] == {"email": "new@example.com"}
    assert "platform_created_token" not in session_state
    assert "platform_last_raw_api_token" not in session_state
    assert "platform_first_call_api_token" not in session_state
    assert "platform_dashboard" not in session_state
    assert "platform_company_results" not in session_state


def test_platform_error_preserves_backend_detail(monkeypatch):
    fake_st = FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    sys.modules.pop("services.web_ui.src.pages.platform", None)
    platform = importlib.import_module("services.web_ui.src.pages.platform")

    class Response:
        status_code = 403
        text = ""

        def json(self):
            return {"detail": "Business account is required"}

    error = requests.HTTPError("Forbidden")
    error.response = Response()

    platform._platform_error("Token creation failed", error)

    assert (
        "Token creation failed: backend returned HTTP 403: "
        "Business account is required."
    ) in fake_st.rendered


def test_platform_authenticated_business_render_smoke(monkeypatch):
    session_state = SessionState(
        {
            "platform_token": "session-token",
            "platform_profile": {
                "email": "buyer@example.com",
                "role": "b2b",
                "account_type": "business",
            },
            "platform_dashboard": {
                "counters": {
                    "api_requests_total": 4,
                    "api_tokens_active": 1,
                    "companies_available": 2,
                    "issues_created": 1,
                }
            },
            "platform_first_call": {
                "description": "Search companies with a B2B token.",
                "curl": "curl -H 'Authorization: Bearer <api-token>'",
            },
            "platform_token_list": {
                "tokens": [
                    {
                        "id": "tok_1",
                        "name": "Production",
                        "prefix": "tbo_preview",
                        "status": "active",
                        "scopes": ["companies:read"],
                    }
                ]
            },
            "platform_company_results": {
                "companies": [
                    {
                        "id": "co_1",
                        "name": "Acme Corp",
                        "domain": "acme.example",
                    }
                ]
            },
        }
    )
    fake_st = FakeStreamlit(session_state)
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    sys.modules.pop("services.web_ui.src.pages.platform", None)
    platform = importlib.import_module("services.web_ui.src.pages.platform")

    platform.render_platform_page()

    rendered = "\n".join(fake_st.rendered)
    assert "Signed in as buyer@example.com" in rendered
    assert "Company Search" in rendered
    assert "Found 1 companies" in rendered
    assert "Acme Corp" in rendered
    assert "B2B API" in rendered
    assert "API Calls 4" in rendered
    assert "First API Call" in rendered
    assert "Production" in rendered
    assert "Maintainer Issues" in rendered
    assert "Maintainer access is required to view platform issues." in rendered
    assert "Log in to search companies." not in rendered


def test_platform_auth_client_request_shapes(monkeypatch):
    client, calls = _client_with_request_spy(monkeypatch)

    client.platform_signup(
        email="buyer@example.com",
        name="Buyer",
        account_type="business",
        company_name="Acme",
        **{"password": "not-a-real-secret"},
    )
    client.platform_login("buyer@example.com", "secret")

    assert calls == [
        {
            "method": "POST",
            "url": "http://backend.test/api/v1/platform/auth/signup",
            "timeout": 7.0,
            "json": {
                "email": "buyer@example.com",
                "password": "not-a-real-secret",
                "name": "Buyer",
                "account_type": "business",
                "company_name": "Acme",
            },
        },
        {
            "method": "POST",
            "url": "http://backend.test/api/v1/platform/auth/login",
            "timeout": 7.0,
            "json": {"email": "buyer@example.com", "password": "secret"},
        },
    ]


def test_platform_authenticated_client_request_shapes(monkeypatch):
    client, calls = _client_with_request_spy(monkeypatch)

    client.platform_me("jwt")
    client.search_companies("acme", limit=7, token="jwt")
    client.get_company("co_1", token="jwt")
    client.create_b2b_token("Production", scopes=["companies:read"], token="jwt")
    client.list_b2b_tokens(token="jwt")
    client.revoke_b2b_token("tok_1", token="jwt")
    client.get_b2b_first_call(token="jwt")
    client.search_companies_with_api_token("acme", api_token="api-token", limit=3)
    client.get_platform_dashboard(token="jwt")
    client.get_maintainer_issues(status="open", token="jwt")

    auth_headers = {"Authorization": "Bearer jwt"}
    assert calls == [
        {
            "method": "GET",
            "url": "http://backend.test/api/v1/platform/me",
            "timeout": 7.0,
            "headers": auth_headers,
        },
        {
            "method": "GET",
            "url": "http://backend.test/api/v1/platform/companies/search",
            "timeout": 7.0,
            "params": {"q": "acme", "limit": 7},
            "headers": auth_headers,
        },
        {
            "method": "GET",
            "url": "http://backend.test/api/v1/platform/companies/co_1",
            "timeout": 7.0,
            "headers": auth_headers,
        },
        {
            "method": "POST",
            "url": "http://backend.test/api/v1/platform/b2b/tokens",
            "timeout": 7.0,
            "json": {"name": "Production", "scopes": ["companies:read"]},
            "headers": auth_headers,
        },
        {
            "method": "GET",
            "url": "http://backend.test/api/v1/platform/b2b/tokens",
            "timeout": 7.0,
            "headers": auth_headers,
        },
        {
            "method": "DELETE",
            "url": "http://backend.test/api/v1/platform/b2b/tokens/tok_1",
            "timeout": 7.0,
            "headers": auth_headers,
        },
        {
            "method": "GET",
            "url": "http://backend.test/api/v1/platform/b2b/first-call",
            "timeout": 7.0,
            "headers": auth_headers,
        },
        {
            "method": "GET",
            "url": "http://backend.test/api/v1/platform/v1/companies/search",
            "timeout": 7.0,
            "params": {"query": "acme", "limit": 3},
            "headers": {"Authorization": "Bearer api-token"},
        },
        {
            "method": "GET",
            "url": "http://backend.test/api/v1/platform/dashboard",
            "timeout": 7.0,
            "headers": auth_headers,
        },
        {
            "method": "GET",
            "url": "http://backend.test/api/v1/platform/maintainer/issues",
            "timeout": 7.0,
            "params": {"status": "open"},
            "headers": auth_headers,
        },
    ]


def test_ai_client_request_shapes_include_shared_token(monkeypatch):
    client, calls = _client_with_request_spy(monkeypatch)
    monkeypatch.setenv("AI_ENGINE_API_TOKEN", "ai-token")
    client = importlib.import_module("services.web_ui.src.api.client").SyncAPIClient()

    client.chat("hello", session_id="sess_1")
    client.switch_provider(llm_provider="claude_code")

    assert calls == [
        {
            # WI0 B7: chat is proxied through Backend Core, not called on
            # ai_url directly, so no client-side AI Engine token header.
            "method": "POST",
            "url": "http://backend.test/api/v1/chat",
            "timeout": 7.0,
            "json": {"query": "hello", "session_id": "sess_1"},
        },
        {
            "method": "POST",
            "url": "http://ai-engine:8090/providers/switch",
            "timeout": 7.0,
            "json": {"llm_provider": "claude_code", "embedding_provider": None},
            "headers": {"Authorization": "Bearer ai-token"},
        },
    ]


def test_ai_runtime_config_update_shape_includes_models_and_api_keys(monkeypatch):
    client, calls = _client_with_request_spy(monkeypatch)
    monkeypatch.setenv("AI_ENGINE_API_TOKEN", "ai-token")
    client = importlib.import_module("services.web_ui.src.api.client").SyncAPIClient()

    client.update_ai_config(
        llm_provider="openrouter",
        llm_model="openrouter/auto",
        embedding_provider="ollama",
        embedding_model="nomic-embed-text",
        api_keys={"OPENROUTER_API_KEY": "local-openrouter-key"},
    )

    assert calls == [
        {
            "method": "POST",
            "url": "http://ai-engine:8090/config",
            "timeout": 7.0,
            "json": {
                "llm_provider": "openrouter",
                "llm_model": "openrouter/auto",
                "embedding_provider": "ollama",
                "embedding_model": "nomic-embed-text",
                "api_keys": {"OPENROUTER_API_KEY": "local-openrouter-key"},
            },
            "headers": {"Authorization": "Bearer ai-token"},
        }
    ]


def _client_with_request_spy(monkeypatch):
    module = importlib.import_module("services.web_ui.src.api.client")
    monkeypatch.setenv("BACKEND_URL", "http://backend.test")
    monkeypatch.setenv("AI_ENGINE_URL", "http://ai-engine:8090")
    monkeypatch.setenv("UI_API_TIMEOUT", "7")
    monkeypatch.delenv("AI_ENGINE_API_TOKEN", raising=False)

    calls = []

    class Response:
        content = b'{"ok": true}'

        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

    def fake_request(method, url, timeout, **kwargs):
        calls.append({"method": method, "url": url, "timeout": timeout, **kwargs})
        return Response()

    monkeypatch.setattr(module.requests, "request", fake_request)
    return module.SyncAPIClient(), calls


def _is_sticker(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x1F300 <= codepoint <= 0x1FAFF
        or 0x2600 <= codepoint <= 0x27BF
        or codepoint == 0xFE0F
    )
