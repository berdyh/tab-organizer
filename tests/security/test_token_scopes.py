"""SEC-16..23: token scopes and fail-closed auth.

Managed mode sets a distinct random token per scope, so cross-acceptance is
genuinely observable: with every primary token configured, each door is opened
by exactly one principal and refused (401) for all others.

HISTORY (SECSUITE 1.3.0): browser-engine used to ACCEPT a cross-scope
fallback (BROWSER_ENGINE_API_TOKEN -> BACKEND_CALLBACK_TOKEN ->
AI_ENGINE_API_TOKEN), previously described here as an intentional documented
tradeoff. That fallback has been REMOVED. It was not merely a widened blast
radius: `scripts/cli.py` minted ONE value for all four scopes in every stock
deployment, so BACKEND_AGENT_API_TOKEN *was* the value browser-engine
accepted, and an agent principal opened the entire credential and CDP control
plane. The CLI now mints four independent tokens and browser-engine accepts
only its own scope (fail-closed 401 when unset). The `browser_auth_pending`
door below is therefore now load-bearing in the deployed configuration, not
just under this harness's synthetic per-scope tokens. Do not reintroduce a
cross-scope accept fallback.
"""

import os

import pytest

from tests.security.conftest import TOKEN_ENVS, WRONG_TOKEN

pytestmark = [pytest.mark.security]


def _principals() -> dict:
    return {
        "none": None,
        "wrong": WRONG_TOKEN,
        "ai": os.environ.get(TOKEN_ENVS["ai"]),
        "callback": os.environ.get(TOKEN_ENVS["callback"]),
        "agent": os.environ.get(TOKEN_ENVS["agent"]),
        "browser": os.environ.get(TOKEN_ENVS["browser"]),
    }


# door -> (service fixture, method, path, body, {principals that must be allowed})
DOORS = {
    "ai_embed": ("ai", "POST", "/embed", {"texts": ["x"]}, {"ai"}),
    # Only "browser" is allowed, and that is now the real runtime contract:
    # the callback/AI accept-fallback was removed (see module docstring), so
    # every other principal is refused here in deployment too, not only under
    # this harness's per-scope tokens.
    "browser_auth_pending": ("browser", "GET", "/auth/pending", None, {"browser"}),
    "backend_tabs_open": (
        "backend",
        "POST",
        "/api/v1/tabs/open",
        {"urls": ["http://example.com/"]},
        {"agent"},
    ),
    "backend_callback": (
        "backend",
        "POST",
        "/api/v1/callback/scrape-complete",
        {
            "session_id": "x",
            "url": "http://example.com/",
            "status": "failed",
            "content": None,
            "metadata": {},
        },
        {"callback"},
    ),
    "backend_ingest_v1": (
        "backend",
        "POST",
        "/api/v1/ingest/v1",
        {
            "capture_id": "sec-probe",
            "attempt": 1,
            "session_id": "x",
            "url": "http://example.com/",
            "status": "failed",
            "content": None,
            "metadata": {},
            "auth_used": False,
            "fetched_at": "2026-07-24T00:00:00",
        },
        {"callback"},
    ),
}


@pytest.mark.parametrize("door", list(DOORS))
@pytest.mark.parametrize("principal", list(_principals()))
def test_token_scope_matrix(request, door, principal):
    """SEC-16..20: exact 401 on deny, non-401 on allow, across the matrix."""
    fixture_name, method, path, body, allowed = DOORS[door]
    client = request.getfixturevalue(fixture_name)
    token = _principals()[principal]
    response = client.request(method, path, token=token, json=body)

    if principal in allowed:
        assert response.status_code != 401, (
            f"{door} refused its own principal {principal}"
        )
    else:
        assert response.status_code == 401, (
            f"{door} accepted foreign principal {principal} "
            f"(status {response.status_code})"
        )


# (method, path). GET /providers was added after a review found it answering
# 200 with no token while every sibling 401'd: it discloses `api_key_configured`
# per provider, the whole model catalog, and the structured selection errors.
# CLAUDE.md already claimed "ai-engine endpoints (except /health) require
# AI_ENGINE_API_TOKEN", so this probe freezes an invariant the docs asserted and
# the code did not honour, rather than adding a new one.
AI_PROTECTED = [
    ("POST", "/embed"),
    ("POST", "/index"),
    ("POST", "/search"),
    ("POST", "/chat"),
    ("POST", "/cluster"),
    ("POST", "/providers/switch"),
    ("GET", "/providers"),
]


@pytest.mark.parametrize("method, path", AI_PROTECTED)
@pytest.mark.parametrize("token", [None, WRONG_TOKEN])
def test_sec16_ai_endpoints_require_token(ai, method, path, token):
    body = {"texts": ["x"]} if path == "/embed" else {"session_id": "s", "query": "q"}
    response = ai.request(method, path, token=token, json=body if method != "GET" else None)
    assert response.status_code == 401


BROWSER_PROTECTED = [
    ("POST", "/scrape", {"session_id": "s", "urls": []}),
    ("POST", "/scrape/single", {"url": "http://example.com/"}),
    ("GET", "/scrape/status/s", None),
    ("POST", "/tabs/import", {"cdp_url": "http://127.0.0.1:9222"}),
    ("POST", "/tabs/open", {"urls": ["http://example.com/"]}),
    ("GET", "/auth/pending", None),
    ("POST", "/auth/expire", None),
]


@pytest.mark.parametrize("method,path,body", BROWSER_PROTECTED)
@pytest.mark.parametrize("token", [None, WRONG_TOKEN])
def test_sec17_browser_endpoints_require_token(browser, method, path, body, token):
    response = browser.request(method, path, token=token, json=body)
    assert response.status_code == 401


BACKEND_AGENT_PROTECTED = [
    ("POST", "/api/v1/tabs/import", {"cdp_url": "http://127.0.0.1:9222"}),
    ("GET", "/api/v1/tabs/import/job-1", None),
    ("POST", "/api/v1/tabs/open", {"urls": ["http://example.com/"]}),
    ("POST", "/api/v1/search", {"query": "q"}),
]


@pytest.mark.parametrize("method,path,body", BACKEND_AGENT_PROTECTED)
@pytest.mark.parametrize("token", [None, WRONG_TOKEN, "callback"])
def test_sec18_backend_agent_scope(request, backend, method, path, body, token):
    resolved = os.environ.get(TOKEN_ENVS["callback"]) if token == "callback" else token
    response = backend.request(method, path, token=resolved, json=body)
    assert response.status_code == 401


def test_sec19_backend_callback_scope(backend):
    body = {
        "session_id": "x",
        "url": "http://example.com/",
        "status": "failed",
        "content": None,
        "metadata": {},
    }
    for token in (None, WRONG_TOKEN, os.environ.get(TOKEN_ENVS["agent"])):
        response = backend.post(
            "/api/v1/callback/scrape-complete", token=token, json=body
        )
        assert response.status_code == 401


def test_sec20_cross_scope_must_nots(request):
    """The agent token opens nothing outside the backend agent endpoints; the
    callback token is refused by ai-engine and backend agent endpoints."""
    ai = request.getfixturevalue("ai")
    browser = request.getfixturevalue("browser")
    backend = request.getfixturevalue("backend")
    agent = os.environ.get(TOKEN_ENVS["agent"])
    callback = os.environ.get(TOKEN_ENVS["callback"])

    assert ai.post("/embed", token=agent, json={"texts": ["x"]}).status_code == 401
    assert ai.post(
        "/chat", token=agent, json={"query": "q"}
    ).status_code == 401
    assert browser.post(
        "/scrape/single", token=agent, json={"url": "http://example.com/"}
    ).status_code == 401
    assert browser.get("/auth/pending", token=agent).status_code == 401
    assert browser.post(
        "/detect-auth", token=agent, params={"url": "http://example.com/"}
    ).status_code == 401
    assert backend.post(
        "/api/v1/callback/scrape-complete",
        token=agent,
        json={"session_id": "x", "url": "http://example.com/", "status": "failed",
              "content": None, "metadata": {}},
    ).status_code == 401

    assert ai.post("/embed", token=callback, json={"texts": ["x"]}).status_code == 401
    assert backend.post(
        "/api/v1/search", token=callback, json={"query": "q"}
    ).status_code == 401


@pytest.mark.sec_managed
@pytest.mark.parametrize(
    "service,method,path,body,unset",
    [
        ("ai", "POST", "/embed", {"texts": ["x"]}, ["ai"]),
        (
            "browser",
            "GET",
            "/auth/pending",
            None,
            ["browser", "callback", "ai"],
        ),
        (
            "backend",
            "POST",
            "/api/v1/tabs/open",
            {"urls": ["http://example.com/"]},
            ["agent"],
        ),
        (
            "backend",
            "POST",
            "/api/v1/callback/scrape-complete",
            {"session_id": "x", "url": "http://example.com/", "status": "failed",
             "content": None, "metadata": {}},
            ["callback", "ai"],
        ),
    ],
)
def test_sec21_fail_closed_on_unset_config(
    request, service, method, path, body, unset, monkeypatch
):
    """Removing the token config must fail closed (401), never open (200)."""
    monkeypatch.delenv("AI_ENGINE_ALLOW_UNAUTHENTICATED", raising=False)
    for scope in unset:
        monkeypatch.delenv(TOKEN_ENVS[scope], raising=False)
    client = request.getfixturevalue(service)
    response = client.request(method, path, token=None, json=body)
    assert response.status_code == 401


@pytest.mark.sec_managed
def test_sec22_ai_allow_unauthenticated_default_off(ai, monkeypatch):
    monkeypatch.delenv(TOKEN_ENVS["ai"], raising=False)

    monkeypatch.delenv("AI_ENGINE_ALLOW_UNAUTHENTICATED", raising=False)
    denied = ai.post("/embed", token=None, json={"texts": ["x"]})
    assert denied.status_code == 401

    monkeypatch.setenv("AI_ENGINE_ALLOW_UNAUTHENTICATED", "true")
    allowed = ai.post("/embed", token=None, json={"texts": ["x"]})
    # Escape hatch opens the auth gate; the hermetic env has no embedding
    # backend, so "opened" is observable as any non-401 status.
    assert allowed.status_code != 401


@pytest.mark.parametrize("service", ["backend", "ai", "browser"])
def test_sec23_health_is_tokenless(request, service):
    client = request.getfixturevalue(service)
    response = client.get("/health", token=None)
    assert response.status_code != 401, "health must never require a secret"
    assert response.status_code in (200, 503)
