"""SEC-43: browser-origin (CORS) policy on all three services.

All three services used to run ``allow_origins=["*"]`` together with
``allow_credentials=True``. Verified live before the fix: a preflight claiming
``Origin: https://evil.example`` came back 200 echoing that origin with
``access-control-allow-credentials: true``. Combined with the unauthenticated
read endpoints on backend-core, that is a full cross-site corpus-exfiltration
primitive -- any page the user has open while the stack runs can fetch
``/api/v1/sessions`` then ``/api/v1/urls/{id}``.

None of these services has a legitimate browser-side caller: the Streamlit UI
calls them from the server side with ``requests``; the user's browser only ever
talks to Streamlit on :8089. The policy is therefore an explicit origin
allowlist with credentials disabled.

The suite had no probe that issued an ``Origin`` header anywhere before this,
so the hole was invisible to CI. These probes are pure HTTP and port directly
to the TypeScript implementation.
"""

import os

import pytest

pytestmark = [pytest.mark.security]

FOREIGN_ORIGIN = "https://evil.example"

# Same env contract the services read (services/cors.py). The TS port must
# honour the same variable and the same default.
CORS_ORIGINS_ENV = "CORS_ALLOWED_ORIGINS"
DEFAULT_UI_ORIGIN = "http://localhost:8089"

SERVICES = ["backend", "ai", "browser"]


def _configured_origin() -> str:
    raw = os.getenv(CORS_ORIGINS_ENV, "")
    for candidate in raw.split(","):
        candidate = candidate.strip().rstrip("/")
        if candidate:
            return candidate
    return DEFAULT_UI_ORIGIN


def _assert_no_credentials(response) -> None:
    allow_credentials = response.headers.get("access-control-allow-credentials")
    assert (allow_credentials or "").lower() != "true", (
        "access-control-allow-credentials must never be true: combined with a "
        "reflected origin it lets a foreign page read authenticated responses"
    )


@pytest.mark.parametrize("service", SERVICES)
def test_sec43_preflight_does_not_admit_foreign_origin(request, service):
    """A preflight from an arbitrary site must not be granted."""
    client = request.getfixturevalue(service)
    response = client.request(
        "OPTIONS",
        "/health",
        token=None,
        headers={
            "Origin": FOREIGN_ORIGIN,
            "Access-Control-Request-Method": "GET",
        },
    )
    allow_origin = response.headers.get("access-control-allow-origin")
    assert allow_origin not in (FOREIGN_ORIGIN, "*"), (
        f"{service} preflight granted foreign origin {FOREIGN_ORIGIN!r} "
        f"(access-control-allow-origin={allow_origin!r})"
    )
    _assert_no_credentials(response)


@pytest.mark.parametrize("service", SERVICES)
def test_sec43_simple_request_does_not_echo_foreign_origin(request, service):
    """The actual (non-preflight) response must not be readable cross-site."""
    client = request.getfixturevalue(service)
    response = client.get("/health", token=None, headers={"Origin": FOREIGN_ORIGIN})
    allow_origin = response.headers.get("access-control-allow-origin")
    assert allow_origin not in (FOREIGN_ORIGIN, "*"), (
        f"{service} response is readable by {FOREIGN_ORIGIN!r} "
        f"(access-control-allow-origin={allow_origin!r})"
    )
    _assert_no_credentials(response)


@pytest.mark.sec_managed
@pytest.mark.parametrize("service", SERVICES)
def test_sec43_configured_ui_origin_is_still_allowed(request, service):
    """Non-vacuity: the policy is an allowlist, not a removed middleware.

    Managed-only because attached mode cannot know the running server's
    configured origin.
    """
    client = request.getfixturevalue(service)
    origin = _configured_origin()
    response = client.request(
        "OPTIONS",
        "/health",
        token=None,
        headers={"Origin": origin, "Access-Control-Request-Method": "GET"},
    )
    assert response.headers.get("access-control-allow-origin") == origin, (
        f"{service} did not grant its own configured UI origin {origin!r}; "
        "the allowlist is misconfigured or CORS was removed entirely"
    )
    _assert_no_credentials(response)
