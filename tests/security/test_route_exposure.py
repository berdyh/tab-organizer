"""SEC-44/45: what backend-core exposes without a token, and what it hands back.

SEC-44 is the probe this suite was missing. Two backend routes
(``GET /api/v1/auth/pending`` and ``POST /api/v1/auth/credentials``) shipped
with no auth dependency while attaching the privileged Browser Engine service
token server-side -- a confused deputy that let any unauthenticated caller read
the pending-auth queue and plant credentials for an arbitrary domain. Nothing
in the suite would have failed. The route table is enumerated from the running
service's own OpenAPI document rather than a hand-maintained list, so a newly
added route is unclassified and fails this test until someone puts it on the
public allowlist below (with a reason) or gives it an auth dependency.

SEC-45 covers the other half of the same exfiltration chain: ``GET
/api/v1/urls/{session_id}`` used to return ``r.metadata`` verbatim, and the
ingest writer stores the whole captured page body under ``metadata["content"]``,
so the list endpoint dumped the full text of every captured page. It now returns
an explicit whitelist projection. Consumers that genuinely need page content use
``/export``.

Both probes are plain HTTP against the service's own published contract and port
directly to the TypeScript implementation.
"""

import os
import re
import uuid

import pytest

from tests.security.conftest import TOKEN_ENVS

pytestmark = [pytest.mark.security]

PATH_PARAM = re.compile(r"\{[^}]+\}")
PLACEHOLDER = "sec-probe-nonexistent"


# ---------------------------------------------------------------------------
# SEC-44: every backend route is either authenticated or deliberately public
# ---------------------------------------------------------------------------

# Routes that are unauthenticated ON PURPOSE, each with the reason it is safe
# or accepted. Anything not listed here must answer 401 to an anonymous caller.
# `/api/v1/platform/*` is excluded from this suite entirely (plan decision 41)
# and is filtered out before the comparison.
PUBLIC_ROUTES: dict[tuple[str, str], str] = {
    # Liveness/identity. Must never require a secret (see SEC-23).
    ("GET", "/"): "service identity banner, no user data",
    ("GET", "/health"): "liveness probe; scripts/cli.py polls it",
    ("GET", "/api/v1/health"): "aggregated liveness probe for the local UI",
    # Session/URL CRUD the local Streamlit UI drives without a token. These are
    # loopback-only by deployment (docker-compose binds 127.0.0.1) and are the
    # accepted local-single-user boundary, not an oversight. They are exactly
    # why CORS must stay an allowlist (SEC-43) and why the /urls projection must
    # not carry page content (SEC-45).
    ("POST", "/api/v1/sessions"): "local UI creates sessions without a token",
    ("GET", "/api/v1/sessions"): "local UI session picker",
    ("GET", "/api/v1/sessions/{session_id}"): "local UI session stats",
    ("DELETE", "/api/v1/sessions/{session_id}"): "local UI session delete",
    ("POST", "/api/v1/urls"): "local UI URL input",
    ("GET", "/api/v1/urls/{session_id}"): "local UI URL list (projection: SEC-45)",
    ("POST", "/api/v1/scrape"): "local UI starts a scrape batch",
    ("GET", "/api/v1/scrape/status/{session_id}"): "local UI polls batch status",
    ("POST", "/api/v1/cluster"): "local UI clustering trigger",
    ("GET", "/api/v1/clusters/{session_id}"): "local UI cluster view",
    ("POST", "/api/v1/export"): "local UI export download",
}


def _route_table(backend) -> list[tuple[str, str]]:
    """Enumerate (METHOD, path) from the service's own OpenAPI document."""
    response = backend.get("/openapi.json", token=None)
    assert response.status_code == 200, (
        "backend must publish an OpenAPI document so the route table can be "
        f"enumerated instead of hand-maintained (got {response.status_code})"
    )
    routes: list[tuple[str, str]] = []
    for path, operations in response.json().get("paths", {}).items():
        if "/platform/" in path:
            continue  # excluded from this suite (plan decision 41)
        for method in operations:
            if method.upper() in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
                routes.append((method.upper(), path))
    assert routes, "OpenAPI document exposed no routes"
    return sorted(routes)


def test_sec44_route_table_is_fully_classified(backend):
    """Every allowlist entry must still exist; no stale exemptions."""
    live = set(_route_table(backend))
    stale = sorted(set(PUBLIC_ROUTES) - live)
    assert not stale, (
        "PUBLIC_ROUTES exempts routes that no longer exist; remove them so the "
        f"allowlist cannot silently cover a future route: {stale}"
    )


def test_sec44_unclassified_routes_require_auth(backend):
    """Table-driven: anonymous callers get 401 on every non-public route.

    A newly added route is unclassified by construction, so it lands here and
    fails until someone either protects it or documents it in PUBLIC_ROUTES.
    """
    unauthenticated_public: list[str] = []
    for method, path in _route_table(backend):
        if (method, path) in PUBLIC_ROUTES:
            continue
        concrete = PATH_PARAM.sub(PLACEHOLDER, path)
        response = backend.request(method, concrete, token=None, json={})
        if response.status_code != 401:
            unauthenticated_public.append(f"{method} {path} -> {response.status_code}")

    assert not unauthenticated_public, (
        "backend routes reachable without a token and not on the reviewed "
        "public allowlist (classify them in PUBLIC_ROUTES with a reason, or "
        f"add an auth dependency): {unauthenticated_public}"
    )


def test_sec44_public_allowlist_is_actually_reachable(backend):
    """Non-vacuity: allowlisted routes really are anonymous-reachable.

    Without this, a route could be quietly protected *and* allowlisted, hiding
    the fact that the allowlist no longer describes reality.
    """
    wrongly_protected: list[str] = []
    for method, path in sorted(PUBLIC_ROUTES):
        concrete = PATH_PARAM.sub(PLACEHOLDER, path)
        response = backend.request(method, concrete, token=None, json={})
        if response.status_code == 401:
            wrongly_protected.append(f"{method} {path}")
    assert not wrongly_protected, (
        "PUBLIC_ROUTES claims these are intentionally public but they now "
        f"require auth; drop them from the allowlist: {wrongly_protected}"
    )


# ---------------------------------------------------------------------------
# SEC-45: the URL listing must not carry captured page content
# ---------------------------------------------------------------------------

SECRET_BODY = "sec45-captured-page-body-must-not-be-listed"


def test_sec45_url_listing_carries_no_page_content(backend):
    """Capture a page with a known body, then list URLs and look for it."""
    callback_token = os.environ.get(TOKEN_ENVS["callback"])
    if not callback_token:
        pytest.skip("callback token unavailable; cannot ingest a capture")

    created = backend.post("/api/v1/sessions", token=None, json={"name": "sec45"})
    assert created.status_code == 200, created.text
    session_id = created.json()["id"]

    url = f"http://example.com/sec45-{uuid.uuid4().hex}"
    added = backend.post(
        "/api/v1/urls", token=None, json={"urls": [url], "session_id": session_id}
    )
    assert added.status_code == 200, added.text

    ingested = backend.post(
        "/api/v1/ingest/v1",
        token=callback_token,
        json={
            "capture_id": f"sec45-{uuid.uuid4().hex}",
            "attempt": 1,
            "session_id": session_id,
            "url": url,
            "status": "success",
            "content": SECRET_BODY,
            "metadata": {
                "title": "sec45 title",
                "status_code": 200,
                "description": SECRET_BODY,
            },
            "auth_used": False,
            "fetched_at": "2026-07-24T00:00:00",
        },
    )
    assert ingested.status_code == 200, ingested.text
    assert ingested.json()["status"] == "applied", ingested.text

    listing = backend.get(f"/api/v1/urls/{session_id}", token=None)
    assert listing.status_code == 200, listing.text
    records = listing.json()

    # Non-vacuity: the capture really did land, so an unfiltered endpoint
    # would have had the body to leak.
    assert any(r["status"] == "scraped" for r in records), records

    assert SECRET_BODY not in listing.text, (
        "GET /api/v1/urls/{session_id} returned captured page text; the listing "
        "must be a whitelist projection -- content belongs to /export only"
    )
    for record in records:
        metadata = record.get("metadata", {})
        assert "content" not in metadata, (
            f"URL listing metadata carries a content field: {sorted(metadata)}"
        )
        # Deny-by-default: unknown metadata keys are dropped, so a future
        # capture field carrying sensitive data cannot leak by accident.
        assert set(metadata) <= {
            "title",
            "status_code",
            "auth_type",
            "auth_used",
            "capture_id",
        }, f"URL listing metadata carries unreviewed keys: {sorted(metadata)}"

    backend.delete(f"/api/v1/sessions/{session_id}", token=None)
