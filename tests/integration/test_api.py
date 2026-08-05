"""Integration tests for API endpoints."""

import concurrent.futures
import time
from datetime import datetime, timedelta, timezone

import pytest
import httpx
import os
import uuid

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")
AI_URL = os.getenv("AI_ENGINE_URL", "http://localhost:8090")
BROWSER_URL = os.getenv("BROWSER_ENGINE_URL", "http://localhost:8083")
SCRAPE_TERMINAL_STATUSES = {"success", "failed", "timeout", "blocked", "auth_required"}

# `index` values a stored ingest ledger row can report back through the
# duplicate-receipt response. "pending" is the only non-settled one.
SETTLED_INDEX_STATES = {"indexed", "failed", "skipped", "superseded"}

# Computed against the clock rather than hardcoded: the boundary is
# `MAX_FUTURE_CLOCK_SKEW` (5 minutes) ahead of the SERVER's now, so a literal
# date would stop being "far future" once the calendar reached it.
FAR_FUTURE_FETCHED_AT = (
    datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=1)
).isoformat(timespec="seconds")


def browser_headers() -> dict[str, str]:
    """Browser Engine accepts BROWSER_ENGINE_API_TOKEN only (no cross-scope fallback)."""
    token = os.getenv("BROWSER_ENGINE_API_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def ai_headers() -> dict[str, str]:
    """Every AI Engine endpoint except /health requires AI_ENGINE_API_TOKEN."""
    token = os.getenv("AI_ENGINE_API_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def callback_headers() -> dict[str, str]:
    """`POST /api/v1/ingest/v1` accepts BACKEND_CALLBACK_TOKEN (callback scope)."""
    token = os.getenv("BACKEND_CALLBACK_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def agent_headers() -> dict[str, str]:
    """Backend agent-scoped routes (/search, /chat, auth proxies) take this one."""
    token = os.getenv("BACKEND_AGENT_API_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def capture_payload(session_id: str, url: str, **overrides) -> dict:
    """One `IngestResultV1` wire payload, as browser-engine (and, at cutover, the
    TS capture process) posts it. Defaults are an indexable success capture."""
    payload = {
        "capture_id": str(uuid.uuid4()),
        "attempt": 1,
        "session_id": session_id,
        "url": url,
        "status": "success",
        "content": "capture body",
        "metadata": {"title": "Capture Title", "status_code": 200},
        "auth_used": False,
        "fetched_at": "2026-07-24T10:00:00",
    }
    payload.update(overrides)
    return payload


def url_records(client, session_id: str) -> dict[str, dict]:
    """`GET /api/v1/urls/{id}` keyed by original url. The listing is a whitelist
    projection (no page content); content is asserted through /export."""
    response = client.get(f"{BACKEND_URL}/api/v1/urls/{session_id}")
    assert response.status_code == 200, response.text
    return {record["original"]: record for record in response.json()}


def post_capture(payload: dict) -> tuple[int, object]:
    """POST one capture on its own connection (for the concurrency tests).

    Returns the status code alongside the decoded body, or the raw text when
    the body is not JSON, so an unexpected 500 shows up as a status assertion
    rather than as a JSON decode error three lines later.
    """
    with httpx.Client(timeout=60.0) as parallel:
        response = parallel.post(
            f"{BACKEND_URL}/api/v1/ingest/v1",
            json=payload,
            headers=callback_headers(),
        )
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, response.text


def export_content(client, session_id: str) -> str:
    """Session export text -- the only endpoint that still returns page bodies."""
    response = client.post(
        f"{BACKEND_URL}/api/v1/export",
        json={"session_id": session_id, "format": "json"},
    )
    assert response.status_code == 200, response.text
    return response.json()["content"]


@pytest.fixture
def client():
    """Create HTTP client."""
    return httpx.Client(timeout=30.0)


class TestBackendAPI:
    """Integration tests for Backend Core API."""
    
    def test_health_check(self, client):
        """Test health endpoint."""
        response = client.get(f"{BACKEND_URL}/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_create_session(self, client):
        """Test session creation."""
        response = client.post(
            f"{BACKEND_URL}/api/v1/sessions",
            json={"name": "Test Session"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["name"] == "Test Session"
    
    def test_list_sessions(self, client):
        """Test listing sessions."""
        # Create a session first
        client.post(
            f"{BACKEND_URL}/api/v1/sessions",
            json={"name": "List Test"},
        )
        
        response = client.get(f"{BACKEND_URL}/api/v1/sessions")
        assert response.status_code == 200
        sessions = response.json()
        assert isinstance(sessions, list)
    
    def test_add_urls(self, client):
        """Test adding URLs to a session."""
        # Create session
        session_resp = client.post(
            f"{BACKEND_URL}/api/v1/sessions",
            json={"name": "URL Test"},
        )
        session_id = session_resp.json()["id"]
        
        # Add URLs
        response = client.post(
            f"{BACKEND_URL}/api/v1/urls",
            json={
                "urls": [
                    "https://example.com/page1",
                    "https://example.com/page2",
                ],
                "session_id": session_id,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["added"] == 2
        assert data["duplicates"] == 0
    
    def test_add_duplicate_urls(self, client):
        """Test that duplicate URLs are detected."""
        # Create session
        session_resp = client.post(
            f"{BACKEND_URL}/api/v1/sessions",
            json={"name": "Dedup Test"},
        )
        session_id = session_resp.json()["id"]
        
        # Add URLs
        client.post(
            f"{BACKEND_URL}/api/v1/urls",
            json={
                "urls": ["https://example.com/page"],
                "session_id": session_id,
            },
        )
        
        # Add same URL again
        response = client.post(
            f"{BACKEND_URL}/api/v1/urls",
            json={
                "urls": ["https://example.com/page"],
                "session_id": session_id,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["added"] == 0
        assert data["duplicates"] == 1
    
    def test_get_session_stats(self, client):
        """Test getting session statistics."""
        # Create session with URLs
        session_resp = client.post(
            f"{BACKEND_URL}/api/v1/sessions",
            json={"name": "Stats Test"},
        )
        session_id = session_resp.json()["id"]
        
        client.post(
            f"{BACKEND_URL}/api/v1/urls",
            json={
                "urls": ["https://example.com/1", "https://example.com/2"],
                "session_id": session_id,
            },
        )
        
        # Get stats
        response = client.get(f"{BACKEND_URL}/api/v1/sessions/{session_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_urls"] == 2
    
    def test_delete_session(self, client):
        """Test session deletion."""
        # Create session
        session_resp = client.post(
            f"{BACKEND_URL}/api/v1/sessions",
            json={"name": "Delete Test"},
        )
        session_id = session_resp.json()["id"]
        
        # Delete session
        response = client.delete(f"{BACKEND_URL}/api/v1/sessions/{session_id}")
        
        assert response.status_code == 200
        
        # Verify deletion
        get_response = client.get(f"{BACKEND_URL}/api/v1/sessions/{session_id}")
        assert get_response.status_code == 404


class TestIngestV1:
    """`POST /api/v1/ingest/v1` against a live stack (gap G3).

    The endpoint had unit coverage only, so replay protection, ordering, auth
    scope and the durable index-forward outbox had never run over real HTTP
    against real SQLite. These tests exercise exactly that seam; they do NOT
    re-prove the decision table `tests/unit/test_ingest_v1.py` already owns.

    On index forwarding: `forward_capture_index` POSTs to ai-engine `/index`,
    which needs a working embedding provider. When none is configured (this
    branch makes provider selection fail closed) the forward FAILS for real,
    and that is the correct bed for the durability assertions below -- a failed
    forward must leave a recoverable ledger row, not a lost capture. Nothing
    here mocks that failure and nothing here requires it either: the durability
    test asserts the ledger settles and stays addressable whichever way the
    forward goes.
    """

    @pytest.fixture
    def ingest_session(self, client):
        """A session with one registered url; torn down after the test.

        Registration is deliberate: ingest 409s an unregistered url (B3 -- a
        callback-token holder must not be able to inject arbitrary urls), so a
        test that skipped this step would only ever see 409s and could never
        reach the code it claims to cover.
        """
        session_id = client.post(
            f"{BACKEND_URL}/api/v1/sessions",
            json={"name": "Ingest v1 Integration"},
        ).json()["id"]
        url = f"https://example.com/ingest-{uuid.uuid4().hex[:10]}"
        added = client.post(
            f"{BACKEND_URL}/api/v1/urls",
            json={"urls": [url], "session_id": session_id},
        )
        assert added.status_code == 200, added.text
        assert added.json()["added"] == 1
        try:
            yield session_id, url
        finally:
            client.delete(f"{BACKEND_URL}/api/v1/sessions/{session_id}")

    def _ingest(self, client, payload, headers=None):
        return client.post(
            f"{BACKEND_URL}/api/v1/ingest/v1",
            json=payload,
            headers=callback_headers() if headers is None else headers,
        )

    def _settle_index(self, client, payload, timeout=60.0):
        """Replay the capture until its stored ledger row leaves ``pending``.

        A duplicate receipt returns the STORED row's index state, which is the
        only view of the forward outbox reachable over HTTP, and it also
        re-schedules the forward for a `pending`/`failed` row (the documented
        retry-heals path). So this both observes and drives the sweep.
        """
        deadline = time.time() + timeout
        state = None
        while time.time() < deadline:
            response = self._ingest(client, payload)
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["status"] == "ignored"
            assert body["reason"] == "duplicate_capture_id"
            state = body.get("index")
            if state in SETTLED_INDEX_STATES:
                return state
            time.sleep(1.5)
        raise AssertionError(
            "ingest ledger row never left 'pending': the index forward neither "
            f"succeeded nor failed within {timeout}s (last state={state!r})"
        )

    def test_applied_capture_writes_one_url_record(self, client, ingest_session):
        """Baseline the rest of the class depends on: a capture posted with the
        callback token lands in SQLite and is visible through the read paths."""
        session_id, url = ingest_session
        payload = capture_payload(session_id, url, content="alpha capture body")

        response = self._ingest(client, payload)

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "applied"
        assert body["capture_id"] == payload["capture_id"]
        # success + content is indexable, so a forward was scheduled.
        assert body["index"] == "pending"

        records = url_records(client, session_id)
        assert set(records) == {url}
        assert records[url]["status"] == "scraped"
        assert records[url]["metadata"]["title"] == "Capture Title"
        assert "alpha capture body" in export_content(client, session_id)

    def test_replay_is_idempotent_and_never_overwrites_the_stored_body(
        self, client, ingest_session
    ):
        """At-least-once delivery: the same capture_id twice leaves one row, and
        a redelivery carrying a MUTATED body must not clobber the stored one."""
        session_id, url = ingest_session
        payload = capture_payload(session_id, url, content="original body text")

        first = self._ingest(client, payload)
        assert first.json()["status"] == "applied"

        replay = self._ingest(
            client,
            {**payload, "content": "mutated body text", "auth_used": True},
        )

        assert replay.status_code == 200, replay.text
        body = replay.json()
        assert body["status"] == "ignored"
        assert body["reason"] == "duplicate_capture_id"
        # capture_id echoed is the STORED one, and `index` summarizes the stored
        # row rather than the replay.
        assert body["capture_id"] == payload["capture_id"]
        assert body["index"] in SETTLED_INDEX_STATES | {"pending"}

        records = url_records(client, session_id)
        assert len(records) == 1
        assert records[url]["status"] == "scraped"
        exported = export_content(client, session_id)
        assert "original body text" in exported
        assert "mutated body text" not in exported

    @pytest.mark.parametrize("order", ["newest_first", "oldest_first"])
    def test_older_capture_cannot_clobber_a_newer_one(
        self, client, ingest_session, order
    ):
        """Newest-wins ordering over the real HTTP + SQLite path.

        `tests/unit/test_ingest_v1.py::test_two_writer_race_older_write_cannot_clobber`
        proves this against a SessionManager; here the same rule has to survive
        two independent requests hitting the service.
        """
        session_id, url = ingest_session
        older = capture_payload(
            session_id,
            url,
            attempt=1,
            fetched_at="2026-07-24T09:00:00",
            content="older capture body",
            metadata={"title": "Older"},
        )
        newer = capture_payload(
            session_id,
            url,
            attempt=2,
            fetched_at="2026-07-24T11:00:00",
            content="newer capture body",
            metadata={"title": "Newer"},
        )

        if order == "newest_first":
            assert self._ingest(client, newer).json()["status"] == "applied"
            late = self._ingest(client, older)
            assert late.status_code == 200, late.text
            # Stale rejection is a 200 "ignored", never a 4xx browser-engine
            # would miscount as a failed callback.
            assert late.json() == {
                "status": "ignored",
                "reason": "stale_capture",
                "capture_id": older["capture_id"],
            }
        else:
            assert self._ingest(client, older).json()["status"] == "applied"
            assert self._ingest(client, newer).json()["status"] == "applied"

        records = url_records(client, session_id)
        assert len(records) == 1
        assert records[url]["metadata"]["title"] == "Newer"
        exported = export_content(client, session_id)
        assert "newer capture body" in exported
        assert "older capture body" not in exported

    def test_ingest_accepts_only_the_callback_token_scope(
        self, client, ingest_session
    ):
        """The four service tokens must stay scope-isolated (CLAUDE.md).

        The frozen suite's `backend_ingest_v1` door checks this in managed mode
        with synthetic per-scope tokens; this runs it against the tokens the
        stack was actually started with, which is where a collapse would show.
        """
        session_id, url = ingest_session
        callback = os.getenv("BACKEND_CALLBACK_TOKEN", "").strip()
        others = {
            "ai": os.getenv("AI_ENGINE_API_TOKEN", "").strip(),
            "browser": os.getenv("BROWSER_ENGINE_API_TOKEN", "").strip(),
        }
        if not callback:
            pytest.fail(
                "BACKEND_CALLBACK_TOKEN must be set for this suite; run "
                "./scripts/cli.py test --type integration, which mints it"
            )
        missing = [name for name, value in others.items() if not value]
        if missing:
            pytest.fail(
                "foreign-principal tokens are unset, so this probe could not "
                f"tell rejection from an empty header: {sorted(missing)}"
            )

        # Non-vacuity: the callback token really does open this door, so a 401
        # below is scope isolation and not a wrong url or a malformed body.
        accepted = self._ingest(client, capture_payload(session_id, url))
        assert accepted.status_code == 200, accepted.text

        payload = capture_payload(session_id, url)
        assert self._ingest(client, payload, headers={}).status_code == 401
        assert (
            self._ingest(
                client, payload, headers={"Authorization": "Bearer not-the-token"}
            ).status_code
            == 401
        )
        for name, token in others.items():
            rejected = self._ingest(
                client, payload, headers={"Authorization": f"Bearer {token}"}
            )
            assert rejected.status_code == 401, (
                f"the {name} service token was accepted by ingest/v1 -- the four "
                "service tokens have collapsed into one principal"
            )

    def test_failed_forward_leaves_a_recoverable_ledger_row(
        self, client, ingest_session
    ):
        """The ingest ledger is the durable outbox for the ai-engine forward.

        Whatever the forward does, the row must reach a settled state and stay
        addressable by capture_id, with the applied content untouched -- that is
        what makes `reconcile_pending_forwards` able to finish the job later.
        A dropped or rewritten row would strand the capture unindexed forever.
        """
        session_id, url = ingest_session
        payload = capture_payload(session_id, url, content="durable body text")

        applied = self._ingest(client, payload)
        assert applied.json()["index"] == "pending"

        state = self._settle_index(client, payload)
        assert state in {"failed", "indexed"}, state

        # Replaying a settled row keeps reporting one state for one stored
        # capture: the retry path never forks the row or loses the content.
        for _ in range(2):
            again = self._ingest(client, payload)
            body = again.json()
            assert body["reason"] == "duplicate_capture_id"
            assert body["capture_id"] == payload["capture_id"]
            assert body["index"] in SETTLED_INDEX_STATES

        records = url_records(client, session_id)
        assert len(records) == 1
        assert records[url]["status"] == "scraped"
        assert "durable body text" in export_content(client, session_id)

    def test_concurrent_replays_apply_exactly_once(self, client, ingest_session):
        """`_forward_lock` and the `BEGIN IMMEDIATE` ordering read exist for
        concurrent delivery. Fire the same capture at the live service from
        several connections at once: exactly one write, the rest acknowledged
        duplicates."""
        session_id, url = ingest_session
        payload = capture_payload(session_id, url, content="concurrent body text")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: post_capture(payload), range(8)))

        # Status codes first: a non-200 here must surface as itself, not as a
        # decode error further down.
        assert [status for status, _ in results] == [200] * 8, results
        bodies = [body for _, body in results]
        applied = [body for body in bodies if body["status"] == "applied"]
        ignored = [body for body in bodies if body["status"] == "ignored"]
        assert len(applied) == 1, results
        assert len(ignored) == 7, results
        assert all(body["reason"] == "duplicate_capture_id" for body in ignored)
        assert all(body["capture_id"] == payload["capture_id"] for body in ignored)

        records = url_records(client, session_id)
        assert len(records) == 1
        assert "concurrent body text" in export_content(client, session_id)

    def test_concurrent_distinct_captures_leave_the_newest_content(
        self, client, ingest_session
    ):
        """Two writers racing on one url: whichever request the server happens
        to serialize first, the newer capture owns the record afterwards."""
        session_id, url = ingest_session
        older = capture_payload(
            session_id,
            url,
            attempt=1,
            fetched_at="2026-07-24T09:00:00",
            content="race older body",
            metadata={"title": "Older"},
        )
        newer = capture_payload(
            session_id,
            url,
            attempt=2,
            fetched_at="2026-07-24T11:00:00",
            content="race newer body",
            metadata={"title": "Newer"},
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(post_capture, [older, newer]))

        assert [status for status, _ in results] == [200, 200], results
        # Arrival order decides whether the older one is `applied` then beaten
        # or rejected as `stale`; both are correct, the final state is not.
        assert {body["status"] for _, body in results} <= {"applied", "ignored"}

        records = url_records(client, session_id)
        assert len(records) == 1
        assert records[url]["metadata"]["title"] == "Newer"
        exported = export_content(client, session_id)
        assert "race newer body" in exported
        assert "race older body" not in exported

    def test_ingest_refuses_unknown_session_and_unregistered_url(
        self, client, ingest_session
    ):
        """Registration stays the dispatcher's job: a callback-token holder can
        neither create a session nor add a url through this endpoint."""
        session_id, url = ingest_session

        unknown_session = self._ingest(
            client, capture_payload(str(uuid.uuid4()), url)
        )
        assert unknown_session.status_code == 404, unknown_session.text

        unregistered = self._ingest(
            client,
            capture_payload(session_id, "https://example.com/never-registered"),
        )
        assert unregistered.status_code == 409, unregistered.text
        assert set(url_records(client, session_id)) == {url}

    @pytest.mark.parametrize(
        "override",
        [
            {"fetched_at": "not-a-timestamp"},
            {"fetched_at": FAR_FUTURE_FETCHED_AT},  # beyond the clock-skew bound
            {"attempt": 0},
            {"status": "made_up_status"},
        ],
        ids=[
            "garbage_fetched_at",
            "far_future_fetched_at",
            "attempt_zero",
            "bad_status",
        ],
    )
    def test_malformed_capture_is_rejected_at_the_boundary(
        self, client, ingest_session, override
    ):
        """422 at the wire boundary, so nothing malformed reaches the ledger --
        an unparseable or far-future `fetched_at` is the content-pinning
        primitive (finding C5)."""
        session_id, url = ingest_session
        response = self._ingest(
            client, capture_payload(session_id, url, **override)
        )
        assert response.status_code == 422, response.text
        assert url_records(client, session_id)[url]["status"] == "pending"


class TestAIEngineAPI:
    """Integration tests for AI Engine API."""
    
    def test_health_check(self, client):
        """Test health endpoint."""
        response = client.get(f"{AI_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in {"healthy", "degraded"}
        assert data["vector_store"]["ready"] is True
        assert "runtime" in data
    
    def test_get_providers(self, client):
        """Test getting provider info."""
        response = client.get(f"{AI_URL}/providers", headers=ai_headers())
        assert response.status_code == 200
        data = response.json()
        assert "llm" in data
        assert "embeddings" in data


class TestBrowserEngineAPI:
    """Integration tests for Browser Engine API."""
    
    def test_health_check(self, client):
        """Test health endpoint."""
        response = client.get(f"{BROWSER_URL}/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_get_pending_auth(self, client):
        """Test getting pending auth requests."""
        response = client.get(f"{BROWSER_URL}/auth/pending", headers=browser_headers())
        assert response.status_code == 200
        data = response.json()
        assert "pending" in data
        assert "pending_count" in data

    def test_single_scrape_public_non_soliq_page(self, client):
        """Browser engine should successfully scrape a public non-Soliq page."""
        public_url = os.getenv(
            "BROWSER_ENGINE_PUBLIC_TEST_URL",
            "https://www.iana.org/domains/reserved",
        )

        response = client.post(
            f"{BROWSER_URL}/scrape/single",
            json={"url": public_url},
            headers=browser_headers(),
        )

        assert response.status_code == 200
        data = response.json()
        assert data["url"] == public_url
        assert data["status"] == "success", data
        assert data["status_code"] is not None
        assert data["content"]


class TestPlatformAPI:
    """Integration tests for local platform auth, B2B, and maintainer flows."""

    def test_platform_auth_b2b_dashboard_and_maintainer_issue_flow(self, client):
        suffix = uuid.uuid4().hex

        signup = client.post(
            f"{BACKEND_URL}/api/v1/platform/auth/signup",
            json={
                "email": f"buyer-{suffix}@example.com",
                "password": "local-secret",
                "name": "Buyer Integration",
                "account_type": "business",
                "company_name": "Buyer Integration Co",
            },
        )
        assert signup.status_code == 200
        session_token = signup.json()["session_token"]
        user_headers = {"Authorization": f"Bearer {session_token}"}

        me = client.get(f"{BACKEND_URL}/api/v1/platform/me", headers=user_headers)
        assert me.status_code == 200
        assert me.json()["user"]["role"] == "b2b"

        companies = client.get(
            f"{BACKEND_URL}/api/v1/platform/companies/search",
            params={"q": "acme"},
            headers=user_headers,
        )
        assert companies.status_code == 200
        assert companies.json()["companies"]

        token_response = client.post(
            f"{BACKEND_URL}/api/v1/platform/b2b/tokens",
            json={"name": "Integration token", "scopes": ["companies:read"]},
            headers=user_headers,
        )
        assert token_response.status_code == 200
        raw_token = token_response.json()["token"]
        assert raw_token.startswith("tbo_")

        first_call = client.get(
            f"{BACKEND_URL}/api/v1/platform/b2b/first-call",
            headers=user_headers,
        )
        assert first_call.status_code == 200
        assert "Authorization: Bearer <api-token>" in first_call.json()["curl"]

        api_search = client.get(
            f"{BACKEND_URL}/api/v1/platform/v1/companies/search",
            params={"query": "acme"},
            headers={"Authorization": f"Bearer {raw_token}"},
        )
        assert api_search.status_code == 200
        assert api_search.json()["companies"]

        dashboard = client.get(
            f"{BACKEND_URL}/api/v1/platform/dashboard",
            headers=user_headers,
        )
        assert dashboard.status_code == 200
        assert dashboard.json()["counters"]["api_requests_total"] >= 1

        issue = client.post(
            f"{BACKEND_URL}/api/v1/platform/issues",
            json={
                "title": "Integration issue",
                "description": "Maintainer should see this issue.",
                "severity": "low",
            },
            headers=user_headers,
        )
        assert issue.status_code == 200

        non_maintainer_issues = client.get(
            f"{BACKEND_URL}/api/v1/platform/maintainer/issues",
            headers=user_headers,
        )
        assert non_maintainer_issues.status_code == 403

        maintainer = client.post(
            f"{BACKEND_URL}/api/v1/platform/auth/signup",
            json={
                "email": f"maintainer-{suffix}@example.com",
                "password": "local-secret",
                "name": "Maintainer Integration",
                "role": "maintainer",
                "maintainer_code": "local-maintainer",
            },
        )
        if maintainer.status_code == 403:
            pytest.fail(
                "PLATFORM_MAINTAINER_SIGNUP_CODE=local-maintainer is required "
                "for maintainer integration coverage"
            )

        assert maintainer.status_code == 200

        issues = client.get(
            f"{BACKEND_URL}/api/v1/platform/maintainer/issues",
            headers={
                "Authorization": f"Bearer {maintainer.json()['session_token']}"
            },
        )
        assert issues.status_code == 200
        assert any(
            item["id"] == issue.json()["issue"]["id"]
            for item in issues.json()["issues"]
        )
