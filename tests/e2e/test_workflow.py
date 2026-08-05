"""End-to-end tests for complete workflows."""

import pytest
import httpx
import os
import time
import uuid

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")
AI_URL = os.getenv("AI_ENGINE_URL", "http://localhost:8090")
BROWSER_URL = os.getenv("BROWSER_ENGINE_URL", "http://localhost:8083")
TERMINAL_URL_STATUSES = {"scraped", "failed", "auth_required"}


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
    """One `IngestResultV1` wire payload as a capture process posts it."""
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


def embedding_backend_status(client) -> tuple[bool, str]:
    """Whether ai-engine can actually embed, and why not when it cannot.

    Provider selection is fail-closed on this branch: with no
    `EMBEDDING_PROVIDER` chosen, ai-engine starts `degraded` and `/index`
    fails. Tests that need the vector leg must say so out loud rather than
    passing on an empty index.
    """
    try:
        response = client.get(f"{AI_URL}/health")
    except httpx.RequestError as error:
        return False, f"ai-engine /health unreachable: {error}"
    if response.status_code != 200:
        return False, f"ai-engine /health returned HTTP {response.status_code}"
    data = response.json()
    if data.get("status") != "healthy":
        runtime = data.get("runtime") or {}
        return False, (
            f"ai-engine reports status={data.get('status')!r}: "
            f"{runtime.get('reason') or runtime}"
        )
    providers = data.get("providers")
    if not isinstance(providers, dict):
        return False, (
            "ai-engine /health carries no `providers` block, so no embedding "
            "provider is announced (pre-provider-routing build, or none chosen)"
        )
    embedding = providers.get("embedding") or providers.get("embeddings") or {}
    if not embedding.get("provider"):
        return False, (
            "ai-engine announces no embedding provider: "
            f"{embedding.get('error') or embedding}"
        )
    return True, ""


@pytest.fixture
def client():
    """Create HTTP client."""
    return httpx.Client(timeout=60.0)


class TestCompleteWorkflow:
    """End-to-end tests for complete user workflows."""
    
    def test_full_workflow(self, client):
        """Test complete workflow: create session, add URLs, scrape, cluster."""
        # 1. Create session
        session_resp = client.post(
            f"{BACKEND_URL}/api/v1/sessions",
            json={"name": "E2E Test Session"},
        )
        assert session_resp.status_code == 200
        session_id = session_resp.json()["id"]

        try:
            # 2. Add public non-Soliq URLs.
            urls = [
                "https://example.com/",
                "https://www.iana.org/domains/reserved",
            ]

            url_resp = client.post(
                f"{BACKEND_URL}/api/v1/urls",
                json={"urls": urls, "session_id": session_id},
            )
            assert url_resp.status_code == 200
            assert url_resp.json()["added"] == 2

            # 3. Get session stats
            stats_resp = client.get(f"{BACKEND_URL}/api/v1/sessions/{session_id}")
            assert stats_resp.status_code == 200
            assert stats_resp.json()["total_urls"] == 2

            # 4. Start scraping
            scrape_resp = client.post(
                f"{BACKEND_URL}/api/v1/scrape",
                json={"session_id": session_id},
            )
            assert scrape_resp.status_code == 200

            # 5. Wait for a terminal browser-engine scrape status.
            max_wait = 60
            start_time = time.time()
            status = None

            while time.time() - start_time < max_wait:
                status_resp = client.get(
                    f"{BROWSER_URL}/scrape/status/{session_id}",
                    headers=browser_headers(),
                )
                if status_resp.status_code == 200:
                    status = status_resp.json()
                    if status.get("status") in {
                        "completed",
                        "completed_with_downstream_errors",
                        "failed",
                    }:
                        break
                time.sleep(2)

            assert status is not None, "Browser-engine scrape status was never created"
            assert status.get("status") in {
                "completed",
                "completed_with_downstream_errors",
            }, status
            assert status.get("total") == len(urls)
            assert status.get("completed") == len(urls)
            assert status.get("backend_callback_failed", 0) == 0, status

            url_status_resp = client.get(f"{BACKEND_URL}/api/v1/urls/{session_id}")
            assert url_status_resp.status_code == 200
            records = url_status_resp.json()
            assert len(records) == len(urls)

            statuses = {record["original"]: record["status"] for record in records}
            assert set(statuses.values()) <= TERMINAL_URL_STATUSES
            assert all(status != "pending" for status in statuses.values())
            assert any(status == "scraped" for status in statuses.values()), statuses
        finally:
            # 6. Clean up
            client.delete(f"{BACKEND_URL}/api/v1/sessions/{session_id}")
    
    def test_deduplication_workflow(self, client):
        """Test URL deduplication across multiple additions."""
        # Create session
        session_resp = client.post(
            f"{BACKEND_URL}/api/v1/sessions",
            json={"name": "Dedup E2E Test"},
        )
        session_id = session_resp.json()["id"]
        
        # Add URLs with various formats
        urls_batch1 = [
            "https://example.com/page",
            "https://www.example.com/page/",  # Same as above (normalized)
            "https://example.com/page?utm_source=test",  # Same (tracking removed)
        ]
        
        resp1 = client.post(
            f"{BACKEND_URL}/api/v1/urls",
            json={"urls": urls_batch1, "session_id": session_id},
        )
        
        # Should only add 1 unique URL
        assert resp1.json()["added"] == 1
        assert resp1.json()["duplicates"] == 2
        
        # Add more URLs
        urls_batch2 = [
            "https://example.com/page",  # Duplicate
            "https://example.com/other",  # New
        ]
        
        resp2 = client.post(
            f"{BACKEND_URL}/api/v1/urls",
            json={"urls": urls_batch2, "session_id": session_id},
        )
        
        assert resp2.json()["added"] == 1
        assert resp2.json()["duplicates"] == 1
        
        # Verify total
        stats = client.get(f"{BACKEND_URL}/api/v1/sessions/{session_id}").json()
        assert stats["total_urls"] == 2
        
        # Clean up
        client.delete(f"{BACKEND_URL}/api/v1/sessions/{session_id}")
    
    def test_export_workflow(self, client):
        """Test session export functionality."""
        # Create session with URLs
        session_resp = client.post(
            f"{BACKEND_URL}/api/v1/sessions",
            json={"name": "Export Test"},
        )
        session_id = session_resp.json()["id"]
        
        client.post(
            f"{BACKEND_URL}/api/v1/urls",
            json={
                "urls": ["https://example.com/1", "https://example.com/2"],
                "session_id": session_id,
            },
        )
        
        # Export as markdown
        export_resp = client.post(
            f"{BACKEND_URL}/api/v1/export",
            json={"session_id": session_id, "format": "markdown"},
        )
        
        assert export_resp.status_code == 200
        data = export_resp.json()
        assert "content" in data
        assert "Export Test" in data["content"]
        
        # Export as JSON
        json_resp = client.post(
            f"{BACKEND_URL}/api/v1/export",
            json={"session_id": session_id, "format": "json"},
        )
        
        assert json_resp.status_code == 200
        
        # Clean up
        client.delete(f"{BACKEND_URL}/api/v1/sessions/{session_id}")


class TestCaptureIngestWorkflow:
    """The capture-process workflow through `POST /api/v1/ingest/v1`.

    `test_full_workflow` above drives browser-engine, which delivers its own
    results to this endpoint. This one plays the capture process directly --
    the wk10 cutover shape, where the TS capture process is the caller -- and
    follows one url from capture to keyword-searchable, through a retry and a
    recapture.
    """

    def _ingest(self, client, payload):
        return client.post(
            f"{BACKEND_URL}/api/v1/ingest/v1",
            json=payload,
            headers=callback_headers(),
        )

    def _search(self, client, session_id, query, mode="keyword"):
        response = client.post(
            f"{BACKEND_URL}/api/v1/search",
            json={"query": query, "session_id": session_id, "mode": mode},
            headers=agent_headers(),
        )
        return response

    def test_capture_to_searchable_workflow(self, client):
        """Capture -> stored -> keyword-searchable -> retried -> recaptured."""
        marker = uuid.uuid4().hex[:10]
        session_resp = client.post(
            f"{BACKEND_URL}/api/v1/sessions",
            json={"name": "Capture Ingest E2E"},
        )
        assert session_resp.status_code == 200
        session_id = session_resp.json()["id"]

        try:
            captured_url = f"https://example.com/e2e-captured-{marker}"
            blocked_url = f"https://example.com/e2e-blocked-{marker}"
            add_resp = client.post(
                f"{BACKEND_URL}/api/v1/urls",
                json={"urls": [captured_url, blocked_url], "session_id": session_id},
            )
            assert add_resp.status_code == 200
            assert add_resp.json()["added"] == 2

            # 1. A successful capture and a failed one, as one batch would send.
            success = capture_payload(
                session_id,
                captured_url,
                content=f"the capture body mentions {marker}quokka once",
                metadata={"title": "Captured Page", "status_code": 200},
            )
            failure = capture_payload(
                session_id,
                blocked_url,
                status="blocked",
                content=None,
                metadata={"title": "Blocked Page", "status_code": 403},
            )
            applied = self._ingest(client, success)
            assert applied.status_code == 200, applied.text
            assert applied.json()["status"] == "applied"
            assert applied.json()["index"] == "pending"

            blocked = self._ingest(client, failure)
            assert blocked.status_code == 200, blocked.text
            # No content to index, so no forward is scheduled at all.
            assert blocked.json() == {
                "status": "applied",
                "capture_id": failure["capture_id"],
                "index": "skipped",
            }

            # 2. Both url records reached a terminal status.
            records = {
                record["original"]: record
                for record in client.get(
                    f"{BACKEND_URL}/api/v1/urls/{session_id}"
                ).json()
            }
            assert records[captured_url]["status"] == "scraped"
            assert records[blocked_url]["status"] == "failed"
            assert set(records) == {captured_url, blocked_url}

            # 3. The FTS row is written in the SAME transaction as the ledger
            #    row, so keyword search must find the capture immediately --
            #    no vector store and no embedding provider involved.
            hits = self._search(client, session_id, f"{marker}quokka")
            assert hits.status_code == 200, hits.text
            found = hits.json()["results"]
            assert [hit["url"] for hit in found] == [captured_url], found
            assert found[0]["title"] == "Captured Page"

            # 4. A retried delivery of the same capture changes nothing.
            replay = self._ingest(client, success)
            assert replay.status_code == 200
            assert replay.json()["reason"] == "duplicate_capture_id"
            assert replay.json()["capture_id"] == success["capture_id"]

            # 5. A genuine recapture (newer fetched_at) replaces the body, and
            #    the search index follows it rather than keeping both.
            recapture = capture_payload(
                session_id,
                captured_url,
                attempt=2,
                fetched_at="2026-07-24T12:00:00",
                content=f"the recaptured body mentions {marker}wombat instead",
                metadata={"title": "Recaptured Page", "status_code": 200},
            )
            assert self._ingest(client, recapture).json()["status"] == "applied"

            new_hits = self._search(client, session_id, f"{marker}wombat")
            assert [hit["url"] for hit in new_hits.json()["results"]] == [captured_url]
            assert self._search(client, session_id, f"{marker}quokka").json()[
                "results"
            ] == []

            exported = client.post(
                f"{BACKEND_URL}/api/v1/export",
                json={"session_id": session_id, "format": "json"},
            ).json()["content"]
            assert f"{marker}wombat" in exported
            assert f"{marker}quokka" not in exported
        finally:
            client.delete(f"{BACKEND_URL}/api/v1/sessions/{session_id}")

    @pytest.mark.requires_ollama
    @pytest.mark.requires_lancedb
    def test_captured_content_becomes_semantically_searchable(self, client):
        """The full leg: capture -> backend forward -> ai-engine /index -> vector hit.

        This is the ONE part of the ingest path that cannot run without a real
        embedding provider. It skips loudly (never silently) when ai-engine
        announces none, and when one IS announced it must pass -- a forward
        that fails then is a genuine defect, not an environment quirk.
        """
        ready, reason = embedding_backend_status(client)
        if not ready:
            pytest.skip(
                "no usable embedding provider, so the capture->vector leg "
                f"cannot be exercised: {reason}"
            )

        marker = uuid.uuid4().hex[:10]
        session_id = client.post(
            f"{BACKEND_URL}/api/v1/sessions",
            json={"name": "Capture Vector E2E"},
        ).json()["id"]
        try:
            url = f"https://example.com/e2e-vector-{marker}"
            client.post(
                f"{BACKEND_URL}/api/v1/urls",
                json={"urls": [url], "session_id": session_id},
            )
            payload = capture_payload(
                session_id,
                url,
                content=(
                    "Hydroponic lettuce grows in nutrient solution without soil. "
                    f"Reference {marker}."
                ),
                metadata={"title": "Soil-free Farming", "status_code": 200},
            )
            assert self._ingest(client, payload).json()["index"] == "pending"

            # The forward runs after the response; replaying reports the stored
            # row's state and re-drives a pending/failed forward.
            deadline = time.time() + 120
            state = "pending"
            while time.time() < deadline:
                state = self._ingest(client, payload).json().get("index")
                if state != "pending":
                    break
                time.sleep(2)
            assert state == "indexed", (
                "ai-engine announced an embedding provider but the single-writer "
                f"forward did not index the capture (state={state!r})"
            )

            semantic = self._search(
                client, session_id, "growing plants without soil", mode="semantic"
            )
            assert semantic.status_code == 200, semantic.text
            assert any(
                hit["url"] == url for hit in semantic.json()["results"]
            ), semantic.json()
        finally:
            client.delete(f"{BACKEND_URL}/api/v1/sessions/{session_id}")


class TestAuthWorkflow:
    """End-to-end tests for authentication workflows."""
    
    def test_auth_queue_workflow(self, client):
        """Test authentication queue functionality."""
        # Check initial pending auth
        pending_resp = client.get(f"{BROWSER_URL}/auth/pending", headers=browser_headers())
        assert pending_resp.status_code == 200
        initial_count = pending_resp.json()["pending_count"]
        
        # The auth queue should be accessible
        assert isinstance(pending_resp.json()["pending"], list)


class TestProviderSwitching:
    """End-to-end tests for AI provider switching."""
    
    def test_get_current_providers(self, client):
        """Test getting current provider configuration."""
        resp = client.get(f"{AI_URL}/providers", headers=ai_headers())
        assert resp.status_code == 200
        
        data = resp.json()
        assert "llm" in data
        assert "provider" in data["llm"]
        assert "embeddings" in data
        assert "provider" in data["embeddings"]
