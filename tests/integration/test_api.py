"""Integration tests for API endpoints."""

import pytest
import httpx
import os
import uuid

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")
AI_URL = os.getenv("AI_ENGINE_URL", "http://localhost:8090")
BROWSER_URL = os.getenv("BROWSER_ENGINE_URL", "http://localhost:8083")
SCRAPE_TERMINAL_STATUSES = {"success", "failed", "timeout", "blocked", "auth_required"}


def browser_headers() -> dict[str, str]:
    """Browser Engine accepts BROWSER_ENGINE_API_TOKEN only (no cross-scope fallback)."""
    token = os.getenv("BROWSER_ENGINE_API_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def ai_headers() -> dict[str, str]:
    """Every AI Engine endpoint except /health requires AI_ENGINE_API_TOKEN."""
    token = os.getenv("AI_ENGINE_API_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


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
