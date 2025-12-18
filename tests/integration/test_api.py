"""Integration tests for API endpoints."""

import pytest
import httpx
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")
AI_URL = os.getenv("AI_ENGINE_URL", "http://localhost:8090")
BROWSER_URL = os.getenv("BROWSER_ENGINE_URL", "http://localhost:8083")


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
        assert response.json()["status"] == "healthy"
    
    def test_get_providers(self, client):
        """Test getting provider info."""
        response = client.get(f"{AI_URL}/providers")
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
        response = client.get(f"{BROWSER_URL}/auth/pending")
        assert response.status_code == 200
        data = response.json()
        assert "pending" in data
        assert "pending_count" in data
