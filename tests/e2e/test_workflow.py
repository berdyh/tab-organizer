"""End-to-end tests for complete workflows."""

import pytest
import httpx
import os
import time

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")
AI_URL = os.getenv("AI_ENGINE_URL", "http://localhost:8090")
BROWSER_URL = os.getenv("BROWSER_ENGINE_URL", "http://localhost:8083")


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
        
        # 2. Add URLs
        urls = [
            "https://httpbin.org/html",
            "https://httpbin.org/robots.txt",
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
        
        # 5. Wait for scraping to complete (with timeout)
        max_wait = 60
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status_resp = client.get(f"{BROWSER_URL}/scrape/status/{session_id}")
            if status_resp.status_code == 200:
                status = status_resp.json()
                if status.get("status") == "completed":
                    break
            time.sleep(2)
        
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


class TestAuthWorkflow:
    """End-to-end tests for authentication workflows."""
    
    def test_auth_queue_workflow(self, client):
        """Test authentication queue functionality."""
        # Check initial pending auth
        pending_resp = client.get(f"{BROWSER_URL}/auth/pending")
        assert pending_resp.status_code == 200
        initial_count = pending_resp.json()["pending_count"]
        
        # The auth queue should be accessible
        assert isinstance(pending_resp.json()["pending"], list)


class TestProviderSwitching:
    """End-to-end tests for AI provider switching."""
    
    def test_get_current_providers(self, client):
        """Test getting current provider configuration."""
        resp = client.get(f"{AI_URL}/providers")
        assert resp.status_code == 200
        
        data = resp.json()
        assert "llm" in data
        assert "provider" in data["llm"]
        assert "embeddings" in data
        assert "provider" in data["embeddings"]
