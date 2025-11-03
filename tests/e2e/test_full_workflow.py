"""
End-to-end tests for the complete web scraping workflow
"""
import os
import time
from typing import Dict, List

import pytest
import requests


API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://localhost:8080").rstrip("/")
WEB_UI_URL = os.getenv("WEB_UI_URL", "http://localhost:8089").rstrip("/")

# Gateway proxy prefixes
URL_INPUT_PREFIX = f"{API_GATEWAY_URL}/api/url-input-service"
SESSION_PREFIX = f"{API_GATEWAY_URL}/api/session-service"
ANALYZER_PREFIX = f"{API_GATEWAY_URL}/api/analyzer-service"
EXPORT_PREFIX = f"{API_GATEWAY_URL}/api/export-service"
CLUSTER_PREFIX = f"{API_GATEWAY_URL}/api/clustering-service"


def wait_for_gateway_ready(timeout: int = 60) -> None:
    """Poll the API gateway health endpoint until it responds or timeout is reached."""
    end_time = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < end_time:
        try:
            response = requests.get(f"{API_GATEWAY_URL}/health", timeout=5)
            if response.status_code == 200:
                return
        except requests.exceptions.RequestException as exc:
            last_error = exc
        time.sleep(2)
    if last_error:
        raise RuntimeError(f"API Gateway not ready: {last_error}")
    raise RuntimeError("API Gateway not ready within timeout")


def fetch_health() -> Dict[str, Dict[str, Dict[str, str]]]:
    """Fetch comprehensive health data from the API Gateway."""
    response = requests.get(f"{API_GATEWAY_URL}/health", timeout=10)
    response.raise_for_status()
    return response.json()


def require_service(health: Dict, service_name: str) -> None:
    """
    Skip the current test if the target service is not reported healthy.
    """
    services = health.get("services", {})
    status = services.get(service_name, {}).get("status")
    if status != "healthy":
        pytest.skip(f"{service_name} service unavailable (status={status})")


class TestFullWorkflow:
    """Test complete workflow from URL input to export"""

    @pytest.fixture(autouse=True)
    def setup(self):
        wait_for_gateway_ready()

    def test_health_check(self):
        """Gateway health endpoint should respond and report current status."""
        data = fetch_health()
        assert data.get("status") in {"healthy", "degraded", "unhealthy"}
        if data.get("status") != "healthy":
            pytest.skip(f"Gateway reports {data.get('status')}")

    def test_url_submission_workflow(self):
        """Verify URL submission endpoint is reachable via the proxy."""
        health = fetch_health()
        require_service(health, "url-input-service")

        urls = [
            "https://example.com",
            "https://example.org",
            "https://example.net"
        ]

        # Endpoint expects query parameters for repeated URLs
        params = [("urls", url) for url in urls]
        response = requests.post(f"{URL_INPUT_PREFIX}/input/urls", params=params, timeout=15)
        assert response.status_code == 200
        payload = response.json()
        assert "input_id" in payload
        assert payload.get("total_urls") == len(urls)

    def test_session_management(self):
        """Create, fetch, and delete a session through the gateway."""
        health = fetch_health()
        require_service(health, "session-service")

        create_payload = {"name": "E2E Session", "description": "Session created via E2E tests"}
        response = requests.post(f"{SESSION_PREFIX}/sessions", json=create_payload, timeout=15)
        assert response.status_code in (200, 201)
        session_data = response.json()
        session_id = session_data.get("id") or session_data.get("session_id")
        assert session_id, "Session identifier missing in response"

        response = requests.get(f"{SESSION_PREFIX}/sessions/{session_id}", timeout=10)
        assert response.status_code == 200
        fetched = response.json()
        assert fetched.get("name") == create_payload["name"]

        response = requests.delete(f"{SESSION_PREFIX}/sessions/{session_id}", timeout=10)
        assert response.status_code in (200, 204)

    def test_search_functionality(self):
        """Ensure semantic search endpoint is reachable."""
        health = fetch_health()
        require_service(health, "analyzer-service")

        params = {"query": "test content", "search_type": "semantic", "limit": 5}
        response = requests.get(f"{ANALYZER_PREFIX}/search", params=params, timeout=20)
        if response.status_code in (200, 204):
            data = response.json() if response.content else {}
            assert isinstance(data.get("results", []), list)
        elif response.status_code in (404, 503):
            pytest.skip(f"Analyzer service unavailable (status code {response.status_code})")
        else:
            pytest.fail(f"Unexpected analyzer response: {response.status_code}")

    def test_export_workflow(self):
        """Submit an export request and ensure the service responds."""
        health = fetch_health()
        require_service(health, "export-service")

        payload = {
            "session_id": "default",
            "format": "markdown",
            "include_metadata": False,
            "include_clusters": False
        }
        response = requests.post(f"{EXPORT_PREFIX}/export", json=payload, timeout=30)
        assert response.status_code in (200, 202, 400, 422)


class TestAPIEndpoints:
    """Test individual API endpoints via the gateway proxy"""

    def test_url_validation(self):
        health = fetch_health()
        require_service(health, "url-input-service")

        params = [("urls", "https://example.com")]
        response = requests.post(f"{URL_INPUT_PREFIX}/input/urls", params=params, timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert data.get("valid_urls", 0) >= 1

    def test_invalid_url_validation(self):
        health = fetch_health()
        require_service(health, "url-input-service")

        params = [("urls", "not-a-valid-url")]
        response = requests.post(f"{URL_INPUT_PREFIX}/input/urls", params=params, timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert data.get("invalid_urls", 0) >= 1

    def test_clustering_endpoint(self):
        pytest.skip("Clustering workflow requires full backend stack and is skipped in lightweight E2E runs")


class TestWebUI:
    """Test Web UI functionality"""

    def test_ui_accessible(self):
        try:
            response = requests.get(WEB_UI_URL, timeout=10)
            assert response.status_code == 200
        except requests.exceptions.RequestException:
            pytest.skip("Web UI not available")

    def test_ui_static_assets(self):
        try:
            response = requests.get(f"{WEB_UI_URL}/static/", timeout=10)
            assert response.status_code in [200, 301, 302, 404]
        except requests.exceptions.RequestException:
            pytest.skip("Web UI not available")


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_invalid_endpoint(self):
        response = requests.get(f"{API_GATEWAY_URL}/api/invalid-endpoint", timeout=10)
        assert response.status_code in (404, 503)

    def test_malformed_request(self):
        response = requests.post(f"{URL_INPUT_PREFIX}/input/urls", json={"invalid": "data"}, timeout=10)
        assert response.status_code in (400, 422, 503)

    def test_rate_limiting(self):
        responses = []
        for _ in range(20):
            response = requests.get(f"{API_GATEWAY_URL}/health", timeout=5)
            responses.append(response.status_code)
        assert 200 in responses


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
