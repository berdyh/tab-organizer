"""Simplified integration tests for API Gateway service."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from api_gateway.app import create_app
from api_gateway.core import AuthMiddleware, RateLimiter, Settings
from api_gateway.state import GatewayState


class DummyServiceRegistry:
    """Minimal service registry for tests."""

    def __init__(self):
        self.services = {
            "url-input-service": {
                "url": "http://url-input-service:8081",
                "healthy": True,
                "type": "internal",
            }
        }

    def get_all_services(self):
        return self.services

    def get_service(self, name: str):
        return self.services.get(name)


class DummyHealthChecker:
    """Minimal health checker for tests."""

    def __init__(self, services):
        self.services = services

    async def get_comprehensive_health(self):
        return {
            "status": "healthy",
            "uptime": 1.0,
            "services": self.services,
        }

    async def check_all_services(self):
        return self.services


@pytest.fixture
def client():
    """Provide a configured TestClient with simplified state."""
    settings = Settings(enable_background_tasks=False)
    app = create_app(settings)

    with TestClient(app) as test_client:
        state: GatewayState = app.state.gateway_state
        state.service_registry = DummyServiceRegistry()
        state.health_checker = DummyHealthChecker(state.service_registry.services)
        state.rate_limiter = RateLimiter()
        state.auth_middleware = AuthMiddleware()
        yield test_client


class TestAPIGatewayBasic:
    """Test basic API Gateway functionality."""

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Web Scraping & Clustering Tool - API Gateway"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"
        assert "endpoints" in data
        assert "features" in data

    def test_health_simple(self, client):
        response = client.get("/health/simple")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data

    def test_metrics_endpoint(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "api_gateway_requests_total" in response.text


class TestAuthentication:
    """Test authentication functionality."""

    def test_login_success(self, client):
        response = client.post("/auth/login", json={"username": "admin", "password": "admin"})
        assert response.status_code == 200
        data = response.json()
        assert "token" in data
        assert "expires_in" in data
        assert "permissions" in data
        assert "admin" in data["permissions"]

    def test_login_failure(self, client):
        response = client.post("/auth/login", json={"username": "invalid", "password": "invalid"})
        assert response.status_code == 401

    def test_authenticated_endpoint(self, client):
        login_response = client.post("/auth/login", json={"username": "admin", "password": "admin"})
        token = login_response.json()["token"]
        response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "token"
        assert data["user_id"] == "admin"


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_info(self, client):
        response = client.get("/rate-limit/info")
        assert response.status_code == 200
        data = response.json()
        assert "client_id" in data
        assert "client_type" in data
        assert "limits" in data
        assert "current_usage" in data

    def test_rate_limit_headers(self, client):
        response = client.get("/rate-limit/info")
        assert response.status_code == 200
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers


class TestServiceDiscovery:
    """Test service discovery functionality."""

    def test_list_services_with_api_key(self, client):
        response = client.get("/services", headers={"X-API-Key": "internal-service-key"})
        assert response.status_code == 200
        data = response.json()
        assert "services" in data
        assert "timestamp" in data

    def test_list_services_without_auth(self, client):
        response = client.get("/services")
        assert response.status_code == 401


class TestModelManagement:
    """Test model management functionality."""

    def test_model_config(self, client):
        response = client.get("/models/config")
        assert response.status_code == 200
        data = response.json()
        assert "llm_model" in data
        assert "embedding_model" in data
        assert "ollama_url" in data
