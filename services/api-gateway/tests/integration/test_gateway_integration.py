"""Integration tests for the refactored API Gateway service."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from api_gateway.app import create_app
from api_gateway.core import AuthMiddleware, RateLimiter, Settings
from api_gateway.state import GatewayState


class DummyServiceRegistry:
    """Lightweight in-memory registry suitable for tests."""

    def __init__(self):
        self.services = {}

    def set_service(self, name: str, config: dict) -> None:
        self.services[name] = config

    def get_all_services(self):
        return self.services

    def get_service(self, name: str):
        return self.services.get(name)


class DummyHealthChecker:
    """Health checker that reports healthy status for registered services."""

    def __init__(self, registry: DummyServiceRegistry):
        self.registry = registry

    async def get_comprehensive_health(self):
        services = {
            name: {"status": "healthy", "response_time": 0.01}
            for name in self.registry.services
        }
        return {
            "status": "healthy",
            "uptime": 1.0,
            "services": services,
        }

    async def check_all_services(self):
        return {
            name: {"status": "healthy", "response_time": 0.01}
            for name in self.registry.services
        }


@pytest.fixture
def test_app():
    settings = Settings(enable_background_tasks=False)
    return create_app(settings)


@pytest.fixture
def client(test_app):
    with TestClient(test_app) as test_client:
        state: GatewayState = test_app.state.gateway_state
        registry = DummyServiceRegistry()
        registry.set_service(
            "test-service",
            {"url": "http://test-service:8080", "healthy": True, "type": "internal"},
        )
        state.service_registry = registry
        state.health_checker = DummyHealthChecker(registry)
        state.rate_limiter = RateLimiter()
        state.auth_middleware = AuthMiddleware()
        yield test_client


@pytest.fixture
def state(client):
    return client.app.state.gateway_state


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Web Scraping & Clustering Tool - API Gateway"
    assert data["status"] == "running"


def test_login_logout_flow(client):
    login_response = client.post("/auth/login", json={"username": "admin", "password": "admin"})
    assert login_response.status_code == 200
    token = login_response.json()["token"]

    me_response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_response.status_code == 200
    assert me_response.json()["user_id"] == "admin"

    logout_response = client.post("/auth/logout", headers={"Authorization": f"Bearer {token}"})
    assert logout_response.status_code == 200

    after_logout = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert after_logout.status_code == 401


def test_unauthorized_admin_access(client):
    response = client.get("/admin/auth/stats")
    assert response.status_code == 403


def test_rate_limiting_enforcement(client):
    statuses = []
    for _ in range(80):
        try:
            response = client.get("/rate-limit/info")
            statuses.append(response.status_code)
        except HTTPException as exc:
            statuses.append(exc.status_code)
        except httpx.HTTPStatusError as exc:
            statuses.append(exc.response.status_code)
    assert 429 in statuses


def test_service_listing_requires_api_key(client):
    assert client.get("/services").status_code == 401


def test_service_listing_success(client):
    response = client.get("/services", headers={"X-API-Key": "internal-service-key"})
    assert response.status_code == 200
    data = response.json()
    assert "test-service" in data["services"]


def test_service_info_not_found(client, state):
    state.service_registry.set_service("another-service", {"url": "http://example", "healthy": True})
    response = client.get("/services/missing", headers={"X-API-Key": "internal-service-key"})
    assert response.status_code == 404


@patch("api_gateway.routes.proxy.httpx.AsyncClient")
def test_proxy_success(mock_client_class, client, state):
    state.service_registry.set_service("proxy-service", {"url": "http://proxy:8080", "healthy": True})

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"ok": True}
    mock_response.headers = {"content-type": "application/json"}

    mock_client = AsyncMock()
    mock_client.request.return_value = mock_response
    mock_client_class.return_value.__aenter__.return_value = mock_client

    response = client.get("/api/proxy-service/test")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_proxy_unhealthy_service(client, state):
    state.service_registry.set_service(
        "bad-service", {"url": "http://bad:8080", "healthy": False}
    )
    response = client.get("/api/bad-service/test")
    assert response.status_code == 503


def test_comprehensive_health_uses_checker(client, state):
    mock_health = AsyncMock(
        return_value={
        "status": "healthy",
        "uptime": 42,
        "services": {"test-service": {"status": "healthy"}},
        }
    )
    state.health_checker.get_comprehensive_health = mock_health
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    mock_health.assert_awaited()


@patch("api_gateway.routes.models.httpx.AsyncClient")
def test_model_listing(mock_client_class, client, state):
    state.service_registry.set_service("ollama", {"url": "http://ollama:11434", "healthy": True})

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"models": [{"name": "llama"}]}

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    mock_client_class.return_value.__aenter__.return_value = mock_client

    response = client.get("/models")
    assert response.status_code == 200
    assert response.json()["installed_models"][0]["name"] == "llama"
