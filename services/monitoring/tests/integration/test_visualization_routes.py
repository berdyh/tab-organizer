"""Integration tests for the monitoring visualization endpoints."""

from fastapi.testclient import TestClient

from services.monitoring.main import app

client = TestClient(app)


def test_visualization_health_endpoint():
    response = client.get("/visualization/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["service"] == "monitoring-visualization"


def test_architecture_diagram_contains_expected_services():
    response = client.get("/visualization/architecture/diagram")
    assert response.status_code == 200
    payload = response.json()

    service_ids = {service["id"] for service in payload["services"]}
    expected = {"api-gateway", "url-input-service", "analyzer-service", "qdrant", "ollama"}
    assert expected.issubset(service_ids)


def test_mermaid_diagram_variants_available():
    for diagram_type in ("architecture", "pipeline", "services"):
        response = client.get(f"/visualization/diagrams/mermaid/{diagram_type}")
        assert response.status_code == 200
        data = response.json()
        assert "graph" in data["diagram"] or "flowchart" in data["diagram"]
