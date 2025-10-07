"""
Tests for Visualization Service
"""

import pytest
from fastapi.testclient import TestClient
from main import app, identify_bottlenecks, PipelineMetrics, PipelineStage

client = TestClient(app)

def test_root_endpoint():
    """Test root endpoint returns service information"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Visualization Service"
    assert data["version"] == "1.0.0"
    assert "endpoints" in data

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert data["service"] == "visualization"

def test_architecture_diagram():
    """Test architecture diagram endpoint"""
    response = client.get("/architecture/diagram")
    assert response.status_code == 200
    data = response.json()
    assert "services" in data
    assert "connections" in data
    assert "data_stores" in data
    assert len(data["services"]) > 0
    assert len(data["connections"]) > 0

def test_pipeline_status():
    """Test pipeline status endpoint"""
    response = client.get("/pipeline/status")
    assert response.status_code == 200
    data = response.json()
    assert "timestamp" in data
    assert "stages" in data
    assert "overall_health" in data
    assert "bottlenecks" in data
    assert len(data["stages"]) == 7  # All pipeline stages

def test_mermaid_diagram_architecture():
    """Test Mermaid architecture diagram generation"""
    response = client.get("/diagrams/mermaid/architecture")
    assert response.status_code == 200
    data = response.json()
    assert data["diagram_type"] == "architecture"
    assert "mermaid_code" in data
    assert "graph TB" in data["mermaid_code"]
    assert "API Gateway" in data["mermaid_code"]

def test_mermaid_diagram_pipeline():
    """Test Mermaid pipeline diagram generation"""
    response = client.get("/diagrams/mermaid/pipeline")
    assert response.status_code == 200
    data = response.json()
    assert data["diagram_type"] == "pipeline"
    assert "flowchart TD" in data["mermaid_code"]
    assert "URL Input" in data["mermaid_code"]

def test_mermaid_diagram_services():
    """Test Mermaid services diagram generation"""
    response = client.get("/diagrams/mermaid/services")
    assert response.status_code == 200
    data = response.json()
    assert data["diagram_type"] == "services"
    assert "sequenceDiagram" in data["mermaid_code"]

def test_mermaid_diagram_dataflow():
    """Test Mermaid data flow diagram generation"""
    response = client.get("/diagrams/mermaid/data-flow")
    assert response.status_code == 200
    data = response.json()
    assert data["diagram_type"] == "data-flow"
    assert "graph LR" in data["mermaid_code"]

def test_mermaid_diagram_deployment():
    """Test Mermaid deployment diagram generation"""
    response = client.get("/diagrams/mermaid/deployment")
    assert response.status_code == 200
    data = response.json()
    assert data["diagram_type"] == "deployment"
    assert "Docker Host" in data["mermaid_code"]

def test_mermaid_diagram_invalid_type():
    """Test invalid diagram type returns 404"""
    response = client.get("/diagrams/mermaid/invalid-type")
    assert response.status_code == 404

def test_dashboard_html():
    """Test dashboard HTML endpoint"""
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "System Architecture Dashboard" in response.text
    assert "mermaid" in response.text

def test_capacity_planning():
    """Test capacity planning endpoint"""
    response = client.get("/capacity/planning")
    assert response.status_code == 200
    data = response.json()
    assert "current_capacity" in data
    assert "recommendations" in data
    assert "scaling_thresholds" in data

def test_bottleneck_identification_high_queue():
    """Test bottleneck identification for high queue size"""
    stages = [
        PipelineMetrics(
            stage=PipelineStage.SCRAPING,
            total_processed=100,
            success_count=95,
            failure_count=5,
            avg_processing_time_ms=2000.0,
            current_queue_size=25  # High queue size
        )
    ]
    
    bottlenecks = identify_bottlenecks(stages)
    assert len(bottlenecks) > 0
    assert any(b["issue"] == "high_queue_size" for b in bottlenecks)

def test_bottleneck_identification_high_failure_rate():
    """Test bottleneck identification for high failure rate"""
    stages = [
        PipelineMetrics(
            stage=PipelineStage.AUTHENTICATION,
            total_processed=100,
            success_count=85,
            failure_count=15,  # 15% failure rate
            avg_processing_time_ms=2000.0,
            current_queue_size=5
        )
    ]
    
    bottlenecks = identify_bottlenecks(stages)
    assert len(bottlenecks) > 0
    assert any(b["issue"] == "high_failure_rate" for b in bottlenecks)

def test_bottleneck_identification_slow_processing():
    """Test bottleneck identification for slow processing"""
    stages = [
        PipelineMetrics(
            stage=PipelineStage.CLUSTERING,
            total_processed=100,
            success_count=100,
            failure_count=0,
            avg_processing_time_ms=6000.0,  # Slow processing
            current_queue_size=2
        )
    ]
    
    bottlenecks = identify_bottlenecks(stages)
    assert len(bottlenecks) > 0
    assert any(b["issue"] == "slow_processing" for b in bottlenecks)

def test_bottleneck_identification_no_issues():
    """Test bottleneck identification with healthy metrics"""
    stages = [
        PipelineMetrics(
            stage=PipelineStage.INPUT,
            total_processed=100,
            success_count=99,
            failure_count=1,
            avg_processing_time_ms=100.0,
            current_queue_size=2
        )
    ]
    
    bottlenecks = identify_bottlenecks(stages)
    assert len(bottlenecks) == 0

def test_service_health_structure():
    """Test service health endpoint returns proper structure"""
    response = client.get("/services/health")
    assert response.status_code == 200
    services = response.json()
    assert isinstance(services, list)
    
    if len(services) > 0:
        service = services[0]
        assert "service_name" in service
        assert "status" in service
        assert "last_check" in service

def test_architecture_diagram_services_count():
    """Test architecture diagram includes all services"""
    response = client.get("/architecture/diagram")
    data = response.json()
    
    # Should include 10 microservices + 2 data stores
    assert len(data["services"]) >= 12

def test_architecture_diagram_connections():
    """Test architecture diagram includes service connections"""
    response = client.get("/architecture/diagram")
    data = response.json()
    
    # Verify key connections exist
    connections = data["connections"]
    connection_pairs = [(c["from"], c["to"]) for c in connections]
    
    assert ("web-ui", "api-gateway") in connection_pairs
    assert ("api-gateway", "url-input") in connection_pairs
    assert ("scraper", "qdrant") in connection_pairs
    assert ("analyzer", "ollama") in connection_pairs

def test_pipeline_stages_completeness():
    """Test pipeline status includes all stages"""
    response = client.get("/pipeline/status")
    data = response.json()
    
    stages = [s["stage"] for s in data["stages"]]
    expected_stages = [
        "input", "validation", "authentication", 
        "scraping", "analysis", "clustering", "export"
    ]
    
    for expected in expected_stages:
        assert expected in stages

def test_mermaid_diagrams_valid_syntax():
    """Test all Mermaid diagrams have valid syntax markers"""
    diagram_types = ["architecture", "pipeline", "services", "data-flow", "deployment"]
    
    for diagram_type in diagram_types:
        response = client.get(f"/diagrams/mermaid/{diagram_type}")
        assert response.status_code == 200
        data = response.json()
        mermaid_code = data["mermaid_code"]
        
        # Check for valid Mermaid diagram type declarations
        valid_declarations = ["graph", "flowchart", "sequenceDiagram"]
        assert any(decl in mermaid_code for decl in valid_declarations)

def test_capacity_planning_structure():
    """Test capacity planning returns proper structure"""
    response = client.get("/capacity/planning")
    data = response.json()
    
    assert "current_capacity" in data
    assert "total_services" in data["current_capacity"]
    assert "cpu_usage_percent" in data["current_capacity"]
    
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)
    
    assert "scaling_thresholds" in data
    assert "cpu_threshold_percent" in data["scaling_thresholds"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
