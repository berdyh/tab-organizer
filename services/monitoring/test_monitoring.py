"""
Comprehensive tests for the monitoring service.
Tests monitoring functionality, metrics collection, alerting, and performance tracking.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

from main import app
from config import MonitoringSettings
from metrics_collector import MetricsCollector
from health_monitor import HealthMonitor
from alert_manager import AlertManager, AlertSeverity
from performance_tracker import PerformanceTracker
from distributed_tracer import DistributedTracer


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


@pytest.fixture
def mock_settings():
    """Mock settings fixture."""
    settings = MonitoringSettings()
    settings.services = {
        "test-service": "http://test-service:8080",
        "another-service": "http://another-service:8081"
    }
    return settings


class TestMonitoringService:
    """Test monitoring service endpoints."""
    
    def test_health_endpoint(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code in [200, 503]  # May be 503 if components not initialized
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
    
    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "Web Scraping Tool - Monitoring Service"
        assert "endpoints" in data
        assert "features" in data


class TestMetricsCollector:
    """Test metrics collector functionality."""
    
    @pytest.fixture
    def metrics_collector(self, mock_settings):
        """Metrics collector fixture."""
        return MetricsCollector(mock_settings)
    
    @pytest.mark.asyncio
    async def test_system_metrics_collection(self, metrics_collector):
        """Test system metrics collection."""
        system_metrics = await metrics_collector._collect_system_metrics()
        
        assert "cpu" in system_metrics
        assert "memory" in system_metrics
        assert "disk" in system_metrics
        assert "network" in system_metrics
        
        # Check CPU metrics
        cpu_metrics = system_metrics["cpu"]
        assert "percent" in cpu_metrics
        assert "count_physical" in cpu_metrics
        assert "count_logical" in cpu_metrics
        
        # Check memory metrics
        memory_metrics = system_metrics["memory"]
        assert "total_bytes" in memory_metrics
        assert "available_bytes" in memory_metrics
        assert "percent" in memory_metrics
    
    @pytest.mark.asyncio
    @patch('docker.from_env')
    async def test_container_metrics_collection(self, mock_docker, metrics_collector):
        """Test container metrics collection."""
        # Mock Docker client and containers
        mock_container = Mock()
        mock_container.name = "test-container"
        mock_container.id = "abc123"
        mock_container.status = "running"
        mock_container.image.tags = ["test:latest"]
        mock_container.attrs = {
            'Created': '2023-01-01T00:00:00Z',
            'State': {'StartedAt': '2023-01-01T00:00:00Z'}
        }
        mock_container.stats.return_value = {
            'cpu_stats': {
                'cpu_usage': {'total_usage': 1000000},
                'system_cpu_usage': 10000000,
                'online_cpus': 2
            },
            'precpu_stats': {
                'cpu_usage': {'total_usage': 900000},
                'system_cpu_usage': 9000000
            },
            'memory_stats': {
                'usage': 1024 * 1024 * 100,  # 100MB
                'limit': 1024 * 1024 * 1024  # 1GB
            },
            'networks': {
                'eth0': {
                    'rx_bytes': 1024,
                    'tx_bytes': 2048
                }
            },
            'blkio_stats': {
                'io_service_bytes_recursive': [
                    {'op': 'Read', 'value': 1024},
                    {'op': 'Write', 'value': 2048}
                ]
            }
        }
        
        mock_docker_client = Mock()
        mock_docker_client.containers.list.return_value = [mock_container]
        mock_docker.return_value = mock_docker_client
        
        metrics_collector.docker_client = mock_docker_client
        
        container_metrics = await metrics_collector._collect_container_metrics()
        
        assert "test-container" in container_metrics
        container_data = container_metrics["test-container"]
        
        assert container_data["name"] == "test-container"
        assert container_data["status"] == "running"
        assert "cpu_percent" in container_data
        assert "memory_usage" in container_data
        assert "memory_percent" in container_data
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_service_metrics_collection(self, mock_client, metrics_collector):
        """Test service metrics collection."""
        # Mock HTTP responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = """
# HELP test_metric A test metric
# TYPE test_metric counter
test_metric 42
        """
        
        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        
        service_metrics = await metrics_collector._collect_service_metrics()
        
        assert "test-service" in service_metrics
        assert "another-service" in service_metrics
        
        # Check service metrics structure
        for service_name, metrics in service_metrics.items():
            assert "status" in metrics
            assert "last_check" in metrics


class TestHealthMonitor:
    """Test health monitor functionality."""
    
    @pytest.fixture
    def health_monitor(self, mock_settings):
        """Health monitor fixture."""
        return HealthMonitor(mock_settings)
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_service_health_check(self, mock_client, health_monitor):
        """Test service health checking."""
        # Mock successful health response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}
        
        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        
        service_health = await health_monitor._check_service_health()
        
        assert "test-service" in service_health
        assert "another-service" in service_health
        
        # Check health data structure
        for service_name, health_data in service_health.items():
            assert "healthy" in health_data
            assert "status" in health_data
            assert "last_check" in health_data
    
    @pytest.mark.asyncio
    @patch('docker.from_env')
    async def test_container_health_check(self, mock_docker, health_monitor):
        """Test container health checking."""
        # Mock Docker container
        mock_container = Mock()
        mock_container.name = "test-container"
        mock_container.status = "running"
        mock_container.image.tags = ["test:latest"]
        mock_container.attrs = {
            'Created': '2023-01-01T00:00:00Z',
            'State': {
                'StartedAt': '2023-01-01T00:00:00Z',
                'Health': {'Status': 'healthy'}
            }
        }
        mock_container.stats.return_value = {
            'cpu_stats': {'cpu_usage': {'total_usage': 1000000}, 'system_cpu_usage': 10000000, 'online_cpus': 2},
            'precpu_stats': {'cpu_usage': {'total_usage': 900000}, 'system_cpu_usage': 9000000},
            'memory_stats': {'usage': 1024 * 1024 * 100}
        }
        
        mock_docker_client = Mock()
        mock_docker_client.containers.list.return_value = [mock_container]
        mock_docker.return_value = mock_docker_client
        
        health_monitor.docker_client = mock_docker_client
        
        container_health = await health_monitor._check_container_health()
        
        assert "test-container" in container_health
        container_data = container_health["test-container"]
        
        assert container_data["status"] == "running"
        assert "healthy" in container_data
        assert "docker_health" in container_data


class TestAlertManager:
    """Test alert manager functionality."""
    
    @pytest.fixture
    def alert_manager(self, mock_settings):
        """Alert manager fixture."""
        return AlertManager(mock_settings)
    
    @pytest.mark.asyncio
    async def test_create_alert(self, alert_manager):
        """Test alert creation."""
        alert_id = await alert_manager.create_alert(
            alert_type="test_alert",
            severity=AlertSeverity.HIGH,
            message="Test alert message",
            service="test-service"
        )
        
        assert alert_id in alert_manager.alerts
        alert = alert_manager.alerts[alert_id]
        
        assert alert.alert_type == "test_alert"
        assert alert.severity == AlertSeverity.HIGH
        assert alert.message == "Test alert message"
        assert alert.service == "test-service"
    
    @pytest.mark.asyncio
    async def test_acknowledge_alert(self, alert_manager):
        """Test alert acknowledgment."""
        # Create an alert
        alert_id = await alert_manager.create_alert(
            alert_type="test_alert",
            severity=AlertSeverity.MEDIUM,
            message="Test alert"
        )
        
        # Acknowledge the alert
        success = await alert_manager.acknowledge_alert(alert_id)
        assert success
        
        alert = alert_manager.alerts[alert_id]
        assert alert.status.value == "acknowledged"
        assert alert.acknowledged_at is not None
    
    @pytest.mark.asyncio
    async def test_resolve_alert(self, alert_manager):
        """Test alert resolution."""
        # Create an alert
        alert_id = await alert_manager.create_alert(
            alert_type="test_alert",
            severity=AlertSeverity.LOW,
            message="Test alert"
        )
        
        # Resolve the alert
        success = await alert_manager.resolve_alert(alert_id)
        assert success
        
        alert = alert_manager.alerts[alert_id]
        assert alert.status.value == "resolved"
        assert alert.resolved_at is not None
    
    @pytest.mark.asyncio
    async def test_get_alert_statistics(self, alert_manager):
        """Test alert statistics."""
        # Create some test alerts
        await alert_manager.create_alert("test1", AlertSeverity.HIGH, "Test 1")
        await alert_manager.create_alert("test2", AlertSeverity.MEDIUM, "Test 2")
        await alert_manager.create_alert("test3", AlertSeverity.LOW, "Test 3")
        
        stats = await alert_manager.get_alert_statistics()
        
        assert stats["total_alerts"] == 3
        assert stats["active_alerts"] == 3
        assert "severity_breakdown" in stats
        assert "service_breakdown" in stats


class TestPerformanceTracker:
    """Test performance tracker functionality."""
    
    @pytest.fixture
    def performance_tracker(self, mock_settings):
        """Performance tracker fixture."""
        return PerformanceTracker(mock_settings)
    
    @pytest.mark.asyncio
    async def test_system_performance_collection(self, performance_tracker):
        """Test system performance collection."""
        system_perf = await performance_tracker._collect_system_performance()
        
        assert "cpu" in system_perf
        assert "memory" in system_perf
        assert "disk" in system_perf
        assert "overall_score" in system_perf
        assert "performance_grade" in system_perf
        
        # Check scores are within valid range
        assert 0 <= system_perf["overall_score"] <= 100
        assert system_perf["performance_grade"] in ["A", "B", "C", "D", "F"]
    
    @pytest.mark.asyncio
    async def test_system_benchmarks(self, performance_tracker):
        """Test system benchmarks."""
        benchmarks = await performance_tracker._run_system_benchmarks()
        
        assert "cpu_benchmark_ms" in benchmarks
        assert "memory_benchmark_ms" in benchmarks
        assert "disk_benchmark_ms" in benchmarks
        assert "overall_score" in benchmarks
        assert "grade" in benchmarks
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_service_benchmarks(self, mock_client, performance_tracker):
        """Test service benchmarks."""
        # Mock HTTP responses
        mock_response = Mock()
        mock_response.status_code = 200
        
        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        
        benchmarks = await performance_tracker._run_service_benchmarks()
        
        assert "test-service" in benchmarks
        assert "another-service" in benchmarks
        
        for service_name, benchmark_data in benchmarks.items():
            assert "performance_score" in benchmark_data
            assert "grade" in benchmark_data
    
    def test_performance_grade_calculation(self, performance_tracker):
        """Test performance grade calculation."""
        assert performance_tracker._calculate_performance_grade(95) == "A"
        assert performance_tracker._calculate_performance_grade(85) == "B"
        assert performance_tracker._calculate_performance_grade(75) == "C"
        assert performance_tracker._calculate_performance_grade(65) == "D"
        assert performance_tracker._calculate_performance_grade(45) == "F"


class TestDistributedTracer:
    """Test distributed tracer functionality."""
    
    @pytest.fixture
    def tracer(self, mock_settings):
        """Distributed tracer fixture."""
        return DistributedTracer(mock_settings)
    
    def test_create_trace(self, tracer):
        """Test trace creation."""
        trace_id = tracer.create_trace("test_operation")
        
        assert trace_id in tracer.traces
        trace = tracer.traces[trace_id]
        
        assert trace.trace_id == trace_id
        assert len(trace.spans) == 1  # Root span
        assert trace.root_span_id is not None
    
    def test_start_and_finish_span(self, tracer):
        """Test span lifecycle."""
        trace_id = tracer.create_trace("test_operation")
        
        # Start a child span
        span = tracer.start_span(
            trace_id=trace_id,
            operation_name="child_operation",
            service_name="test-service",
            parent_span_id=tracer.traces[trace_id].root_span_id
        )
        
        assert span.span_id in tracer.active_spans
        assert span.status == "active"
        
        # Finish the span
        tracer.finish_span(span.span_id, status="completed")
        
        assert span.span_id not in tracer.active_spans
        assert span.status == "completed"
        assert span.duration is not None
    
    def test_span_logging_and_tagging(self, tracer):
        """Test span logging and tagging."""
        trace_id = tracer.create_trace("test_operation")
        span = tracer.start_span(trace_id, "test_span", "test-service")
        
        # Add log and tag
        tracer.add_span_log(span.span_id, "Test log message", key="value")
        tracer.add_span_tag(span.span_id, "test_tag", "test_value")
        
        assert len(span.logs) == 1
        assert span.logs[0]["message"] == "Test log message"
        assert span.tags["test_tag"] == "test_value"
    
    @pytest.mark.asyncio
    async def test_get_traces(self, tracer):
        """Test trace retrieval."""
        # Create some test traces
        trace_id1 = tracer.create_trace("operation1")
        trace_id2 = tracer.create_trace("operation2")
        
        traces = await tracer.get_traces()
        
        assert len(traces) == 2
        trace_ids = [t["trace_id"] for t in traces]
        assert trace_id1 in trace_ids
        assert trace_id2 in trace_ids
    
    @pytest.mark.asyncio
    async def test_search_traces(self, tracer):
        """Test trace search functionality."""
        # Create traces with different operations
        trace_id1 = tracer.create_trace("user_login")
        trace_id2 = tracer.create_trace("data_processing")
        
        # Search for login traces
        results = await tracer.search_traces("login")
        
        assert len(results) == 1
        assert results[0]["trace_id"] == trace_id1


@pytest.mark.asyncio
async def test_monitoring_integration():
    """Test integration between monitoring components."""
    settings = MonitoringSettings()
    
    # Initialize components
    metrics_collector = MetricsCollector(settings)
    health_monitor = HealthMonitor(settings)
    alert_manager = AlertManager(settings)
    
    # Test that components can work together
    # This is a basic integration test
    assert metrics_collector.settings == settings
    assert health_monitor.settings == settings
    assert alert_manager.settings == settings


if __name__ == "__main__":
    pytest.main([__file__, "-v"])