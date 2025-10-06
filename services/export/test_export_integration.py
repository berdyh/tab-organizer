"""Integration tests for the Export Service."""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from qdrant_client.http import models

from main import app, export_engine, export_jobs, ExportFormat, ExportStatus


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_session_data():
    """Sample session data for testing."""
    return {
        'session_id': 'test_session_123',
        'session_name': 'Test Session',
        'items': [
            {
                'id': '1',
                'title': 'Test Article 1',
                'url': 'https://example.com/article1',
                'domain': 'example.com',
                'content': 'This is test content for article 1. It contains information about testing.',
                'cluster_id': 0,
                'cluster_label': 'Technology Articles',
                'created_at': '2024-01-01T00:00:00Z',
                'metadata': {'author': 'Test Author'}
            },
            {
                'id': '2',
                'title': 'Test Article 2',
                'url': 'https://example.com/article2',
                'domain': 'example.com',
                'content': 'This is test content for article 2. It discusses different topics.',
                'cluster_id': 1,
                'cluster_label': 'General Articles',
                'created_at': '2024-01-01T01:00:00Z',
                'metadata': {'author': 'Another Author'}
            }
        ],
        'clusters': [
            {
                'id': 0,
                'label': 'Technology Articles',
                'items': [
                    {
                        'id': '1',
                        'title': 'Test Article 1',
                        'url': 'https://example.com/article1',
                        'domain': 'example.com',
                        'content': 'This is test content for article 1.',
                        'cluster_id': 0,
                        'cluster_label': 'Technology Articles',
                        'created_at': '2024-01-01T00:00:00Z',
                        'metadata': {}
                    }
                ],
                'size': 1,
                'coherence_score': 0.85
            },
            {
                'id': 1,
                'label': 'General Articles',
                'items': [
                    {
                        'id': '2',
                        'title': 'Test Article 2',
                        'url': 'https://example.com/article2',
                        'domain': 'example.com',
                        'content': 'This is test content for article 2.',
                        'cluster_id': 1,
                        'cluster_label': 'General Articles',
                        'created_at': '2024-01-01T01:00:00Z',
                        'metadata': {}
                    }
                ],
                'size': 1,
                'coherence_score': 0.75
            }
        ],
        'total_items': 2,
        'total_clusters': 2,
        'export_date': '2024-01-01T12:00:00Z',
        'summary': 'Exported 2 items across 2 clusters'
    }


@pytest.fixture
def mock_qdrant_points():
    """Mock Qdrant points data."""
    return [
        MagicMock(
            id='1',
            payload={
                'title': 'Test Article 1',
                'url': 'https://example.com/article1',
                'domain': 'example.com',
                'content': 'This is test content for article 1. It contains information about testing.',
                'cluster_id': 0,
                'cluster_label': 'Technology Articles',
                'created_at': '2024-01-01T00:00:00Z',
                'coherence_score': 0.85,
                'metadata': {'author': 'Test Author'}
            }
        ),
        MagicMock(
            id='2',
            payload={
                'title': 'Test Article 2',
                'url': 'https://example.com/article2',
                'domain': 'example.com',
                'content': 'This is test content for article 2. It discusses different topics.',
                'cluster_id': 1,
                'cluster_label': 'General Articles',
                'created_at': '2024-01-01T01:00:00Z',
                'coherence_score': 0.75,
                'metadata': {'author': 'Another Author'}
            }
        )
    ]


class TestExportAPI:
    """Test export API endpoints."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "export"

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Export Service"
        assert "supported_formats" in data

    def test_get_supported_formats(self, client):
        """Test getting supported export formats."""
        response = client.get("/formats")
        assert response.status_code == 200
        formats = response.json()
        
        assert len(formats) > 0
        format_names = [f["format"] for f in formats]
        assert "json" in format_names
        assert "csv" in format_names
        assert "markdown" in format_names
        assert "word" in format_names

    @patch('main.qdrant_client')
    def test_create_export_job(self, mock_qdrant, client, mock_qdrant_points):
        """Test creating an export job."""
        # Mock Qdrant response
        mock_qdrant.scroll.return_value = (mock_qdrant_points, None)
        
        export_request = {
            "session_id": "test_session_123",
            "format": "json",
            "include_metadata": True,
            "include_clusters": True
        }
        
        response = client.post("/export", json=export_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        
        # Verify job was created
        job_id = data["job_id"]
        assert job_id in export_jobs

    def test_get_export_status_not_found(self, client):
        """Test getting status of non-existent export job."""
        response = client.get("/export/nonexistent/status")
        assert response.status_code == 404

    @patch('main.qdrant_client')
    def test_batch_export(self, mock_qdrant, client, mock_qdrant_points):
        """Test batch export functionality."""
        # Mock Qdrant response
        mock_qdrant.scroll.return_value = (mock_qdrant_points, None)
        
        batch_request = [
            {
                "session_id": "session1",
                "format": "json"
            },
            {
                "session_id": "session2",
                "format": "csv"
            }
        ]
        
        response = client.post("/export/batch", json=batch_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "job_ids" in data
        assert data["total_jobs"] == 2
        assert len(data["job_ids"]) == 2

    def test_list_export_jobs_empty(self, client):
        """Test listing export jobs when none exist."""
        # Clear existing jobs
        export_jobs.clear()
        
        response = client.get("/export/jobs")
        assert response.status_code == 200
        assert response.json() == []

    def test_create_template(self, client):
        """Test creating a custom template."""
        import time
        import uuid
        unique_name = f"test_template_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        
        template_data = {
            "name": unique_name,
            "format": "markdown",
            "template_content": "# {{ session_name }}\n\nTotal: {{ total_items }}"
        }
        
        response = client.post("/templates", json=template_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["template_name"] == unique_name

    def test_list_templates(self, client):
        """Test listing available templates."""
        response = client.get("/templates")
        assert response.status_code == 200
        templates = response.json()
        assert isinstance(templates, list)

    @patch('main.qdrant_client')
    def test_preview_export(self, mock_qdrant, client, mock_qdrant_points):
        """Test export preview functionality."""
        # Mock Qdrant response
        mock_qdrant.scroll.return_value = (mock_qdrant_points, None)
        
        response = client.post(
            "/export/preview",
            params={
                "session_id": "test_session",
                "format": "json",
                "limit": 2
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "preview" in data
        assert "sample_items" in data
        assert "sample_clusters" in data


class TestExportEngine:
    """Test export engine functionality."""

    @pytest.mark.asyncio
    @patch('main.qdrant_client')
    async def test_get_session_data(self, mock_qdrant, mock_qdrant_points):
        """Test retrieving session data from Qdrant."""
        # Mock Qdrant response
        mock_qdrant.scroll.return_value = (mock_qdrant_points, None)
        
        data = await export_engine.get_session_data("test_session")
        
        assert data["session_id"] == "test_session"
        assert data["total_items"] == 2
        assert data["total_clusters"] == 2
        assert len(data["items"]) == 2
        assert len(data["clusters"]) == 2

    @pytest.mark.asyncio
    async def test_export_to_json(self, sample_session_data):
        """Test JSON export functionality."""
        result = await export_engine.export_to_json(sample_session_data)
        
        # Verify it's valid JSON
        parsed = json.loads(result)
        assert parsed["session_id"] == "test_session_123"
        assert parsed["total_items"] == 2

    @pytest.mark.asyncio
    async def test_export_to_csv(self, sample_session_data):
        """Test CSV export functionality."""
        result = await export_engine.export_to_csv(sample_session_data)
        
        lines = result.strip().split('\n')
        assert len(lines) == 3  # Header + 2 data rows
        
        # Check header
        header = lines[0]
        assert "id" in header
        assert "title" in header
        assert "url" in header

    @pytest.mark.asyncio
    async def test_export_to_markdown(self, sample_session_data):
        """Test Markdown export functionality."""
        result = await export_engine.export_to_markdown(sample_session_data)
        
        assert "# Test Session - Export Report" in result
        assert "Total Items: 2" in result
        assert "Technology Articles" in result

    @pytest.mark.asyncio
    async def test_export_to_obsidian(self, sample_session_data):
        """Test Obsidian export functionality."""
        result = await export_engine.export_to_obsidian(sample_session_data)
        
        assert "---" in result  # YAML frontmatter
        assert "tags:" in result
        assert "[[" in result  # Obsidian links
        assert "#example_com" in result  # Tags

    @pytest.mark.asyncio
    async def test_export_to_word(self, sample_session_data):
        """Test Word document export functionality."""
        result = await export_engine.export_to_word(sample_session_data)
        
        # Verify it's a BytesIO object with content
        assert hasattr(result, 'read')
        assert hasattr(result, 'seek')
        
        # Check that it has content
        content = result.getvalue()
        assert len(content) > 0

    @pytest.mark.asyncio
    async def test_custom_template_json(self, sample_session_data):
        """Test JSON export with custom template."""
        custom_template = '{"custom": "{{ session_name }}", "count": {{ total_items }}}'
        
        result = await export_engine.export_to_json(sample_session_data, custom_template)
        parsed = json.loads(result)
        
        assert parsed["custom"] == "Test Session"
        assert parsed["count"] == 2

    @pytest.mark.asyncio
    async def test_custom_template_markdown(self, sample_session_data):
        """Test Markdown export with custom template."""
        custom_template = "# Custom Export\n\nSession: {{ session_name }}\nItems: {{ total_items }}"
        
        result = await export_engine.export_to_markdown(sample_session_data, custom_template)
        
        assert "# Custom Export" in result
        assert "Session: Test Session" in result
        assert "Items: 2" in result


class TestExportFilters:
    """Test export filtering functionality."""

    def test_apply_filters_cluster_ids(self, mock_qdrant_points):
        """Test filtering by cluster IDs."""
        filtered = export_engine._apply_filters(mock_qdrant_points, 
                                              type('Filter', (), {'cluster_ids': [0], 'domains': None, 'min_score': None, 'keywords': None})())
        
        assert len(filtered) == 1
        assert filtered[0].payload['cluster_id'] == 0

    def test_apply_filters_domains(self, mock_qdrant_points):
        """Test filtering by domains."""
        filtered = export_engine._apply_filters(mock_qdrant_points,
                                              type('Filter', (), {'cluster_ids': None, 'domains': ['example.com'], 'min_score': None, 'keywords': None})())
        
        assert len(filtered) == 2  # Both articles are from example.com

    def test_apply_filters_min_score(self, mock_qdrant_points):
        """Test filtering by minimum coherence score."""
        filtered = export_engine._apply_filters(mock_qdrant_points,
                                              type('Filter', (), {'cluster_ids': None, 'domains': None, 'min_score': 0.8, 'keywords': None})())
        
        assert len(filtered) == 1  # Only one article has score >= 0.8

    def test_apply_filters_keywords(self, mock_qdrant_points):
        """Test filtering by keywords."""
        filtered = export_engine._apply_filters(mock_qdrant_points,
                                              type('Filter', (), {'cluster_ids': None, 'domains': None, 'min_score': None, 'keywords': ['testing']})())
        
        assert len(filtered) == 1  # Only first article contains 'testing'


class TestExportJobProcessing:
    """Test export job processing functionality."""

    @pytest.mark.asyncio
    @patch('main.qdrant_client')
    async def test_process_export_job_json(self, mock_qdrant, mock_qdrant_points):
        """Test processing a JSON export job."""
        from main import process_export_job, ExportRequest, ExportJob, ExportStatus
        from datetime import datetime
        
        # Mock Qdrant response
        mock_qdrant.scroll.return_value = (mock_qdrant_points, None)
        
        # Create job and request
        job_id = "test_job_123"
        job = ExportJob(
            job_id=job_id,
            session_id="test_session",
            format=ExportFormat.JSON,
            status=ExportStatus.PENDING,
            created_at=datetime.now()
        )
        export_jobs[job_id] = job
        
        request = ExportRequest(
            session_id="test_session",
            format=ExportFormat.JSON
        )
        
        # Process the job
        await process_export_job(job_id, request)
        
        # Verify job completion
        processed_job = export_jobs[job_id]
        assert processed_job.status == ExportStatus.COMPLETED
        assert processed_job.file_path is not None
        assert Path(processed_job.file_path).exists()
        
        # Clean up
        if processed_job.file_path:
            Path(processed_job.file_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    @patch('main.qdrant_client')
    async def test_process_export_job_failure(self, mock_qdrant):
        """Test handling of export job failures."""
        from main import process_export_job, ExportRequest, ExportJob, ExportStatus
        from datetime import datetime
        
        # Mock Qdrant to raise an exception
        mock_qdrant.scroll.side_effect = Exception("Database connection failed")
        
        # Create job and request
        job_id = "test_job_fail"
        job = ExportJob(
            job_id=job_id,
            session_id="test_session",
            format=ExportFormat.JSON,
            status=ExportStatus.PENDING,
            created_at=datetime.now()
        )
        export_jobs[job_id] = job
        
        request = ExportRequest(
            session_id="test_session",
            format=ExportFormat.JSON
        )
        
        # Process the job
        await process_export_job(job_id, request)
        
        # Verify job failure
        processed_job = export_jobs[job_id]
        assert processed_job.status == ExportStatus.FAILED
        assert processed_job.error_message is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])