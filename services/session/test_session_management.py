"""Unit tests for session management operations."""

import pytest
import uuid
import json
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient

from main import (
    app, SessionModel, SessionStatus, CreateSessionRequest, 
    UpdateSessionRequest, ShareSessionRequest, SessionExportData,
    ProcessingStats, ModelUsageHistory, SessionConfiguration,
    sessions_storage
)

# Test client
client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_qdrant_operations():
    """Mock Qdrant operations for testing."""
    with patch('main.get_qdrant_client') as mock_get_client:
        # Create a mock client
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client.create_collection.return_value = None
        mock_client.delete_collection.return_value = None
        mock_client.get_collection.return_value = Mock(
            points_count=100,
            vectors_count=100,
            indexed_vectors_count=100,
            status="green"
        )
        mock_client.scroll.return_value = ([], None)
        mock_client.upsert.return_value = None
        
        # Make get_qdrant_client return our mock
        mock_get_client.return_value = mock_client
        yield mock_client

@pytest.fixture
def sample_session():
    """Create a sample session for testing."""
    session = SessionModel(
        name="Test Session",
        description="A test session",
        owner_id="user123",
        tags=["test", "sample"]
    )
    sessions_storage[session.id] = session
    return session

@pytest.fixture(autouse=True)
def clear_sessions():
    """Clear sessions storage before each test."""
    sessions_storage.clear()
    yield
    sessions_storage.clear()

class TestSessionLifecycle:
    """Test session lifecycle management."""
    
    def test_create_session(self, mock_qdrant_operations):
        """Test session creation."""
        request_data = {
            "name": "New Session",
            "description": "Test session creation",
            "owner_id": "user123",
            "tags": ["test"],
            "metadata": {"key": "value"}
        }
        
        response = client.post("/sessions", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "New Session"
        assert data["description"] == "Test session creation"
        assert data["owner_id"] == "user123"
        assert data["status"] == "active"
        assert "test" in data["tags"]
        assert data["metadata"]["key"] == "value"
        
        # Verify Qdrant collection creation was called
        mock_qdrant_operations.create_collection.assert_called_once()
    
    def test_create_session_minimal(self, mock_qdrant_operations):
        """Test session creation with minimal data."""
        request_data = {"name": "Minimal Session"}
        
        response = client.post("/sessions", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "Minimal Session"
        assert data["status"] == "active"
        assert data["tags"] == []
        assert data["metadata"] == {}
    
    def test_list_sessions_empty(self):
        """Test listing sessions when none exist."""
        response = client.get("/sessions")
        assert response.status_code == 200
        assert response.json() == []
    
    def test_list_sessions(self, sample_session):
        """Test listing sessions."""
        response = client.get("/sessions")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == sample_session.id
        assert data[0]["name"] == sample_session.name
    
    def test_list_sessions_with_filters(self, mock_qdrant_operations):
        """Test listing sessions with filters."""
        # Create multiple sessions
        session1_data = {"name": "Session 1", "owner_id": "user1", "tags": ["tag1"]}
        session2_data = {"name": "Session 2", "owner_id": "user2", "tags": ["tag2"]}
        
        client.post("/sessions", json=session1_data)
        client.post("/sessions", json=session2_data)
        
        # Test owner filter
        response = client.get("/sessions?owner_id=user1")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["owner_id"] == "user1"
        
        # Test tags filter
        response = client.get("/sessions?tags=tag2")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert "tag2" in data[0]["tags"]
    
    def test_get_session(self, sample_session):
        """Test getting a specific session."""
        response = client.get(f"/sessions/{sample_session.id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["id"] == sample_session.id
        assert data["name"] == sample_session.name
    
    def test_get_session_not_found(self):
        """Test getting a non-existent session."""
        fake_id = str(uuid.uuid4())
        response = client.get(f"/sessions/{fake_id}")
        assert response.status_code == 404
    
    def test_update_session(self, sample_session):
        """Test updating a session."""
        update_data = {
            "name": "Updated Session",
            "description": "Updated description",
            "tags": ["updated", "test"],
            "metadata": {"updated": True}
        }
        
        response = client.put(f"/sessions/{sample_session.id}", json=update_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "Updated Session"
        assert data["description"] == "Updated description"
        assert "updated" in data["tags"]
        assert data["metadata"]["updated"] is True
    
    def test_update_session_not_found(self):
        """Test updating a non-existent session."""
        fake_id = str(uuid.uuid4())
        update_data = {"name": "Updated"}
        
        response = client.put(f"/sessions/{fake_id}", json=update_data)
        assert response.status_code == 404
    
    def test_soft_delete_session(self, sample_session):
        """Test soft deleting a session."""
        response = client.delete(f"/sessions/{sample_session.id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["permanent"] is False
        
        # Verify session is marked as deleted
        session = sessions_storage[sample_session.id]
        assert session.status == SessionStatus.DELETED
    
    def test_permanent_delete_session(self, sample_session, mock_qdrant_operations):
        """Test permanently deleting a session."""
        response = client.delete(f"/sessions/{sample_session.id}?permanent=true")
        assert response.status_code == 200
        
        data = response.json()
        assert data["permanent"] is True
        
        # Verify session is removed from storage
        assert sample_session.id not in sessions_storage
        
        # Verify Qdrant collection deletion was called
        mock_qdrant_operations.delete_collection.assert_called_once()
    
    def test_archive_session(self, sample_session):
        """Test archiving a session."""
        response = client.post(f"/sessions/{sample_session.id}/archive")
        assert response.status_code == 200
        
        # Verify session is marked as archived
        session = sessions_storage[sample_session.id]
        assert session.status == SessionStatus.ARCHIVED
    
    def test_restore_session(self, sample_session):
        """Test restoring a session."""
        # First archive the session
        sample_session.status = SessionStatus.ARCHIVED
        sessions_storage[sample_session.id] = sample_session
        
        response = client.post(f"/sessions/{sample_session.id}/restore")
        assert response.status_code == 200
        
        # Verify session is marked as active
        session = sessions_storage[sample_session.id]
        assert session.status == SessionStatus.ACTIVE

class TestSessionSharing:
    """Test session sharing and collaboration features."""
    
    def test_share_session(self, sample_session):
        """Test sharing a session with users."""
        share_data = {
            "user_ids": ["user456", "user789"],
            "permissions": ["read", "write"]
        }
        
        response = client.post(f"/sessions/{sample_session.id}/share", json=share_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "user456" in data["shared_with"]
        assert "user789" in data["shared_with"]
        
        # Verify session status changed to shared
        session = sessions_storage[sample_session.id]
        assert session.status == SessionStatus.SHARED
    
    def test_unshare_session(self, sample_session):
        """Test removing user access from shared session."""
        # First share the session
        sample_session.shared_with = ["user456", "user789"]
        sample_session.status = SessionStatus.SHARED
        sessions_storage[sample_session.id] = sample_session
        
        response = client.delete(f"/sessions/{sample_session.id}/share/user456")
        assert response.status_code == 200
        
        # Verify user was removed
        session = sessions_storage[sample_session.id]
        assert "user456" not in session.shared_with
        assert "user789" in session.shared_with
    
    def test_unshare_last_user(self, sample_session):
        """Test removing the last shared user changes status back to active."""
        # Share with one user
        sample_session.shared_with = ["user456"]
        sample_session.status = SessionStatus.SHARED
        sessions_storage[sample_session.id] = sample_session
        
        response = client.delete(f"/sessions/{sample_session.id}/share/user456")
        assert response.status_code == 200
        
        # Verify status changed back to active
        session = sessions_storage[sample_session.id]
        assert session.status == SessionStatus.ACTIVE
        assert len(session.shared_with) == 0
    
    def test_get_collaborators(self, sample_session):
        """Test getting session collaborators."""
        sample_session.shared_with = ["user456", "user789"]
        sessions_storage[sample_session.id] = sample_session
        
        response = client.get(f"/sessions/{sample_session.id}/collaborators")
        assert response.status_code == 200
        
        data = response.json()
        assert data["owner_id"] == sample_session.owner_id
        assert len(data["shared_with"]) == 2
        assert data["total_collaborators"] == 3  # owner + 2 shared users

class TestSessionExportImport:
    """Test session export and import functionality."""
    
    def test_export_session_without_data(self, sample_session, mock_qdrant_operations):
        """Test exporting session without collection data."""
        response = client.get(f"/sessions/{sample_session.id}/export?include_data=false")
        assert response.status_code == 200
        
        data = response.json()
        assert data["session"]["id"] == sample_session.id
        assert data["session"]["name"] == sample_session.name
        assert data["collection_data"] is None
    
    def test_export_session_with_data(self, sample_session, mock_qdrant_operations):
        """Test exporting session with collection data."""
        # Mock collection data
        mock_points = [
            Mock(id="1", vector=[0.1, 0.2, 0.3], payload={"text": "test1"}),
            Mock(id="2", vector=[0.4, 0.5, 0.6], payload={"text": "test2"})
        ]
        mock_qdrant_operations.scroll.return_value = (mock_points, None)
        
        response = client.get(f"/sessions/{sample_session.id}/export?include_data=true")
        assert response.status_code == 200
        
        data = response.json()
        assert data["session"]["id"] == sample_session.id
        assert data["collection_data"] is not None
        assert data["collection_data"]["exported_points"] == 2
    
    def test_import_session(self, mock_qdrant_operations):
        """Test importing a session."""
        # Create export data
        original_session = SessionModel(
            name="Original Session",
            description="Original description",
            tags=["original"]
        )
        
        export_data = {
            "session": json.loads(original_session.model_dump_json()),
            "collection_data": {
                "points": [
                    {"id": "1", "vector": [0.1, 0.2, 0.3], "payload": {"text": "test1"}},
                    {"id": "2", "vector": [0.4, 0.5, 0.6], "payload": {"text": "test2"}}
                ]
            },
            "export_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        response = client.post("/sessions/import?new_name=Imported Session", json=export_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "Imported Session"
        assert "imported" in data["tags"]
        assert data["metadata"]["imported_from"] == original_session.id
        
        # Verify Qdrant operations were called for import
        assert mock_qdrant_operations.create_collection.called
        assert mock_qdrant_operations.upsert.called

class TestSessionStatistics:
    """Test session statistics and metadata management."""
    
    def test_get_session_stats(self, sample_session, mock_qdrant_operations):
        """Test getting session statistics."""
        response = client.get(f"/sessions/{sample_session.id}/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert data["session_id"] == sample_session.id
        assert "session_stats" in data
        assert "model_usage" in data
        assert "collection_stats" in data
        assert "session_age_days" in data
    
    def test_update_session_stats(self, sample_session):
        """Test updating session processing statistics."""
        stats_update = {
            "urls_processed": 5,
            "content_analyzed": 10,
            "clusters_generated": 3,
            "embeddings_created": 15
        }
        
        response = client.put(f"/sessions/{sample_session.id}/stats", json=stats_update)
        assert response.status_code == 200
        
        # Verify stats were updated
        session = sessions_storage[sample_session.id]
        assert session.processing_stats.urls_processed == 5
        assert session.processing_stats.content_analyzed == 10
        assert session.processing_stats.clusters_generated == 3
        assert session.processing_stats.embeddings_created == 15
    
    def test_update_model_usage(self, sample_session):
        """Test updating model usage history."""
        response = client.put(
            f"/sessions/{sample_session.id}/model-usage?model_type=llm&model_name=gpt-4"
        )
        assert response.status_code == 200
        
        # Verify model usage was updated
        session = sessions_storage[sample_session.id]
        assert "gpt-4" in session.model_usage_history.llm_models_used
        assert session.model_usage_history.model_switches == 1
    
    def test_compare_sessions(self, mock_qdrant_operations):
        """Test comparing multiple sessions."""
        # Create two sessions
        session1_data = {"name": "Session 1", "tags": ["tag1"]}
        session2_data = {"name": "Session 2", "tags": ["tag2"]}
        
        response1 = client.post("/sessions", json=session1_data)
        response2 = client.post("/sessions", json=session2_data)
        
        session1_id = response1.json()["id"]
        session2_id = response2.json()["id"]
        
        # Compare sessions
        response = client.get(f"/sessions/compare?session_ids={session1_id},{session2_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["comparison_summary"]["total_sessions"] == 2
        assert len(data["sessions"]) == 2

if __name__ == "__main__":
    pytest.main([__file__, "-v"])