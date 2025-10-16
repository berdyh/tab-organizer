"""Tests for the Chatbot Service."""

from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

SERVICE_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from datetime import datetime, timezone

import chatbot  # noqa: E402
from chatbot import app, ChatbotService, ChatResponse  # noqa: E402
from chatbot.models import ClusterSummary, ConversationEntry, SearchResult  # noqa: E402

client = TestClient(app)

_MODULE_CACHE_NAME = "_chatbot_service_main"


def _load_chatbot_main():
    module = sys.modules.get(_MODULE_CACHE_NAME)
    if module is not None:
        return module

    module_path = SERVICE_ROOT / "main.py"
    spec = importlib.util.spec_from_file_location(_MODULE_CACHE_NAME, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load chatbot main module for compatibility")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_CACHE_NAME] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def register_main_alias():
    """Expose the chatbot compatibility module under the legacy 'main' name."""

    previous = sys.modules.get("main")
    module = _load_chatbot_main()
    sys.modules["main"] = module
    try:
        yield module
    finally:
        if previous is not None:
            sys.modules["main"] = previous
        else:
            sys.modules.pop("main", None)

@pytest.fixture
def chatbot_service():
    """Create a ChatbotService instance for testing."""
    service = ChatbotService()
    service.ollama_client = AsyncMock()
    return service

@pytest.fixture
def mock_qdrant():
    """Mock Qdrant client."""
    with patch('main.qdrant_client') as mock:
        yield mock

class TestChatbotService:
    """Test the ChatbotService class."""
    
    @pytest.mark.asyncio
    async def test_generate_embedding(self, chatbot_service):
        """Test embedding generation."""
        # Mock Ollama response
        mock_response = Mock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_response.raise_for_status.return_value = None
        chatbot_service.ollama_client.post.return_value = mock_response
        
        embedding = await chatbot_service.generate_embedding("test text")
        
        assert embedding == [0.1, 0.2, 0.3]
        chatbot_service.ollama_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_similar_content(self, chatbot_service, mock_qdrant):
        """Test content search functionality."""
        # Mock embedding generation
        chatbot_service.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
        
        # Mock Qdrant search results
        mock_hit = Mock()
        mock_hit.id = "doc1"
        mock_hit.score = 0.95
        mock_hit.payload = {
            "title": "Test Article",
            "url": "https://example.com",
            "content": "This is test content for the article",
            "cluster_label": "Technology",
            "domain": "example.com"
        }
        mock_qdrant.search.return_value = [mock_hit]
        
        results = await chatbot_service.search_similar_content("AI technology", "test_session")

        assert len(results) == 1
        first = results[0]
        assert first.title == "Test Article"
        assert str(first.url) == "https://example.com/"
        assert first.url.__class__.__name__.endswith("Url")
        assert first.relevance_score == 0.95
        assert first.cluster == "Technology"

    @pytest.mark.asyncio
    async def test_get_session_stats(self, chatbot_service, mock_qdrant):
        """Test session statistics retrieval."""
        # Mock collection info
        mock_qdrant.get_collection.return_value = Mock()
        
        # Mock scroll results
        mock_doc = Mock()
        mock_doc.payload = {
            "domain": "example.com",
            "cluster_label": "Technology",
            "word_count": 500
        }
        mock_qdrant.scroll.return_value = ([mock_doc, mock_doc], None)
        
        stats = await chatbot_service.get_session_stats("test_session")
        
        assert stats["total_documents"] == 2
        assert stats["unique_domains"] == 1
        assert stats["clusters_count"] == 1
        assert stats["average_words"] == 500

    @pytest.mark.asyncio
    async def test_get_cluster_info(self, chatbot_service, mock_qdrant):
        """Test cluster information retrieval."""
        # Mock scroll results
        mock_doc1 = Mock()
        mock_doc1.payload = {
            "title": "AI Article 1",
            "url": "https://example.com/ai1",
            "domain": "example.com",
            "cluster_label": "AI Technology"
        }
        mock_doc2 = Mock()
        mock_doc2.payload = {
            "title": "AI Article 2", 
            "url": "https://example.com/ai2",
            "domain": "example.com",
            "cluster_label": "AI Technology"
        }
        mock_qdrant.scroll.return_value = ([mock_doc1, mock_doc2], None)
        
        clusters = await chatbot_service.get_cluster_info("test_session")

        assert len(clusters) == 1
        cluster = clusters[0]
        assert cluster.title == "AI Technology"
        assert cluster.count == 2
        assert len(cluster.sample_articles) == 2
        assert str(cluster.sample_articles[0].url) == "https://example.com/ai1"

    @pytest.mark.asyncio
    async def test_generate_llm_response(self, chatbot_service):
        """Test LLM response generation."""
        # Mock Ollama response
        mock_response = Mock()
        mock_response.json.return_value = {"response": "This is a test response"}
        mock_response.raise_for_status.return_value = None
        chatbot_service.ollama_client.post.return_value = mock_response
        
        response = await chatbot_service.generate_llm_response("What is AI?", "context")
        
        assert response == "This is a test response"
        chatbot_service.ollama_client.post.assert_called_once()

    def test_extract_intent(self, chatbot_service):
        """Test intent extraction from user messages."""
        # Test search intent
        intent = chatbot_service.extract_intent("Show me articles about AI")
        assert intent["intent"] == "search_content"
        assert "AI" in intent["topic"]
        
        # Test cluster intent
        intent = chatbot_service.extract_intent("What clusters do I have?")
        assert intent["intent"] == "explore_clusters"
        
        # Test summary intent
        intent = chatbot_service.extract_intent("Give me a summary")
        assert intent["intent"] == "get_summary"
        
        # Test general query
        intent = chatbot_service.extract_intent("Hello there")
        assert intent["intent"] == "general_query"

    @pytest.mark.asyncio
    async def test_process_message_search_content(self, chatbot_service):
        """Test processing a content search message."""
        # Mock dependencies
        chatbot_service.search_similar_content = AsyncMock(return_value=[
            SearchResult(
                title="AI Article",
                url="https://example.com",
                snippet="Content about AI",
                cluster="Technology",
                relevance_score=0.95,
            )
        ])
        
        response = await chatbot_service.process_message(
            "test_session",
            "Show me articles about AI",
            []
        )
        
        assert isinstance(response, ChatResponse)
        assert "AI" in response.response
        assert len(response.sources) == 1
        assert len(response.suggestions) > 0

    @pytest.mark.asyncio
    async def test_process_message_explore_clusters(self, chatbot_service):
        """Test processing a cluster exploration message."""
        # Mock dependencies
        chatbot_service.get_cluster_info = AsyncMock(return_value=[
            ClusterSummary(
                title="Technology",
                count=10,
                description="Tech articles",
                sample_articles=[]
            )
        ])
        
        response = await chatbot_service.process_message(
            "test_session",
            "What clusters do I have?",
            []
        )
        
        assert isinstance(response, ChatResponse)
        assert "cluster" in response.response.lower()
        assert len(response.sources) == 1
        assert response.sources[0]["title"] == "Technology"

    @pytest.mark.asyncio
    async def test_process_message_get_summary(self, chatbot_service):
        """Test processing a summary request message."""
        # Mock dependencies
        chatbot_service.get_session_stats = AsyncMock(return_value={
            "total_documents": 100,
            "unique_domains": 20,
            "clusters_count": 5,
            "average_words": 750
        })
        
        response = await chatbot_service.process_message(
            "test_session",
            "Give me a summary",
            []
        )
        
        assert isinstance(response, ChatResponse)
        assert "summary" in response.response.lower()
        assert len(response.sources) == 1
        assert "100" in str(response.sources[0]["metadata"]["Total Articles"])

class TestChatbotAPI:
    """Test the FastAPI endpoints."""
    
    def test_send_message_endpoint(self):
        """Test the chat message endpoint."""
        with patch('main.chatbot_service') as mock_service:
            mock_service.process_message = AsyncMock(return_value=ChatResponse(
                response="Test response",
                sources=[],
                suggestions=["Test suggestion"]
            ))
            
            response = client.post(
                "/chat/message",
                json={
                    "session_id": "test_session",
                    "message": "Hello",
                    "context": []
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["response"] == "Test response"
            assert len(data["suggestions"]) == 1

    def test_get_conversation_history(self):
        """Test getting conversation history."""
        # Add some test conversation data
        from chatbot import conversations
        conversations["test_session"] = [
            {
                "timestamp": "2024-01-01T00:00:00",
                "user_message": "Hello",
                "bot_response": "Hi there!",
                "intent": "general_query",
                "sources_count": 0
            }
        ]
        
        response = client.get("/chat/history/test_session")
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test_session"
        assert len(data["history"]) == 1

    def test_clear_conversation_history(self):
        """Test clearing conversation history."""
        # Add some test conversation data
        from chatbot import conversations

        conversations.append(
            "test_session",
            ConversationEntry(
                timestamp=datetime.now(timezone.utc),
                user_message="Hello",
                bot_response="Hi there!",
                intent="general_query",
                sources_count=0,
            ),
        )
        
        response = client.delete("/chat/history/test_session")
        
        assert response.status_code == 200
        assert conversations.get("test_session") == []

    def test_provide_feedback(self):
        """Test providing feedback on responses."""
        response = client.post(
            "/chat/feedback",
            json={
                "message_id": "msg123",
                "feedback": "positive",
                "comment": "Great response!"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Feedback received"

    def test_health_check(self):
        """Test the health check endpoint."""
        with patch('main.qdrant_client') as mock_qdrant, \
             patch('main.chatbot_service') as mock_service:
            
            mock_qdrant.get_collections.return_value = []
            mock_ollama_client = AsyncMock()
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_ollama_client.get.return_value = mock_response
            mock_service.ollama_client = mock_ollama_client
            
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "qdrant" in data["services"]
            assert "ollama" in data["services"]

    def test_robot_parser_patch_support(self):
        """Ensure robot parser patches propagate to the app state."""

        class DummyRobot:
            pass

        with patch('urllib.robotparser.RobotFileParser', DummyRobot):
            refreshed = app.state.resolve_robot_parser_cls()
            assert refreshed is DummyRobot

@pytest.mark.asyncio
async def test_integration_flow():
    """Test a complete integration flow."""
    service = ChatbotService()
    
    # Mock all external dependencies
    service.ollama_client = AsyncMock()
    
    with patch('main.qdrant_client') as mock_qdrant:
        # Mock embedding generation
        mock_embed_response = Mock()
        mock_embed_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_embed_response.raise_for_status.return_value = None
        service.ollama_client.post.return_value = mock_embed_response
        
        # Mock search results
        mock_hit = Mock()
        mock_hit.id = "doc1"
        mock_hit.score = 0.95
        mock_hit.payload = {
            "title": "AI in Healthcare",
            "url": "https://example.com/ai-healthcare",
            "content": "AI is transforming healthcare...",
            "cluster_label": "Healthcare Technology",
            "domain": "example.com"
        }
        mock_qdrant.search.return_value = [mock_hit]
        
        # Process a search message
        response = await service.process_message(
            "test_session",
            "Show me articles about AI in healthcare",
            []
        )
        
        assert isinstance(response, ChatResponse)
        assert "healthcare" in response.response.lower() or "AI" in response.response
        assert len(response.sources) > 0
        assert len(response.suggestions) > 0

if __name__ == "__main__":
    pytest.main([__file__])
