"""Integration tests for Web Scraper Service authentication features."""

import asyncio
from datetime import datetime, timedelta
from typing import cast
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from pydantic import HttpUrl

from services.scraper.app.auth_client import AuthenticationServiceClient
from services.scraper.app.engine import ScrapingEngine
from services.scraper.app.models import AuthSession, ScrapeRequest, URLRequest
from services.scraper.app.state import state

active_jobs = state.active_jobs
auth_sessions = state.auth_sessions


def _http(url: str) -> HttpUrl:
    """Helper for type-check friendly HttpUrl literals."""
    return cast(HttpUrl, url)


class TestAuthenticationServiceClient:
    """Test authentication service client functionality."""
    
    @pytest.fixture
    def auth_client(self):
        """Create auth client for testing."""
        return AuthenticationServiceClient("http://test-auth:8082")
    
    @pytest.mark.asyncio
    async def test_get_session_for_domain_success(self, auth_client):
        """Test successful session retrieval."""
        mock_session_data = {
            "session_id": "test_session_123",
            "domain": "example.com",
            "cookies": {"session": "abc123"},
            "headers": {"Authorization": "Bearer token"},
            "user_agent": "TestAgent/1.0",
            "expires_at": (datetime.now() + timedelta(hours=1)).isoformat(),
            "is_active": True
        }
        
        with patch.object(auth_client, '_get_session') as mock_get_session:
            # Mock the response object (what __aenter__ should yield)
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_session_data)
            
            # Mock the context manager returned by session.get(...) - use MagicMock, not AsyncMock
            mock_get_cm = MagicMock()
            mock_get_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get_cm.__aexit__ = AsyncMock(return_value=None)
            
            # Mock the session
            mock_session = MagicMock()
            mock_session.get.return_value = mock_get_cm
            
            mock_get_session.return_value = mock_session
            
            session = await auth_client.get_session_for_domain("example.com", "test_corr_123")
            
            assert session is not None
            assert session.session_id == "test_session_123"
            assert session.domain == "example.com"
            assert session.cookies == {"session": "abc123"}
            assert session.is_active is True
    
    @pytest.mark.asyncio
    async def test_get_session_for_domain_not_found(self, auth_client):
        """Test session not found scenario."""
        with patch.object(auth_client, '_get_session') as mock_get_session:
            # Mock the response object
            mock_response = MagicMock()
            mock_response.status = 404
            
            # Mock the context manager returned by session.get(...) - use MagicMock
            mock_get_cm = MagicMock()
            mock_get_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get_cm.__aexit__ = AsyncMock(return_value=None)
            
            # Mock the session
            mock_session = MagicMock()
            mock_session.get.return_value = mock_get_cm
            
            mock_get_session.return_value = mock_session
            
            session = await auth_client.get_session_for_domain("example.com", "test_corr_123")
            
            assert session is None
    
    @pytest.mark.asyncio
    async def test_check_auth_required_success(self, auth_client):
        """Test auth requirement detection."""
        mock_auth_data = {
            "requires_auth": True,
            "detected_method": "form",
            "confidence": 0.8,
            "indicators": ["Login form detected"],
            "recommended_action": "authenticate"
        }
        
        with patch.object(auth_client, '_get_session') as mock_get_session:
            # Mock the response object (what __aenter__ should yield)
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_auth_data)
            
            # Mock the context manager returned by session.post(...) - use MagicMock
            mock_post_cm = MagicMock()
            mock_post_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post_cm.__aexit__ = AsyncMock(return_value=None)
            
            # Mock the session
            mock_session = MagicMock()
            mock_session.post.return_value = mock_post_cm
            
            mock_get_session.return_value = mock_session
            
            result = await auth_client.check_auth_required(
                "https://example.com/protected",
                "<html>Please log in</html>",
                401,
                {"www-authenticate": "Basic"},
                "test_corr_123"
            )
            
            assert result["requires_auth"] is True
            assert result["detected_method"] == "form"
            assert result["confidence"] == 0.8
    
    @pytest.mark.asyncio
    async def test_check_auth_required_error_handling(self, auth_client):
        """Test error handling in auth detection."""
        with patch.object(auth_client, '_get_session') as mock_get_session:
            mock_get_session.side_effect = Exception("Network error")
            
            result = await auth_client.check_auth_required(
                "https://example.com/test",
                "<html>Test</html>",
                200,
                {},
                "test_corr_123"
            )
            
            assert result["requires_auth"] is False
            assert result["confidence"] == 0.0


class TestScrapingEngineAuthIntegration:
    """Test scraping engine authentication integration."""
    
    @pytest.fixture
    def scraping_engine(self):
        """Create scraping engine for testing."""
        return ScrapingEngine()
    
    @pytest.fixture
    def sample_scrape_request(self):
        """Create sample scrape request."""
        return ScrapeRequest(
            urls=[
                URLRequest(url=_http("https://example.com/page1")),
                URLRequest(url=_http("https://admin.example.org/secure/dashboard")),
                URLRequest(url=_http("https://member.example.net/profile")),
            ],
            session_id="test_session",
            rate_limit_delay=0.5,
            respect_robots=True,
            max_retries=2
        )
    
    def setup_method(self):
        """Clear global state before each test."""
        active_jobs.clear()
        auth_sessions.clear()
    
    @pytest.mark.asyncio
    async def test_prepare_auth_sessions(self, scraping_engine, sample_scrape_request):
        """Test auth session preparation for domains."""
        # Mock auth client
        mock_session = AuthSession(
            session_id="session_123",
            domain="example.com",
            cookies={"auth": "token123"},
            headers={"Authorization": "Bearer abc"},
            is_active=True
        )
        
        with patch.object(scraping_engine.auth_client, 'get_session_for_domain') as mock_get_session:
            mock_get_session.return_value = mock_session
            
            # Create a job
            job = Mock()
            job.auth_sessions = {}
            
            await scraping_engine._prepare_auth_sessions(sample_scrape_request, job, "test_corr_123")
            
            # Verify session was stored
            assert "session_123" in auth_sessions
            assert auth_sessions["session_123"].domain == "example.com"
            assert job.auth_sessions["example.com"] == "session_123"
    
    @pytest.mark.asyncio
    async def test_scrape_urls_with_auth_integration(self, scraping_engine, sample_scrape_request):
        """Test complete scraping workflow with auth integration."""
        with patch.object(scraping_engine, '_prepare_auth_sessions') as mock_prepare_auth:
            with patch.object(scraping_engine.runner, 'crawl') as mock_crawl:
                mock_crawl.return_value = Mock()
                
                job_id = await scraping_engine.scrape_urls(sample_scrape_request)
                
                # Verify job was created
                assert job_id in active_jobs
                job = active_jobs[job_id]
                assert job.status == "running"
                assert job.total_urls == 3
                assert job.correlation_id.startswith("scrape_")
                
                # Verify auth preparation was called
                mock_prepare_auth.assert_called_once()
                
                # Verify crawler was started with correct parameters
                mock_crawl.assert_called_once()
                call_args = mock_crawl.call_args
                assert call_args[1]['job_id'] == job_id
                assert call_args[1]['correlation_id'] == job.correlation_id


class TestAuthenticationWorkflow:
    """Test complete authentication workflow integration."""
    
    def setup_method(self):
        """Clear global state before each test."""
        active_jobs.clear()
        auth_sessions.clear()
    
    def test_auth_session_injection_logic(self):
        """Test auth session injection into requests."""
        # Create mock session
        session = AuthSession(
            session_id="test_session",
            domain="example.com",
            cookies={"session_id": "abc123", "csrf_token": "xyz789"},
            headers={"Authorization": "Bearer token123"},
            user_agent="CustomAgent/1.0",
            is_active=True
        )
        auth_sessions["test_session"] = session
        
        # Create mock job with auth session mapping
        job = Mock()
        job.auth_sessions = {"example.com": "test_session"}
        active_jobs["test_job"] = job
        
        # Create mock request
        mock_request = Mock()
        mock_request.cookies = {}
        mock_request.headers = {}
        mock_request.meta = {}
        
        # Test injection (would be called by spider)
        from services.scraper.main import ContentSpider
        spider = ContentSpider(job_id="test_job")
        spider._inject_auth_session(mock_request, "example.com")
        
        # Verify session data was injected
        assert mock_request.cookies == {"session_id": "abc123", "csrf_token": "xyz789"}
        assert mock_request.headers["Authorization"] == "Bearer token123"
        assert mock_request.headers["User-Agent"] == "CustomAgent/1.0"
        assert mock_request.meta["auth_session_used"] is True
        assert mock_request.meta["session_id"] == "test_session"
    
    def test_auth_requirement_detection(self):
        """Test authentication requirement detection logic."""
        from services.scraper.main import ContentSpider
        spider = ContentSpider()
        
        # Test 401 status code
        mock_response_401 = Mock()
        mock_response_401.status = 401
        mock_response_401.headers = {}
        mock_response_401.text = ""
        
        assert spider._requires_authentication(mock_response_401) is True
        
        # Test 403 status code
        mock_response_403 = Mock()
        mock_response_403.status = 403
        mock_response_403.headers = {}
        mock_response_403.text = ""
        
        assert spider._requires_authentication(mock_response_403) is True
        
        # Test auth redirect
        mock_response_redirect = Mock()
        mock_response_redirect.status = 302
        mock_response_redirect.headers = {"Location": b"https://example.com/login"}
        mock_response_redirect.text = ""
        
        assert spider._requires_authentication(mock_response_redirect) is True
        
        # Test content-based detection
        mock_response_content = Mock()
        mock_response_content.status = 200
        mock_response_content.headers = {}
        mock_response_content.text = "<html><body>Please log in to continue</body></html>"
        
        assert spider._requires_authentication(mock_response_content) is True
        
        # Test normal response
        mock_response_normal = Mock()
        mock_response_normal.status = 200
        mock_response_normal.headers = {}
        mock_response_normal.text = "<html><body>Welcome to our site</body></html>"
        
        assert spider._requires_authentication(mock_response_normal) is False
    
    def test_retry_queue_functionality(self):
        """Test retry queue for auth-failed URLs."""
        from services.scraper.main import ContentSpider
        spider = ContentSpider(job_id="test_job")
        
        # Create mock URL request
        url_request = Mock()
        url_request.url = "https://example.com/protected"
        
        # Queue for retry
        spider._queue_for_auth_retry(url_request, "example.com", 1)
        
        # Verify item was queued
        assert len(spider.retry_queue) == 1
        retry_item = spider.retry_queue[0]
        assert retry_item['url_request'] == url_request
        assert retry_item['domain'] == "example.com"
        assert retry_item['retry_count'] == 1
        assert 'queued_at' in retry_item


class TestErrorHandlingAndLogging:
    """Test error handling and logging with correlation IDs."""
    
    def test_correlation_id_generation(self):
        """Test correlation ID generation and usage."""
        from services.scraper.main import ScrapingEngine
        engine = ScrapingEngine()
        
        # Mock the scraping process
        with patch.object(engine, '_prepare_auth_sessions') as mock_prepare:
            with patch.object(engine.runner, 'crawl') as mock_crawl:
                mock_crawl.return_value = Mock()
                
                # Create request
                request = ScrapeRequest(
                    urls=[URLRequest(url=_http("https://example.com/reference"))],
                    session_id="test_session"
                )
                
                # Run async function
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    job_id = loop.run_until_complete(engine.scrape_urls(request))
                    
                    # Verify correlation ID was generated
                    job = active_jobs[job_id]
                    assert job.correlation_id.startswith("scrape_")
                    # Correlation ID format: scrape_job_timestamp_counter_uuid
                    assert len(job.correlation_id.split('_')) >= 3  # At least 3 parts
                finally:
                    loop.close()
    
    def test_graceful_degradation(self):
        """Test graceful degradation when auth service is unavailable."""
        from services.scraper.main import AuthenticationServiceClient
        
        client = AuthenticationServiceClient()
        
        # Test with network error
        with patch.object(client, '_get_session') as mock_get_session:
            mock_get_session.side_effect = Exception("Connection refused")
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    client.get_session_for_domain("example.com", "test_corr")
                )
                # Should return None instead of crashing
                assert result is None
            finally:
                loop.close()


if __name__ == "__main__":
    pytest.main([__file__])
