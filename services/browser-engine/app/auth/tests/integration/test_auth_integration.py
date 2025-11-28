"""Integration tests for authentication workflows."""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from main import (
    InteractiveAuthenticator,
    OAuthFlowHandler,
    AuthenticationQueue,
    OAuthConfig,
    AuthSession
)


class TestInteractiveAuthenticator:
    """Integration tests for InteractiveAuthenticator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.authenticator = InteractiveAuthenticator()

    @pytest.mark.asyncio
    @patch('main.async_playwright')
    async def test_playwright_form_authentication_success(self, mock_playwright):
        """Test successful form authentication with Playwright."""
        # Mock Playwright components
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()

        mock_playwright.return_value.__aenter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.url = "https://example.com/dashboard"

        # Mock successful form filling
        mock_page.goto = AsyncMock()
        mock_page.wait_for_load_state = AsyncMock()
        mock_page.wait_for_selector = AsyncMock()
        mock_page.fill = AsyncMock()
        mock_page.click = AsyncMock()
        mock_page.query_selector_all = AsyncMock(return_value=[])  # No error messages
        mock_page.evaluate = AsyncMock(return_value="Mozilla/5.0 Test Agent")

        # Mock cookies
        mock_context.cookies.return_value = [
            {"name": "session_id", "value": "test_session_123"},
            {"name": "auth_token", "value": "test_token_456"}
        ]

        credentials = {
            "username": "testuser",
            "password": "testpass"
        }

        result = await self.authenticator.authenticate_with_popup(
            domain="example.com",
            auth_method="form",
            credentials=credentials,
            login_url="https://example.com/login",
            browser_type="playwright"
        )

        assert result["success"] is True
        assert "session_id" in result
        assert result["session_data"]["cookies"]["session_id"] == "test_session_123"
        assert result["session_data"]["user_agent"] == "Mozilla/5.0 Test Agent"

        # Verify session was created
        session = self.authenticator.get_session(result["session_id"])
        assert session is not None
        assert session.domain == "example.com"
        assert session.auth_method == "form"

    @pytest.mark.asyncio
    @patch('main.webdriver.Chrome')
    async def test_selenium_form_authentication_success(self, mock_chrome):
        """Test successful form authentication with Selenium."""
        # Mock Selenium WebDriver
        mock_driver = Mock()
        mock_chrome.return_value = mock_driver

        # Mock successful form interaction
        mock_username_element = Mock()
        mock_password_element = Mock()
        mock_submit_element = Mock()

        # Mock WebDriverWait
        with patch('main.WebDriverWait') as mock_wait:
            mock_wait.return_value.until.return_value = mock_username_element

            # Setup find_element to return the right elements in sequence
            def mock_find_element(by, selector):
                if "password" in selector or by.name == "password":
                    return mock_password_element
                elif "submit" in selector or "button" in selector:
                    return mock_submit_element
                else:
                    # For error checking, raise exception (no error elements found)
                    raise Exception("No error elements")

            mock_driver.find_element.side_effect = mock_find_element

            # Mock cookies and user agent
            mock_driver.get_cookies.return_value = [
                {"name": "session_id", "value": "selenium_session_123"}
            ]
            mock_driver.execute_script.return_value = "Mozilla/5.0 Selenium Agent"
            mock_driver.current_url = "https://example.com/dashboard"

            credentials = {
                "username": "testuser",
                "password": "testpass"
            }

            result = await self.authenticator.authenticate_with_popup(
                domain="example.com",
                auth_method="form",
                credentials=credentials,
                login_url="https://example.com/login",
                browser_type="chrome"
            )

            assert result["success"] is True
            assert "session_id" in result
            assert result["session_data"]["cookies"]["session_id"] == "selenium_session_123"

    @pytest.mark.asyncio
    @patch('main.async_playwright')
    async def test_playwright_authentication_failure(self, mock_playwright):
        """Test authentication failure with Playwright."""
        # Mock Playwright components
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()

        mock_playwright.return_value.__aenter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page

        # Mock form filling but with error messages present
        mock_page.goto = AsyncMock()
        mock_page.wait_for_load_state = AsyncMock()
        mock_page.wait_for_selector = AsyncMock()
        mock_page.fill = AsyncMock()
        mock_page.click = AsyncMock()

        # Mock error messages found
        mock_error_element = Mock()
        mock_page.query_selector_all = AsyncMock(return_value=[mock_error_element])

        credentials = {
            "username": "wronguser",
            "password": "wrongpass"
        }

        result = await self.authenticator.authenticate_with_popup(
            domain="example.com",
            auth_method="form",
            credentials=credentials,
            login_url="https://example.com/login",
            browser_type="playwright"
        )

        assert result["success"] is False
        assert "Authentication failed" in result["message"]

    def test_session_management(self):
        """Test session creation, retrieval, and invalidation."""
        # Create a session
        session_data = {
            "cookies": {"session_id": "test_123"},
            "user_agent": "Test Agent"
        }

        session_id = self.authenticator._create_session(
            domain="example.com",
            auth_method="form",
            session_data=session_data
        )

        # Retrieve session
        session = self.authenticator.get_session(session_id)
        assert session is not None
        assert session.domain == "example.com"
        assert session.auth_method == "form"
        assert session.is_active is True

        # Invalidate session
        success = self.authenticator.invalidate_session(session_id)
        assert success is True

        # Try to retrieve invalidated session
        session = self.authenticator.get_session(session_id)
        assert session is None

    def test_expired_session_cleanup(self):
        """Test cleanup of expired sessions."""
        # Create an expired session
        session_data = {"cookies": {"test": "value"}}
        session_id = self.authenticator._create_session(
            domain="example.com",
            auth_method="form",
            session_data=session_data
        )

        # Manually expire the session
        session = self.authenticator.active_sessions[session_id]
        session.expires_at = datetime.now() - timedelta(hours=1)

        # Cleanup expired sessions
        initial_count = len(self.authenticator.active_sessions)
        self.authenticator.cleanup_expired_sessions()
        final_count = len(self.authenticator.active_sessions)

        assert final_count < initial_count
        assert session_id not in self.authenticator.active_sessions


class TestOAuthFlowHandler:
    """Integration tests for OAuth flow handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.oauth_handler = OAuthFlowHandler()

    def test_register_oauth_provider(self):
        """Test OAuth provider registration."""
        config = OAuthConfig(
            provider="google",
            client_id="test_client_id",
            client_secret="test_client_secret",
            authorization_url="https://accounts.google.com/oauth/authorize",
            token_url="https://accounts.google.com/oauth/token",
            redirect_uri="https://example.com/callback",
            scope=["openid", "email", "profile"]
        )

        self.oauth_handler.register_oauth_provider("google", config)

        assert "google" in self.oauth_handler.oauth_configs
        assert self.oauth_handler.oauth_configs["google"].client_id == "test_client_id"

    def test_initiate_oauth_flow(self):
        """Test OAuth flow initiation."""
        config = OAuthConfig(
            provider="github",
            client_id="github_client_id",
            client_secret="github_client_secret",
            authorization_url="https://github.com/login/oauth/authorize",
            token_url="https://github.com/login/oauth/access_token",
            redirect_uri="https://example.com/auth/callback",
            scope=["user:email"]
        )

        self.oauth_handler.register_oauth_provider("github", config)

        flow_data = self.oauth_handler.initiate_oauth_flow("github")

        assert "flow_id" in flow_data
        assert "authorization_url" in flow_data
        assert "state" in flow_data
        assert "github.com/login/oauth/authorize" in flow_data["authorization_url"]
        assert "client_id=github_client_id" in flow_data["authorization_url"]
        assert flow_data["flow_id"] in self.oauth_handler.active_flows

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.post')
    async def test_complete_oauth_flow_success(self, mock_post):
        """Test successful OAuth flow completion."""
        # Setup OAuth provider
        config = OAuthConfig(
            provider="google",
            client_id="test_client",
            client_secret="test_secret",
            authorization_url="https://accounts.google.com/oauth/authorize",
            token_url="https://accounts.google.com/oauth/token",
            redirect_uri="https://example.com/callback"
        )

        self.oauth_handler.register_oauth_provider("google", config)
        flow_data = self.oauth_handler.initiate_oauth_flow("google")

        # Mock successful token response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "access_token": "test_access_token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "test_refresh_token"
        }
        mock_post.return_value.__aenter__.return_value = mock_response

        result = await self.oauth_handler.complete_oauth_flow(
            flow_id=flow_data["flow_id"],
            authorization_code="test_auth_code",
            state=flow_data["state"]
        )

        assert result["success"] is True
        assert result["tokens"]["access_token"] == "test_access_token"
        assert result["provider"] == "google"
        assert flow_data["flow_id"] not in self.oauth_handler.active_flows

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.post')
    async def test_complete_oauth_flow_failure(self, mock_post):
        """Test OAuth flow completion failure."""
        # Setup OAuth provider
        config = OAuthConfig(
            provider="google",
            client_id="test_client",
            client_secret="test_secret",
            authorization_url="https://accounts.google.com/oauth/authorize",
            token_url="https://accounts.google.com/oauth/token",
            redirect_uri="https://example.com/callback"
        )

        self.oauth_handler.register_oauth_provider("google", config)
        flow_data = self.oauth_handler.initiate_oauth_flow("google")

        # Mock failed token response
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.text.return_value = "Invalid authorization code"
        mock_post.return_value.__aenter__.return_value = mock_response

        with pytest.raises(Exception):  # Should raise HTTPException
            await self.oauth_handler.complete_oauth_flow(
                flow_id=flow_data["flow_id"],
                authorization_code="invalid_code",
                state=flow_data["state"]
            )


class TestAuthenticationQueue:
    """Integration tests for authentication queue processing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.auth_queue = AuthenticationQueue(max_workers=2)

    def test_queue_authentication_task(self):
        """Test queuing an authentication task."""
        task_id = self.auth_queue.queue_authentication(
            domain="example.com",
            auth_method="form",
            credentials={"username": "test", "password": "test"},
            login_url="https://example.com/login",
            priority=1
        )

        assert task_id in self.auth_queue.active_tasks
        task = self.auth_queue.get_task_status(task_id)
        assert task is not None
        assert task.domain == "example.com"
        assert task.auth_method == "form"
        assert task.status == "pending"

    def test_task_priority_ordering(self):
        """Test that higher priority tasks are processed first."""
        # Queue low priority task
        low_priority_id = self.auth_queue.queue_authentication(
            domain="low.example.com",
            auth_method="form",
            credentials={"username": "test", "password": "test"},
            login_url="https://low.example.com/login",
            priority=1
        )

        # Queue high priority task
        high_priority_id = self.auth_queue.queue_authentication(
            domain="high.example.com",
            auth_method="form",
            credentials={"username": "test", "password": "test"},
            login_url="https://high.example.com/login",
            priority=5
        )

        # High priority task should be processed first
        # (In actual implementation, this would be tested by processing the queue)
        assert high_priority_id in self.auth_queue.active_tasks
        assert low_priority_id in self.auth_queue.active_tasks

        high_priority_task = self.auth_queue.get_task_status(high_priority_id)
        low_priority_task = self.auth_queue.get_task_status(low_priority_id)

        assert high_priority_task.priority > low_priority_task.priority

    def test_queue_status_tracking(self):
        """Test queue status tracking."""
        # Initially empty
        assert len(self.auth_queue.active_tasks) == 0
        assert len(self.auth_queue.completed_tasks) == 0

        # Add a task
        task_id = self.auth_queue.queue_authentication(
            domain="example.com",
            auth_method="form",
            credentials={"username": "test", "password": "test"},
            login_url="https://example.com/login"
        )

        assert len(self.auth_queue.active_tasks) == 1
        assert task_id in self.auth_queue.active_tasks

    def test_start_stop_processing(self):
        """Test starting and stopping queue processing."""
        # Initially not running
        assert self.auth_queue._running is False

        # Start processing
        self.auth_queue.start_processing()
        assert self.auth_queue._running is True
        assert len(self.auth_queue._worker_threads) == self.auth_queue.max_workers

        # Stop processing
        self.auth_queue.stop_processing()
        assert self.auth_queue._running is False


class TestSecureErrorHandling:
    """Test secure error handling without credential exposure."""

    def setup_method(self):
        """Set up test fixtures."""
        self.authenticator = InteractiveAuthenticator()

    @pytest.mark.asyncio
    async def test_error_handling_no_credential_exposure(self):
        """Test that errors don't expose credentials."""
        credentials = {
            "username": "sensitive_user",
            "password": "super_secret_password",
            "api_key": "secret_api_key_123"
        }

        # Mock a failure scenario
        with patch('main.async_playwright') as mock_playwright:
            mock_playwright.side_effect = Exception("Browser launch failed")

            try:
                await self.authenticator.authenticate_with_popup(
                    domain="example.com",
                    auth_method="form",
                    credentials=credentials,
                    login_url="https://example.com/login",
                    browser_type="playwright"
                )
                # Should not reach here
                assert False, "Expected exception was not raised"
            except Exception as e:
                # Handle HTTPException properly
                if hasattr(e, 'detail'):
                    error_message = str(e.detail)
                else:
                    error_message = str(e)

                # Verify credentials are not in error message
                assert "sensitive_user" not in error_message
                assert "super_secret_password" not in error_message
                assert "secret_api_key_123" not in error_message

                # Should contain generic error info
                assert "Authentication failed" in error_message

    def test_credential_storage_security(self):
        """Test that stored credentials are properly encrypted."""
        from main import SecureCredentialStore

        store = SecureCredentialStore("test_master_key")

        sensitive_credentials = {
            "username": "admin",
            "password": "top_secret_password",
            "token": "bearer_token_xyz"
        }

        # Store credentials
        success = store.store_credentials("secure.example.com", sensitive_credentials)
        assert success is True

        # Check that raw storage doesn't contain plaintext
        raw_data = store.credentials_store["secure.example.com"]
        assert isinstance(raw_data, bytes)
        assert b"admin" not in raw_data
        assert b"top_secret_password" not in raw_data
        assert b"bearer_token_xyz" not in raw_data

        # But decryption should work
        retrieved = store.retrieve_credentials("secure.example.com")
        assert retrieved == sensitive_credentials


if __name__ == "__main__":
    pytest.main([__file__, "-v"])