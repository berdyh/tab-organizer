"""Unit tests for authentication detection logic."""

import pytest
import asyncio
from datetime import datetime
from main import (
    AuthenticationDetector,
    SecureCredentialStore,
    DomainAuthMapper,
    AuthenticationRequirement
)


class TestAuthenticationDetector:
    """Test cases for AuthenticationDetector class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = AuthenticationDetector()

    @pytest.mark.asyncio
    async def test_detect_auth_from_status_code(self):
        """Test authentication detection from HTTP status codes."""
        # Test 401 Unauthorized
        auth_req = await self.detector.detect_auth_required(
            url="https://example.com/protected",
            status_code=401
        )

        assert auth_req.detection_confidence >= 0.8
        assert "HTTP 401 status code" in auth_req.auth_indicators
        assert auth_req.detected_method == "form"
        assert auth_req.domain == "example.com"

    @pytest.mark.asyncio
    async def test_detect_auth_from_headers(self):
        """Test authentication detection from WWW-Authenticate headers."""
        headers = {"www-authenticate": "Basic realm=\"Protected Area\""}

        auth_req = await self.detector.detect_auth_required(
            url="https://example.com/api",
            headers=headers
        )

        assert auth_req.detection_confidence >= 0.9
        assert "WWW-Authenticate header" in str(auth_req.auth_indicators)
        assert auth_req.detected_method == "basic"

    @pytest.mark.asyncio
    async def test_detect_auth_from_login_form(self):
        """Test authentication detection from login forms in content."""
        html_content = """
        <html>
        <body>
            <form action="/login" method="post">
                <input type="text" name="username" />
                <input type="password" name="password" />
                <button type="submit">Login</button>
            </form>
        </body>
        </html>
        """

        auth_req = await self.detector.detect_auth_required(
            url="https://example.com/login",
            response_content=html_content
        )

        assert auth_req.detection_confidence >= 0.7
        assert "Login form detected" in auth_req.auth_indicators
        assert auth_req.detected_method == "form"

    @pytest.mark.asyncio
    async def test_detect_oauth_indicators(self):
        """Test detection of OAuth authentication indicators."""
        html_content = """
        <html>
        <body>
            <div>
                <button>Sign in with Google</button>
                <button>Sign in with GitHub</button>
            </div>
        </body>
        </html>
        """

        auth_req = await self.detector.detect_auth_required(
            url="https://example.com/auth",
            response_content=html_content
        )

        assert auth_req.detection_confidence > 0.5
        assert any("OAuth indicator" in indicator for indicator in auth_req.auth_indicators)
        assert auth_req.detected_method == "oauth"

    @pytest.mark.asyncio
    async def test_detect_protected_content_messages(self):
        """Test detection of protected content messages."""
        html_content = """
        <html>
        <body>
            <div class="error">
                <h2>Access Denied</h2>
                <p>Please log in to continue accessing this content.</p>
            </div>
        </body>
        </html>
        """

        auth_req = await self.detector.detect_auth_required(
            url="https://example.com/protected",
            response_content=html_content
        )

        assert auth_req.detection_confidence >= 0.8
        assert any("Protected content message" in indicator for indicator in auth_req.auth_indicators)

    @pytest.mark.asyncio
    async def test_no_auth_required(self):
        """Test detection when no authentication is required."""
        html_content = """
        <html>
        <body>
            <h1>Welcome to our public website</h1>
            <p>This content is freely accessible.</p>
        </body>
        </html>
        """

        auth_req = await self.detector.detect_auth_required(
            url="https://example.com/public",
            response_content=html_content,
            status_code=200
        )

        assert auth_req.detection_confidence < 0.5
        assert len(auth_req.auth_indicators) == 0

    def test_classify_auth_method_oauth(self):
        """Test OAuth authentication method classification."""
        content = """
        <html>
        <body>
            <button onclick="signInWithGoogle()">Sign in with Google</button>
        </body>
        </html>
        """

        method = self.detector.classify_auth_method(content)
        assert method == "oauth"

    def test_classify_auth_method_form(self):
        """Test form-based authentication method classification."""
        content = """
        <form>
            <input type="text" name="username" />
            <input type="password" name="password" />
        </form>
        """

        method = self.detector.classify_auth_method(content)
        assert method == "form"

    def test_classify_auth_method_basic(self):
        """Test basic authentication method classification."""
        headers = {"www-authenticate": "Basic realm=\"Test\""}

        method = self.detector.classify_auth_method("", headers)
        assert method == "basic"

    def test_classify_auth_method_captcha(self):
        """Test CAPTCHA authentication method classification."""
        content = """
        <div class="g-recaptcha" data-sitekey="test"></div>
        """

        method = self.detector.classify_auth_method(content)
        assert method == "captcha"


class TestSecureCredentialStore:
    """Test cases for SecureCredentialStore class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.store = SecureCredentialStore("test_master_password")

    def test_store_and_retrieve_credentials(self):
        """Test storing and retrieving credentials."""
        domain = "example.com"
        credentials = {
            "username": "testuser",
            "password": "testpass",
            "login_url": "https://example.com/login"
        }

        # Store credentials
        success = self.store.store_credentials(domain, credentials)
        assert success is True

        # Retrieve credentials
        retrieved = self.store.retrieve_credentials(domain)
        assert retrieved == credentials

    def test_retrieve_nonexistent_credentials(self):
        """Test retrieving credentials for non-existent domain."""
        result = self.store.retrieve_credentials("nonexistent.com")
        assert result is None

    def test_delete_credentials(self):
        """Test deleting stored credentials."""
        domain = "example.com"
        credentials = {"username": "test", "password": "test"}

        # Store and then delete
        self.store.store_credentials(domain, credentials)
        success = self.store.delete_credentials(domain)
        assert success is True

        # Verify deletion
        retrieved = self.store.retrieve_credentials(domain)
        assert retrieved is None

    def test_list_stored_domains(self):
        """Test listing domains with stored credentials."""
        domains = ["example1.com", "example2.com", "example3.com"]
        credentials = {"username": "test", "password": "test"}

        # Store credentials for multiple domains
        for domain in domains:
            self.store.store_credentials(domain, credentials)

        # Check listed domains
        stored_domains = self.store.list_stored_domains()
        assert set(stored_domains) == set(domains)

    def test_encryption_integrity(self):
        """Test that stored credentials are actually encrypted."""
        domain = "example.com"
        credentials = {"username": "testuser", "password": "secret123"}

        self.store.store_credentials(domain, credentials)

        # Check that raw stored data doesn't contain plaintext
        raw_data = self.store.credentials_store[domain]
        assert b"testuser" not in raw_data
        assert b"secret123" not in raw_data
        assert isinstance(raw_data, bytes)


class TestDomainAuthMapper:
    """Test cases for DomainAuthMapper class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mapper = DomainAuthMapper()

    def test_learn_domain_auth_new_domain(self):
        """Test learning authentication for a new domain."""
        domain = "example.com"
        auth_req = AuthenticationRequirement(
            url=f"https://{domain}/login",
            domain=domain,
            detected_method="form",
            auth_indicators=["Login form detected"],
            detection_confidence=0.8
        )

        self.mapper.learn_domain_auth(domain, auth_req)

        mapping = self.mapper.get_domain_auth_info(domain)
        assert mapping is not None
        assert mapping.domain == domain
        assert mapping.auth_method == "form"
        assert mapping.requires_auth is True

    def test_learn_domain_auth_update_existing(self):
        """Test updating authentication info for existing domain."""
        domain = "example.com"

        # First learning with low confidence
        auth_req1 = AuthenticationRequirement(
            url=f"https://{domain}",
            domain=domain,
            detected_method="unknown",
            auth_indicators=[],
            detection_confidence=0.3
        )
        self.mapper.learn_domain_auth(domain, auth_req1)

        # Second learning with high confidence
        auth_req2 = AuthenticationRequirement(
            url=f"https://{domain}/login",
            domain=domain,
            detected_method="oauth",
            auth_indicators=["OAuth detected"],
            detection_confidence=0.9
        )
        self.mapper.learn_domain_auth(domain, auth_req2)

        mapping = self.mapper.get_domain_auth_info(domain)
        assert mapping.auth_method == "oauth"
        assert mapping.requires_auth is True
        assert mapping.last_verified is not None

    def test_mark_auth_success_and_failure(self):
        """Test marking authentication success and failure."""
        domain = "example.com"
        auth_req = AuthenticationRequirement(
            url=f"https://{domain}",
            domain=domain,
            detected_method="form",
            auth_indicators=["Form detected"],
            detection_confidence=0.8
        )

        self.mapper.learn_domain_auth(domain, auth_req)

        # Mark successes and failures
        self.mapper.mark_auth_success(domain)
        self.mapper.mark_auth_success(domain)
        self.mapper.mark_auth_failure(domain)

        mapping = self.mapper.get_domain_auth_info(domain)
        assert mapping.success_count == 2
        assert mapping.failure_count == 1
        assert mapping.last_verified is not None

    def test_get_all_mappings(self):
        """Test getting all domain mappings."""
        domains = ["example1.com", "example2.com", "example3.com"]

        for domain in domains:
            auth_req = AuthenticationRequirement(
                url=f"https://{domain}",
                domain=domain,
                detected_method="form",
                auth_indicators=["Test"],
                detection_confidence=0.7
            )
            self.mapper.learn_domain_auth(domain, auth_req)

        all_mappings = self.mapper.get_all_mappings()
        assert len(all_mappings) == 3
        assert set(all_mappings.keys()) == set(domains)

    def test_get_nonexistent_domain_info(self):
        """Test getting info for non-existent domain."""
        result = self.mapper.get_domain_auth_info("nonexistent.com")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])