"""Unit tests for Auth Detector."""

import pytest
import sys
sys.path.insert(0, '/app')

from services.browser_engine.app.auth.detector import AuthDetector, AuthDetectionResult


class TestAuthDetector:
    """Tests for AuthDetector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = AuthDetector()
    
    def test_detect_login_url(self):
        """Test detection of login URL patterns."""
        urls = [
            "https://example.com/login",
            "https://example.com/signin",
            "https://example.com/auth/login",
            "https://example.com/user/sign-in",
        ]
        
        for url in urls:
            result = self.detector.detect_from_url(url)
            assert result.requires_auth is True, f"Failed for {url}"
    
    def test_detect_public_url(self):
        """Test that public URLs are not flagged."""
        urls = [
            "https://example.com/",
            "https://example.com/about",
            "https://example.com/products/item-1",
            "https://blog.example.com/post/123",
        ]
        
        for url in urls:
            result = self.detector.detect_from_url(url)
            assert result.requires_auth is False, f"Failed for {url}"
    
    def test_detect_oauth_url(self):
        """Test detection of OAuth URLs."""
        urls = [
            "https://accounts.google.com/oauth/authorize",
            "https://github.com/login/oauth/authorize",
        ]
        
        for url in urls:
            result = self.detector.detect_from_url(url)
            assert result.requires_auth is True
            assert result.auth_type == "oauth"
    
    def test_detect_401_response(self):
        """Test detection of 401 Unauthorized."""
        result = self.detector.detect_from_response(
            status_code=401,
            headers={"www-authenticate": "Basic realm='test'"},
            url="https://example.com/api",
        )
        
        assert result.requires_auth is True
        assert result.auth_type == "basic"
        assert result.confidence == 1.0
    
    def test_detect_403_response(self):
        """Test detection of 403 Forbidden."""
        result = self.detector.detect_from_response(
            status_code=403,
            headers={},
            url="https://example.com/private",
        )
        
        assert result.requires_auth is True
        assert result.confidence == 0.7
    
    def test_detect_redirect_to_login(self):
        """Test detection of redirect to login page."""
        result = self.detector.detect_from_response(
            status_code=302,
            headers={"location": "https://example.com/login?next=/private"},
            url="https://example.com/private",
        )
        
        assert result.requires_auth is True
    
    def test_detect_login_form_html(self):
        """Test detection of login form in HTML."""
        html = """
        <html>
        <body>
            <form action="/login" method="post">
                <input type="text" name="username">
                <input type="password" name="password">
                <button type="submit">Login</button>
            </form>
        </body>
        </html>
        """
        
        result = self.detector.detect_from_html(html, "https://example.com/login")
        
        assert result.requires_auth is True
        assert result.auth_type == "form"
        assert "username" in result.form_fields
        assert "password" in result.form_fields
    
    def test_detect_oauth_button_html(self):
        """Test detection of OAuth buttons in HTML."""
        html = """
        <html>
        <body>
            <button>Sign in with Google</button>
            <a href="/oauth/github">Login with GitHub</a>
        </body>
        </html>
        """
        
        result = self.detector.detect_from_html(html, "https://example.com/login")
        
        assert result.requires_auth is True
        assert result.auth_type == "oauth"
    
    def test_detect_login_text_html(self):
        """Test detection of login-related text in HTML."""
        html = """
        <html>
        <body>
            <h1>Access Denied</h1>
            <p>Please log in to continue viewing this content.</p>
        </body>
        </html>
        """
        
        result = self.detector.detect_from_html(html, "https://example.com/private")
        
        assert result.requires_auth is True
    
    def test_detect_public_html(self):
        """Test that public HTML is not flagged."""
        html = """
        <html>
        <body>
            <h1>Welcome to our site</h1>
            <p>This is public content.</p>
        </body>
        </html>
        """
        
        result = self.detector.detect_from_html(html, "https://example.com/")
        
        assert result.requires_auth is False
    
    def test_comprehensive_detection(self):
        """Test comprehensive detection combining all methods."""
        result = self.detector.detect(
            url="https://example.com/private",
            status_code=401,
            headers={"www-authenticate": "Basic"},
            html=None,
        )
        
        assert result.requires_auth is True
        assert result.confidence == 1.0
    
    def test_highest_confidence_wins(self):
        """Test that highest confidence result is returned."""
        # URL suggests login, but 401 is more definitive
        result = self.detector.detect(
            url="https://example.com/login",
            status_code=401,
            headers={"www-authenticate": "Basic"},
            html=None,
        )
        
        # 401 with Basic auth has confidence 1.0
        assert result.confidence == 1.0
        assert result.auth_type == "basic"
