"""Authentication detection for web pages."""

import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse


@dataclass
class AuthDetectionResult:
    """Result of authentication detection."""
    requires_auth: bool
    auth_type: Optional[str] = None  # basic, oauth, form, cookie
    login_url: Optional[str] = None
    form_fields: Optional[list[str]] = None
    oauth_provider: Optional[str] = None
    confidence: float = 0.0
    details: dict = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class AuthDetector:
    """Detect authentication requirements for web pages."""
    
    # URL patterns indicating login pages
    LOGIN_URL_PATTERNS = [
        r'/login',
        r'/signin',
        r'/sign-in',
        r'/auth',
        r'/authenticate',
        r'/sso',
        r'/oauth',
        r'/account/login',
        r'/user/login',
        r'/session/new',
    ]
    
    # Form field patterns indicating login forms
    LOGIN_FORM_FIELDS = [
        'username', 'user', 'email', 'login',
        'password', 'passwd', 'pass', 'pwd',
    ]
    
    # OAuth provider patterns
    OAUTH_PROVIDERS = {
        'google': ['accounts.google.com', 'googleapis.com/oauth'],
        'github': ['github.com/login/oauth'],
        'facebook': ['facebook.com/dialog/oauth'],
        'twitter': ['api.twitter.com/oauth'],
        'microsoft': ['login.microsoftonline.com', 'login.live.com'],
        'apple': ['appleid.apple.com'],
    }
    
    def __init__(self):
        self._login_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.LOGIN_URL_PATTERNS
        ]
    
    def detect_from_url(self, url: str) -> AuthDetectionResult:
        """Detect auth requirements from URL patterns."""
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        # Check for login URL patterns
        for pattern in self._login_patterns:
            if pattern.search(path):
                return AuthDetectionResult(
                    requires_auth=True,
                    auth_type="form",
                    login_url=url,
                    confidence=0.8,
                    details={"matched_pattern": pattern.pattern},
                )
        
        # Check for OAuth providers
        full_url = url.lower()
        for provider, patterns in self.OAUTH_PROVIDERS.items():
            for pattern in patterns:
                if pattern in full_url:
                    return AuthDetectionResult(
                        requires_auth=True,
                        auth_type="oauth",
                        oauth_provider=provider,
                        login_url=url,
                        confidence=0.9,
                    )
        
        return AuthDetectionResult(requires_auth=False)
    
    def detect_from_response(
        self,
        status_code: int,
        headers: dict,
        url: str,
    ) -> AuthDetectionResult:
        """Detect auth requirements from HTTP response."""
        # Check for 401 Unauthorized
        if status_code == 401:
            www_auth = headers.get('www-authenticate', '').lower()
            if 'basic' in www_auth:
                return AuthDetectionResult(
                    requires_auth=True,
                    auth_type="basic",
                    confidence=1.0,
                    details={"www_authenticate": www_auth},
                )
            elif 'bearer' in www_auth:
                return AuthDetectionResult(
                    requires_auth=True,
                    auth_type="oauth",
                    confidence=1.0,
                    details={"www_authenticate": www_auth},
                )
            return AuthDetectionResult(
                requires_auth=True,
                auth_type="unknown",
                confidence=0.9,
            )
        
        # Check for 403 Forbidden
        if status_code == 403:
            return AuthDetectionResult(
                requires_auth=True,
                auth_type="unknown",
                confidence=0.7,
                details={"reason": "403 Forbidden"},
            )
        
        # Check for redirect to login
        if status_code in (301, 302, 303, 307, 308):
            location = headers.get('location', '')
            url_result = self.detect_from_url(location)
            if url_result.requires_auth:
                url_result.details["redirect_from"] = url
                return url_result
        
        return AuthDetectionResult(requires_auth=False)
    
    def detect_from_html(self, html: str, url: str) -> AuthDetectionResult:
        """Detect auth requirements from HTML content."""
        html_lower = html.lower()
        
        # Check for login form
        form_indicators = [
            '<form',
            'type="password"',
            "type='password'",
            'name="password"',
            "name='password'",
        ]
        
        has_form = '<form' in html_lower
        has_password = 'type="password"' in html_lower or "type='password'" in html_lower
        
        if has_form and has_password:
            # Extract form fields
            form_fields = self._extract_form_fields(html)
            login_fields = [
                f for f in form_fields 
                if any(lf in f.lower() for lf in self.LOGIN_FORM_FIELDS)
            ]
            
            if login_fields:
                return AuthDetectionResult(
                    requires_auth=True,
                    auth_type="form",
                    login_url=url,
                    form_fields=login_fields,
                    confidence=0.85,
                )
        
        # Check for OAuth buttons/links
        oauth_patterns = [
            r'sign\s*in\s*with\s*google',
            r'login\s*with\s*github',
            r'continue\s*with\s*facebook',
            r'sign\s*in\s*with\s*microsoft',
        ]
        
        for pattern in oauth_patterns:
            if re.search(pattern, html_lower):
                return AuthDetectionResult(
                    requires_auth=True,
                    auth_type="oauth",
                    login_url=url,
                    confidence=0.75,
                )
        
        # Check for login-related text
        login_text_patterns = [
            r'please\s*(log\s*in|sign\s*in)',
            r'you\s*must\s*(log\s*in|sign\s*in)',
            r'(log\s*in|sign\s*in)\s*to\s*continue',
            r'authentication\s*required',
        ]
        
        for pattern in login_text_patterns:
            if re.search(pattern, html_lower):
                return AuthDetectionResult(
                    requires_auth=True,
                    auth_type="unknown",
                    login_url=url,
                    confidence=0.6,
                )
        
        return AuthDetectionResult(requires_auth=False)
    
    def _extract_form_fields(self, html: str) -> list[str]:
        """Extract input field names from HTML."""
        # Simple regex extraction
        input_pattern = r'<input[^>]*name=["\']([^"\']+)["\']'
        matches = re.findall(input_pattern, html, re.IGNORECASE)
        return list(set(matches))
    
    def detect(
        self,
        url: str,
        status_code: Optional[int] = None,
        headers: Optional[dict] = None,
        html: Optional[str] = None,
    ) -> AuthDetectionResult:
        """
        Comprehensive auth detection combining all methods.
        
        Returns the highest confidence result.
        """
        results = []
        
        # URL-based detection
        results.append(self.detect_from_url(url))
        
        # Response-based detection
        if status_code and headers:
            results.append(self.detect_from_response(status_code, headers, url))
        
        # HTML-based detection
        if html:
            results.append(self.detect_from_html(html, url))
        
        # Return highest confidence result that requires auth
        auth_results = [r for r in results if r.requires_auth]
        if auth_results:
            return max(auth_results, key=lambda r: r.confidence)
        
        return AuthDetectionResult(requires_auth=False)
