"""Authentication detection logic."""

import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from .models import AuthenticationRequirement


class AuthenticationDetector:
    """Detects authentication requirements from HTTP responses and page content."""

    def __init__(self) -> None:
        self.auth_patterns = {
            "login_forms": [
                r'<form[^>]*(?:login|signin|auth)[^>]*>',
                r'<input[^>]*type=["\']password["\'][^>]*>',
                r'<input[^>]*name=["\'](?:password|pass|pwd)["\'][^>]*>',
            ],
            "login_links": [
                r'<a[^>]*href=["\'][^"\']*(?:login|signin|auth)[^"\']*["\'][^>]*>',
                r'href=["\'][^"\']*(?:login|signin|auth)[^"\']*["\']',
            ],
            "oauth_indicators": [
                r"oauth",
                r"Sign in with Google",
                r"Sign in with GitHub",
                r"Sign in with Microsoft",
                r"Continue with",
            ],
            "auth_redirects": [
                r"window\.location.*(?:login|signin|auth)",
                r"redirect.*(?:login|signin|auth)",
            ],
            "protected_content": [
                r"Please log in",
                r"Access denied",
                r"Authentication required",
                r"You must be logged in",
                r"Sign in to continue",
            ],
        }

        self.auth_status_codes = {401, 403, 407}

    async def detect_auth_required(
        self,
        url: str,
        response_content: Optional[str] = None,
        status_code: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> AuthenticationRequirement:
        """Analyze response to determine if authentication is required."""
        domain = urlparse(url).netloc
        indicators: List[str] = []
        confidence = 0.0
        detected_method = "unknown"

        if status_code in self.auth_status_codes:
            indicators.append(f"HTTP {status_code} status code")
            confidence += 0.8
            detected_method = "form"

        if headers:
            www_auth = headers.get("www-authenticate", "").lower()
            if www_auth:
                indicators.append(f"WWW-Authenticate header: {www_auth}")
                confidence += 0.9
                if "basic" in www_auth:
                    detected_method = "basic"
                elif "bearer" in www_auth:
                    detected_method = "oauth"

        if response_content:
            content_indicators, content_method, content_confidence = self._analyze_content(response_content)
            indicators.extend(content_indicators)
            confidence = max(confidence, content_confidence)
            if detected_method == "unknown":
                detected_method = content_method

        confidence = min(confidence, 1.0)

        return AuthenticationRequirement(
            url=url,
            domain=domain,
            detected_method=detected_method,
            auth_indicators=indicators,
            detection_confidence=confidence,
        )

    def _analyze_content(self, content: str) -> Tuple[List[str], str, float]:
        """Analyze page content for authentication indicators."""
        indicators: List[str] = []
        confidence = 0.0
        detected_method = "form"

        content_lower = content.lower()

        for pattern in self.auth_patterns["login_forms"]:
            if re.search(pattern, content, re.IGNORECASE):
                indicators.append("Login form detected")
                confidence += 0.7
                detected_method = "form"
                break

        for pattern in self.auth_patterns["oauth_indicators"]:
            if pattern.lower() in content_lower:
                indicators.append(f"OAuth indicator: {pattern}")
                confidence += 0.6
                detected_method = "oauth"

        for pattern in self.auth_patterns["login_links"]:
            if re.search(pattern, content, re.IGNORECASE):
                indicators.append("Login link detected")
                confidence += 0.5

        for pattern in self.auth_patterns["protected_content"]:
            if pattern.lower() in content_lower:
                indicators.append(f"Protected content message: {pattern}")
                confidence += 0.8

        for pattern in self.auth_patterns["auth_redirects"]:
            if re.search(pattern, content, re.IGNORECASE):
                indicators.append("Authentication redirect detected")
                confidence += 0.6

        return indicators, detected_method, confidence

    def classify_auth_method(self, content: str, headers: Optional[Dict[str, str]] = None) -> str:
        """Classify the type of authentication method required."""
        content_lower = content.lower()

        oauth_providers = ["google", "github", "microsoft", "facebook", "twitter", "linkedin"]
        for provider in oauth_providers:
            if f"sign in with {provider}" in content_lower or f"oauth/{provider}" in content_lower:
                return "oauth"

        if re.search(r'<input[^>]*type=["\']password["\']', content, re.IGNORECASE):
            return "form"

        if headers and "www-authenticate" in headers:
            auth_header = headers["www-authenticate"].lower()
            if "basic" in auth_header:
                return "basic"
            if "bearer" in auth_header:
                return "bearer"

        if "captcha" in content_lower or "recaptcha" in content_lower:
            return "captcha"

        return "unknown"


__all__ = ["AuthenticationDetector"]
