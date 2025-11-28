"""Data models and schemas for the authentication service."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, HttpUrl


@dataclass
class AuthenticationRequirement:
    """Represents an authentication requirement detected for a URL/domain."""

    url: str
    domain: str
    detected_method: str  # form, oauth, captcha, etc.
    auth_indicators: List[str]
    priority: int = 1
    detection_confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class DomainAuthMapping:
    """Maps domain to authentication requirements and credentials."""

    domain: str
    auth_method: str
    requires_auth: bool
    login_url: Optional[str] = None
    form_selectors: Dict[str, str] = field(default_factory=dict)
    oauth_config: Dict[str, Any] = field(default_factory=dict)
    last_verified: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0


@dataclass
class AuthSession:
    """Represents an active authentication session."""

    session_id: str
    domain: str
    auth_method: str
    cookies: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    tokens: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    last_used: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    user_agent: Optional[str] = None


@dataclass
class OAuthConfig:
    """OAuth 2.0 configuration for a provider."""

    provider: str
    client_id: str
    client_secret: str
    authorization_url: str
    token_url: str
    redirect_uri: str
    scope: List[str] = field(default_factory=list)
    additional_params: Dict[str, str] = field(default_factory=dict)


@dataclass
class AuthenticationTask:
    """Represents a queued authentication task."""

    task_id: str
    domain: str
    auth_method: str
    credentials: Dict[str, Any]
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, processing, completed, failed
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class URLAnalysisRequest(BaseModel):
    url: HttpUrl
    response_content: Optional[str] = None
    status_code: Optional[int] = None
    headers: Optional[Dict[str, str]] = None


class CredentialStoreRequest(BaseModel):
    domain: str
    auth_method: str
    credentials: Dict[str, Any]
    login_url: Optional[str] = None


class AuthDetectionResponse(BaseModel):
    requires_auth: bool
    detected_method: str
    confidence: float
    indicators: List[str]
    recommended_action: str


class InteractiveAuthRequest(BaseModel):
    domain: str
    auth_method: str
    credentials: Dict[str, Any]
    login_url: str
    browser_type: str = "chrome"
    headless: bool = True
    timeout: int = 30


class OAuthAuthRequest(BaseModel):
    domain: str
    provider: str
    client_id: str
    client_secret: str
    redirect_uri: str
    scope: Optional[List[str]] = None


class SessionRequest(BaseModel):
    domain: str
    session_data: Optional[Dict[str, Any]] = None


class AuthTaskResponse(BaseModel):
    task_id: str
    status: str
    message: str
    session_id: Optional[str] = None


__all__ = [
    "AuthenticationRequirement",
    "DomainAuthMapping",
    "AuthSession",
    "OAuthConfig",
    "AuthenticationTask",
    "URLAnalysisRequest",
    "CredentialStoreRequest",
    "AuthDetectionResponse",
    "InteractiveAuthRequest",
    "OAuthAuthRequest",
    "SessionRequest",
    "AuthTaskResponse",
]
