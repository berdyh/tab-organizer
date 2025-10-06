"""Authentication and authorization middleware for API Gateway."""

import time
import hashlib
import secrets
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta

import structlog
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

logger = structlog.get_logger()


class AuthToken(BaseModel):
    """Authentication token model."""
    token: str
    user_id: str
    permissions: List[str]
    expires_at: float
    created_at: float
    last_used: float


class AuthMiddleware:
    """Authentication and authorization middleware."""
    
    def __init__(self):
        # Token storage (in production, use Redis or database)
        self.tokens: Dict[str, AuthToken] = {}
        
        # API keys for service-to-service communication
        self.api_keys: Dict[str, Dict[str, Any]] = {
            # Default internal service key
            "internal-service-key": {
                "name": "internal-services",
                "permissions": ["*"],  # Full access
                "created_at": time.time(),
                "last_used": time.time()
            }
        }
        
        # Session storage
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        # Permission definitions
        self.permissions = {
            "read": "Read access to resources",
            "write": "Write access to resources", 
            "admin": "Administrative access",
            "scrape": "Web scraping operations",
            "analyze": "Content analysis operations",
            "cluster": "Clustering operations",
            "export": "Export operations",
            "manage_sessions": "Session management",
            "manage_models": "Model management"
        }
        
        # Public endpoints that don't require authentication
        self.public_endpoints: Set[str] = {
            "/",
            "/health",
            "/health/simple",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/redoc"
        }
        
        # Internal endpoints that require API key
        self.internal_endpoints: Set[str] = {
            "/services",
            "/models"
        }
    
    def generate_token(self, user_id: str, permissions: List[str], expires_in: int = 3600) -> str:
        """Generate a new authentication token."""
        token = secrets.token_urlsafe(32)
        expires_at = time.time() + expires_in
        
        auth_token = AuthToken(
            token=token,
            user_id=user_id,
            permissions=permissions,
            expires_at=expires_at,
            created_at=time.time(),
            last_used=time.time()
        )
        
        self.tokens[token] = auth_token
        
        logger.info(
            "Authentication token generated",
            user_id=user_id,
            permissions=permissions,
            expires_in=expires_in
        )
        
        return token
    
    def validate_token(self, token: str) -> Optional[AuthToken]:
        """Validate an authentication token."""
        if token not in self.tokens:
            return None
        
        auth_token = self.tokens[token]
        
        # Check if token is expired
        if auth_token.expires_at < time.time():
            del self.tokens[token]
            logger.warning("Expired token removed", token=token[:8] + "...")
            return None
        
        # Update last used time
        auth_token.last_used = time.time()
        
        return auth_token
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate an API key."""
        if api_key not in self.api_keys:
            return None
        
        key_info = self.api_keys[api_key]
        key_info["last_used"] = time.time()
        
        return key_info
    
    def check_permission(self, user_permissions: List[str], required_permission: str) -> bool:
        """Check if user has required permission."""
        # Admin permission grants all access
        if "admin" in user_permissions or "*" in user_permissions:
            return True
        
        return required_permission in user_permissions
    
    def get_endpoint_permission(self, method: str, path: str) -> Optional[str]:
        """Get required permission for an endpoint."""
        # Map endpoints to required permissions
        permission_map = {
            # URL management
            ("POST", "/api/url-input"): "write",
            ("GET", "/api/url-input"): "read",
            
            # Authentication
            ("POST", "/api/auth"): "write",
            ("GET", "/api/auth"): "read",
            
            # Scraping
            ("POST", "/api/scraper"): "scrape",
            ("GET", "/api/scraper"): "read",
            
            # Analysis
            ("POST", "/api/analyzer"): "analyze",
            ("GET", "/api/analyzer"): "read",
            
            # Clustering
            ("POST", "/api/clustering"): "cluster",
            ("GET", "/api/clustering"): "read",
            
            # Export
            ("POST", "/api/export"): "export",
            ("GET", "/api/export"): "read",
            
            # Session management
            ("POST", "/api/session"): "manage_sessions",
            ("GET", "/api/session"): "read",
            ("PUT", "/api/session"): "manage_sessions",
            ("DELETE", "/api/session"): "manage_sessions",
            
            # Model management
            ("POST", "/models"): "manage_models",
            ("GET", "/models"): "read",
        }
        
        # Check exact match first
        key = (method, path)
        if key in permission_map:
            return permission_map[key]
        
        # Check prefix matches
        for (perm_method, perm_path), permission in permission_map.items():
            if method == perm_method and path.startswith(perm_path):
                return permission
        
        # Default permission for API endpoints
        if path.startswith("/api/"):
            return "read" if method == "GET" else "write"
        
        return None
    
    def is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public."""
        return path in self.public_endpoints
    
    def is_internal_endpoint(self, path: str) -> bool:
        """Check if endpoint requires internal API key."""
        return any(path.startswith(endpoint) for endpoint in self.internal_endpoints)
    
    async def authenticate_request(self, request: Request) -> Optional[Dict[str, Any]]:
        """Authenticate incoming request."""
        path = request.url.path
        method = request.method
        
        # Skip authentication for public endpoints
        if self.is_public_endpoint(path):
            return {"type": "public", "permissions": ["read"]}
        
        # Check for API key (for internal services)
        api_key = request.headers.get("x-api-key")
        if api_key:
            key_info = self.validate_api_key(api_key)
            if key_info:
                return {
                    "type": "api_key",
                    "name": key_info["name"],
                    "permissions": key_info["permissions"]
                }
            else:
                logger.warning("Invalid API key", api_key=api_key[:8] + "...")
                raise HTTPException(status_code=401, detail="Invalid API key")
        
        # Check for Bearer token
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove "Bearer " prefix
            auth_token = self.validate_token(token)
            
            if auth_token:
                return {
                    "type": "token",
                    "user_id": auth_token.user_id,
                    "permissions": auth_token.permissions,
                    "token": auth_token
                }
            else:
                logger.warning("Invalid or expired token", token=token[:8] + "...")
                raise HTTPException(status_code=401, detail="Invalid or expired token")
        
        # For internal endpoints, require authentication
        if self.is_internal_endpoint(path):
            raise HTTPException(
                status_code=401,
                detail="Authentication required for internal endpoints"
            )
        
        # For API endpoints, check if authentication is required
        required_permission = self.get_endpoint_permission(method, path)
        if required_permission and required_permission != "read":
            raise HTTPException(
                status_code=401,
                detail="Authentication required"
            )
        
        # Allow unauthenticated read access to most endpoints
        return {"type": "anonymous", "permissions": ["read"]}
    
    async def authorize_request(self, request: Request, auth_info: Dict[str, Any]) -> bool:
        """Authorize request based on authentication info."""
        path = request.url.path
        method = request.method
        
        # Get required permission
        required_permission = self.get_endpoint_permission(method, path)
        
        # No permission required
        if not required_permission:
            return True
        
        # Check if user has required permission
        user_permissions = auth_info.get("permissions", [])
        
        if self.check_permission(user_permissions, required_permission):
            return True
        
        logger.warning(
            "Authorization failed",
            path=path,
            method=method,
            required_permission=required_permission,
            user_permissions=user_permissions,
            auth_type=auth_info.get("type")
        )
        
        raise HTTPException(
            status_code=403,
            detail=f"Insufficient permissions. Required: {required_permission}"
        )
    
    def create_session(self, user_id: str, metadata: Dict[str, Any] = None) -> str:
        """Create a new user session."""
        session_id = secrets.token_urlsafe(16)
        
        self.sessions[session_id] = {
            "user_id": user_id,
            "created_at": time.time(),
            "last_accessed": time.time(),
            "metadata": metadata or {}
        }
        
        logger.info("Session created", session_id=session_id, user_id=user_id)
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        session["last_accessed"] = time.time()
        
        return session
    
    def cleanup_expired_tokens(self):
        """Clean up expired tokens and sessions."""
        current_time = time.time()
        
        # Clean up expired tokens
        expired_tokens = [
            token for token, auth_token in self.tokens.items()
            if auth_token.expires_at < current_time
        ]
        
        for token in expired_tokens:
            del self.tokens[token]
        
        # Clean up old sessions (24 hours)
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if session["last_accessed"] < current_time - 86400
        ]
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        if expired_tokens or expired_sessions:
            logger.info(
                "Cleaned up expired auth data",
                expired_tokens=len(expired_tokens),
                expired_sessions=len(expired_sessions)
            )
    
    def get_auth_stats(self) -> Dict[str, Any]:
        """Get authentication statistics."""
        current_time = time.time()
        
        active_tokens = sum(
            1 for token in self.tokens.values()
            if token.expires_at > current_time
        )
        
        active_sessions = sum(
            1 for session in self.sessions.values()
            if session["last_accessed"] > current_time - 3600  # Active in last hour
        )
        
        return {
            "active_tokens": active_tokens,
            "total_tokens": len(self.tokens),
            "active_sessions": active_sessions,
            "total_sessions": len(self.sessions),
            "api_keys": len(self.api_keys)
        }