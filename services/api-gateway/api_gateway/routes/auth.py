"""Authentication endpoints."""

import time
from typing import Any, Dict

import structlog
from fastapi import APIRouter, Depends, HTTPException

from ..core import AuthMiddleware
from ..dependencies import authenticate, get_auth_middleware
from ..schemas import AuthRequest, AuthResponse

logger = structlog.get_logger()
router = APIRouter()


@router.post("/auth/login", response_model=AuthResponse)
async def login(auth_request: AuthRequest, auth_middleware: AuthMiddleware = Depends(get_auth_middleware)):
    """Authenticate user and return access token."""
    if auth_request.username == "admin" and auth_request.password == "admin":
        permissions = ["admin", "*"]
    elif auth_request.username == "user" and auth_request.password == "user":
        permissions = [
            "read",
            "write",
            "scrape",
            "analyze",
            "cluster",
            "export",
        ]
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = auth_middleware.generate_token(auth_request.username, permissions, expires_in=3600)

    return AuthResponse(token=token, expires_in=3600, permissions=permissions)


@router.post("/auth/logout")
async def logout(
    auth_info: Dict[str, Any] = Depends(authenticate),
    auth_middleware: AuthMiddleware = Depends(get_auth_middleware),
):
    """Logout user and invalidate token."""
    if auth_info.get("type") == "token" and "token" in auth_info:
        token_obj = auth_info["token"]
        if token_obj.token in auth_middleware.tokens:
            del auth_middleware.tokens[token_obj.token]
            logger.info("User logged out", user_id=token_obj.user_id)

    return {"message": "Logged out successfully"}


@router.get("/auth/me")
async def get_current_user(auth_info: Dict[str, Any] = Depends(authenticate)):
    """Get current user information."""
    return {
        "type": auth_info.get("type"),
        "user_id": auth_info.get("user_id"),
        "permissions": auth_info.get("permissions", []),
        "timestamp": time.time(),
    }


@router.get("/admin/auth/stats")
async def get_auth_stats(
    auth_info: Dict[str, Any] = Depends(authenticate),
    auth_middleware: AuthMiddleware = Depends(get_auth_middleware),
):
    """Get authentication statistics (admin only)."""
    if not auth_middleware.check_permission(auth_info.get("permissions", []), "admin"):
        raise HTTPException(status_code=403, detail="Admin access required")

    return auth_middleware.get_auth_stats()

