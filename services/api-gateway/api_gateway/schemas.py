"""Shared Pydantic models used by API routes."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    timestamp: float
    services: Dict[str, Dict[str, Any]]
    uptime: float


class ServiceStatus(BaseModel):
    name: str
    status: str
    url: str
    last_check: float
    response_time: Optional[float] = None
    error: Optional[str] = None


class AuthRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    token: str
    expires_in: int
    permissions: List[str]


class RateLimitInfo(BaseModel):
    client_id: str
    client_type: str
    limits: Dict[str, int]
    current_usage: Dict[str, int]
    blocked_until: Optional[float] = None

