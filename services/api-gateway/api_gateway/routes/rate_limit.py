"""Rate limiting endpoints."""

from fastapi import APIRouter, Depends, Request

from ..dependencies import get_rate_limiter
from ..schemas import RateLimitInfo

router = APIRouter()


@router.get("/rate-limit/info", response_model=RateLimitInfo)
async def get_rate_limit_info(request: Request, rate_limiter=Depends(get_rate_limiter)):
    """Get current rate limit status."""
    rate_info = rate_limiter.get_rate_limit_info(request)
    return RateLimitInfo(
        client_id=rate_info["client_id"],
        client_type=rate_info["client_type"],
        limits=rate_info["limits"],
        current_usage=rate_info["current_usage"],
        blocked_until=rate_info["blocked_until"],
    )

