"""Application middleware utilities."""

import time
from typing import Callable, Optional

import structlog
from fastapi import HTTPException, Request, Response

from .metrics import observe_request
from .state import GatewayState

logger = structlog.get_logger()


def _get_state(request: Request) -> Optional[GatewayState]:
    return getattr(request.app.state, "gateway_state", None)


async def request_metrics_middleware(
    request: Request, call_next: Callable[[Request], Response]
) -> Response:
    """Middleware that applies rate limiting, records metrics, and logs requests."""
    start_time = time.time()
    state = _get_state(request)

    try:
        # Apply rate limiting (except for health checks)
        if (
            state
            and state.rate_limiter
            and not request.url.path.startswith("/health")
        ):
            await state.rate_limiter.check_rate_limit(request)

        response = await call_next(request)

        duration = observe_request(
            request.method, request.url.path, response.status_code, start_time
        )

        # Add rate limit headers
        if (
            state
            and state.rate_limiter
            and not request.url.path.startswith("/health")
        ):
            try:
                rate_info = state.rate_limiter.get_rate_limit_info(request)
                response.headers["X-RateLimit-Limit"] = str(
                    rate_info["limits"]["requests_per_minute"]
                )
                response.headers["X-RateLimit-Remaining"] = str(
                    max(
                        0,
                        rate_info["limits"]["requests_per_minute"]
                        - rate_info["current_usage"]["requests_last_minute"],
                    )
                )
                response.headers["X-RateLimit-Reset"] = str(int(time.time() + 60))
            except Exception:  # pragma: no cover - defensive
                pass

        logger.info(
            "Request processed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration=duration,
            client_ip=request.client.host if request.client else None,
        )

        return response

    except HTTPException as exc:
        duration = observe_request(
            request.method, request.url.path, exc.status_code, start_time
        )
        logger.warning(
            "Request rejected",
            method=request.method,
            path=request.url.path,
            status_code=exc.status_code,
            detail=exc.detail,
            duration=duration,
            client_ip=request.client.host if request.client else None,
        )
        raise

