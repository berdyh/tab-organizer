"""Rate limiting middleware for API Gateway."""

import asyncio
import hashlib
import time
from collections import deque
from typing import Any, Dict, Optional

import structlog
from fastapi import HTTPException, Request

logger = structlog.get_logger()


class RateLimiter:
    """Token bucket rate limiter with sliding window support."""

    def __init__(self):
        # Rate limiting storage: client_id -> bucket info
        # Note: tokens will be initialized to burst_limit when first accessed
        self.buckets: Dict[str, Dict[str, Any]] = {}

        # Rate limiting rules
        self.rules = {
            "default": {
                "requests_per_minute": 60,
                "requests_per_hour": 1000,
                "burst_limit": 10,
                "block_duration": 300,  # 5 minutes
            },
            "authenticated": {
                "requests_per_minute": 120,
                "requests_per_hour": 5000,
                "burst_limit": 20,
                "block_duration": 60,  # 1 minute
            },
            "internal": {
                "requests_per_minute": 1000,
                "requests_per_hour": 50000,
                "burst_limit": 100,
                "block_duration": 0,  # No blocking for internal services
            },
        }

        # Cleanup task
        self.cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_started = False

    def start_cleanup_task(self) -> None:
        """Start background cleanup task."""
        try:
            if not self._cleanup_started and (
                self.cleanup_task is None or self.cleanup_task.done()
            ):
                self.cleanup_task = asyncio.create_task(self._cleanup_old_entries())
                self._cleanup_started = True
        except RuntimeError:  # pragma: no cover - depends on loop availability
            # No event loop running, will start later when needed
            pass

    async def _cleanup_old_entries(self) -> None:
        """Clean up old rate limiting entries."""
        while True:
            try:
                current_time = time.time()
                cutoff_time = current_time - 3600  # Keep 1 hour of history

                # Clean up old buckets
                expired_clients = []
                for client_id, bucket in list(self.buckets.items()):
                    # Remove old requests from sliding window
                    while bucket["requests"] and bucket["requests"][0] < cutoff_time:
                        bucket["requests"].popleft()

                    # Mark inactive clients for removal
                    if bucket["last_refill"] < cutoff_time and not bucket["requests"]:
                        expired_clients.append(client_id)

                # Remove expired clients
                for client_id in expired_clients:
                    del self.buckets[client_id]

                if expired_clients:
                    logger.debug(
                        "Cleaned up rate limiter entries", count=len(expired_clients)
                    )

                await asyncio.sleep(300)  # Cleanup every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Rate limiter cleanup error", error=str(exc))
                await asyncio.sleep(60)

    def get_client_id(self, request: Request) -> str:
        """Extract client identifier from request."""
        # Try to get authenticated user ID
        auth_header = request.headers.get("authorization")
        if auth_header:
            # Use hash of auth token as client ID
            return hashlib.sha256(auth_header.encode()).hexdigest()[:16]

        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"

        # Check for forwarded IP headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            client_ip = real_ip

        return f"ip:{client_ip}"

    def get_client_type(self, request: Request, client_id: str) -> str:
        """Determine client type for rate limiting rules."""
        # Check if it's an internal service
        user_agent = request.headers.get("user-agent", "")
        if "internal-service" in user_agent.lower():
            return "internal"

        # Check if authenticated
        if request.headers.get("authorization"):
            return "authenticated"

        return "default"

    def refill_tokens(self, bucket: Dict[str, Any], rule: Dict[str, Any]) -> None:
        """Refill tokens in the bucket based on time elapsed."""
        current_time = time.time()
        time_elapsed = current_time - bucket["last_refill"]

        # Calculate tokens to add (1 token per second for requests_per_minute rate)
        tokens_per_second = rule["requests_per_minute"] / 60.0
        tokens_to_add = int(time_elapsed * tokens_per_second)

        if tokens_to_add > 0:
            bucket["tokens"] = min(
                rule["burst_limit"], bucket["tokens"] + tokens_to_add
            )
            bucket["last_refill"] = current_time

    def check_sliding_window(self, bucket: Dict[str, Any], rule: Dict[str, Any]) -> bool:
        """Check sliding window rate limits."""
        current_time = time.time()

        # Clean old requests
        minute_ago = current_time - 60
        hour_ago = current_time - 3600

        while bucket["requests"] and bucket["requests"][0] < hour_ago:
            bucket["requests"].popleft()

        # Count requests in windows
        requests_last_minute = sum(
            1 for req_time in bucket["requests"] if req_time >= minute_ago
        )
        requests_last_hour = len(bucket["requests"])

        # Check limits
        if requests_last_minute >= rule["requests_per_minute"]:
            return False

        if requests_last_hour >= rule["requests_per_hour"]:
            return False

        return True

    async def check_rate_limit(self, request: Request) -> bool:
        """Check if request should be rate limited."""
        # Start cleanup task if not already started
        if not self._cleanup_started:
            self.start_cleanup_task()

        client_id = self.get_client_id(request)
        client_type = self.get_client_type(request, client_id)
        rule = self.rules[client_type]

        # Initialize bucket if it doesn't exist
        if client_id not in self.buckets:
            self.buckets[client_id] = {
                "tokens": rule["burst_limit"],  # Start with full burst capacity
                "last_refill": time.time(),
                "requests": deque(),  # For sliding window
                "blocked_until": 0,
            }

        bucket = self.buckets[client_id]
        current_time = time.time()

        # Check if client is currently blocked
        if bucket["blocked_until"] > current_time:
            remaining_block_time = int(bucket["blocked_until"] - current_time)
            logger.warning(
                "Rate limited client still blocked",
                client_id=client_id,
                remaining_seconds=remaining_block_time,
            )
            raise HTTPException(
                status_code=429,
                detail=f"Rate limited. Try again in {remaining_block_time} seconds.",
                headers={"Retry-After": str(remaining_block_time)},
            )

        # Refill tokens
        self.refill_tokens(bucket, rule)

        # Check sliding window limits
        if not self.check_sliding_window(bucket, rule):
            # Block client if they have a block duration
            if rule["block_duration"] > 0:
                bucket["blocked_until"] = current_time + rule["block_duration"]
                logger.warning(
                    "Client rate limited and blocked",
                    client_id=client_id,
                    client_type=client_type,
                    block_duration=rule["block_duration"],
                )

            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": "60"},
            )

        # Check token bucket
        if bucket["tokens"] <= 0:
            logger.warning(
                "Client rate limited - no tokens",
                client_id=client_id,
                client_type=client_type,
            )
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded - no tokens available",
                headers={"Retry-After": "1"},
            )

        # Consume token and record request
        bucket["tokens"] -= 1
        bucket["requests"].append(current_time)

        logger.debug(
            "Rate limit check passed",
            client_id=client_id,
            client_type=client_type,
            remaining_tokens=bucket["tokens"],
        )

        return True

    def get_rate_limit_info(self, request: Request) -> Dict[str, Any]:
        """Get current rate limit status for client."""
        client_id = self.get_client_id(request)
        client_type = self.get_client_type(request, client_id)
        rule = self.rules[client_type]

        # Initialize bucket if it doesn't exist
        if client_id not in self.buckets:
            self.buckets[client_id] = {
                "tokens": rule["burst_limit"],  # Start with full burst capacity
                "last_refill": time.time(),
                "requests": deque(),  # For sliding window
                "blocked_until": 0,
            }

        bucket = self.buckets[client_id]

        current_time = time.time()

        # Count recent requests
        minute_ago = current_time - 60
        hour_ago = current_time - 3600

        requests_last_minute = sum(
            1 for req_time in bucket["requests"] if req_time >= minute_ago
        )
        requests_last_hour = sum(
            1 for req_time in bucket["requests"] if req_time >= hour_ago
        )

        return {
            "client_id": client_id,
            "client_type": client_type,
            "limits": {
                "requests_per_minute": rule["requests_per_minute"],
                "requests_per_hour": rule["requests_per_hour"],
                "burst_limit": rule["burst_limit"],
            },
            "current_usage": {
                "requests_last_minute": requests_last_minute,
                "requests_last_hour": requests_last_hour,
                "available_tokens": bucket["tokens"],
            },
            "blocked_until": (
                bucket["blocked_until"]
                if bucket["blocked_until"] > current_time
                else None
            ),
        }

    async def close(self) -> None:
        """Clean up resources."""
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

