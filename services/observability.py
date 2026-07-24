"""Shared structured logging and request-ID propagation for backend services.

Emits JSON lines to stdout via the stdlib ``logging`` module only — no third
party dependencies. Each service:

1. calls :func:`configure_logging` once at import time,
2. installs :class:`RequestIDMiddleware` so every request accepts or mints an
   ``X-Request-ID``, binds it for the request, and echoes it in the response,
3. propagates the bound id on outbound service calls via
   :func:`request_id_headers`,
4. records cross-service seams with :func:`log_event`.

Never pass credentials, bearer tokens, or page-content bodies as fields; log
lengths or hashes instead. This module is copied into the backend-core,
ai-engine, and browser-engine images (all three build from the repo root), so
keep it dependency-free.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Optional

REQUEST_ID_HEADER = "X-Request-ID"
_MAX_REQUEST_ID_LEN = 128
_REQUEST_ID_ALLOWED = re.compile(r"[^A-Za-z0-9._\-]")

_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_service_name = "unknown"


def new_request_id() -> str:
    """Return a fresh request id."""
    return uuid.uuid4().hex


def _sanitize_request_id(raw: str) -> str:
    """Constrain an inbound id so it cannot inject headers or log lines."""
    cleaned = _REQUEST_ID_ALLOWED.sub("", raw.strip())[:_MAX_REQUEST_ID_LEN]
    return cleaned or new_request_id()


def set_request_id(request_id: Optional[str]):
    """Bind ``request_id`` for the current context; return the reset token."""
    return _request_id.set(request_id)


def reset_request_id(token) -> None:
    """Restore the request id to its previous value."""
    _request_id.reset(token)


def get_request_id() -> Optional[str]:
    """Return the request id bound to the current context, if any."""
    return _request_id.get()


def request_id_headers(request_id: Optional[str] = None) -> dict[str, str]:
    """Return ``{X-Request-ID: ...}`` for outbound calls, or ``{}`` when unbound."""
    rid = request_id or get_request_id()
    return {REQUEST_ID_HEADER: rid} if rid else {}


class _JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "service": getattr(record, "service", _service_name),
            "request_id": getattr(record, "request_id", None),
            "event": getattr(record, "event", record.getMessage()),
        }
        fields = getattr(record, "fields", None)
        if fields:
            payload.update(fields)
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(service: str) -> logging.Logger:
    """Configure a JSON stdout logger for ``service`` (idempotent)."""
    global _service_name
    _service_name = service
    logger = logging.getLogger(f"taborganizer.{service}")
    already = any(getattr(h, "_json_obs", False) for h in logger.handlers)
    if not already:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        handler._json_obs = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    return logger


def log_event(
    event: str,
    *,
    level: int = logging.INFO,
    request_id: Optional[str] = None,
    exc_info: bool = False,
    **fields,
) -> None:
    """Emit a structured record for a cross-service seam.

    ``fields`` are merged into the JSON object. Do not pass tokens,
    credentials, or raw page bodies — use lengths or hashes.
    """
    logger = logging.getLogger(f"taborganizer.{_service_name}")
    logger.log(
        level,
        event,
        extra={
            "event": event,
            "service": _service_name,
            "request_id": request_id or get_request_id(),
            "fields": fields,
        },
        exc_info=exc_info,
    )


class RequestIDMiddleware:
    """ASGI middleware that binds an ``X-Request-ID`` per request.

    Implemented as raw ASGI (not ``BaseHTTPMiddleware``) so the context var is
    set in the same task that runs the endpoint, and is reliably visible to
    handler code and outbound calls.
    """

    def __init__(self, app, service: Optional[str] = None):
        self.app = app
        self.service = service

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = None
        for name, value in scope.get("headers", []):
            if name == b"x-request-id":
                incoming = value.decode("latin-1")
                break
        request_id = _sanitize_request_id(incoming) if incoming else new_request_id()
        token = set_request_id(request_id)
        encoded = request_id.encode("latin-1")

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers = [h for h in headers if h[0].lower() != b"x-request-id"]
                headers.append((b"x-request-id", encoded))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            reset_request_id(token)
