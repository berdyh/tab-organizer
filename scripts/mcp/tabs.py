"""Python-callable tab tools backed by Backend Core.

The functions in this module are intentionally plain Python callables so a
stdio MCP adapter can import and expose them without requiring a runtime MCP
SDK in the project.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, TextIO


AGENT_TOKEN_ENV = "BACKEND_AGENT_API_TOKEN"
DEFAULT_BACKEND_URL = "http://localhost:8080"
DEFAULT_CDP_URL = "http://localhost:9222"


class BackendCoreError(RuntimeError):
    """Raised when Backend Core returns an error or cannot be reached."""


class BackendCoreConfigurationError(BackendCoreError):
    """Raised when required local Backend Core client config is missing."""


def _api_base_url(base_url: str | None = None) -> str:
    raw_base = (
        base_url
        or os.getenv("BACKEND_CORE_URL")
        or os.getenv("BACKEND_URL")
        or DEFAULT_BACKEND_URL
    ).strip()
    base = raw_base.rstrip("/")
    if base.endswith("/api/v1"):
        return base
    if base.endswith("/api"):
        return f"{base}/v1"
    return f"{base}/api/v1"


def _clean_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _require_non_empty(value: str, name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{name} is required")
    return cleaned


def _require_urls(urls: list[str]) -> list[str]:
    cleaned = [url.strip() for url in urls if url and url.strip()]
    if not cleaned:
        raise ValueError("at least one URL is required")
    return cleaned


def redact_configured_secrets(message: str) -> str:
    """Redact configured service tokens from user-visible text."""
    redacted = message
    for key in (
        AGENT_TOKEN_ENV,
        "AI_ENGINE_API_TOKEN",
        "BACKEND_CALLBACK_TOKEN",
        "BROWSER_ENGINE_API_TOKEN",
    ):
        token = os.getenv(key, "").strip()
        if token:
            redacted = redacted.replace(token, "<redacted>")
    return redacted


class BackendCoreClient:
    """Minimal JSON client for the Backend Core agent tab API."""

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float | None = None,
    ):
        self.base_url = _api_base_url(base_url)
        self.token = token if token is not None else os.getenv(AGENT_TOKEN_ENV, "")
        self.token = self.token.strip()
        self.timeout = float(timeout or os.getenv("BACKEND_AGENT_TIMEOUT", "30"))

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a JSON request to Backend Core and decode the JSON response."""
        headers = self._headers()
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(
            self._url(path),
            data=data,
            headers=headers,
            method=method.upper(),
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            detail = self._read_error_detail(error)
            raise BackendCoreError(
                f"Backend Core returned HTTP {error.code} for "
                f"{method.upper()} {path}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise BackendCoreError(
                f"Backend Core request failed for {method.upper()} {path}: "
                f"{error.reason}"
            ) from error

        if not body:
            return {}

        text = body.decode("utf-8")
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return {"content": text}
        if isinstance(decoded, dict):
            return decoded
        return {"data": decoded}

    def _headers(self) -> dict[str, str]:
        if not self.token:
            raise BackendCoreConfigurationError(f"{AGENT_TOKEN_ENV} is required")
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _read_error_detail(self, error: urllib.error.HTTPError) -> str:
        body = error.read()
        if not body:
            return error.reason or "empty response"
        text = body.decode("utf-8", errors="replace")
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(decoded, dict):
            return str(decoded.get("detail") or decoded.get("message") or decoded)
        return str(decoded)


def tab_import_from_browser(
    cdp_url: str = DEFAULT_CDP_URL,
    session_id: str | None = None,
    session_name: str | None = None,
) -> dict[str, Any]:
    """Start importing currently open browser tabs through Backend Core."""
    payload = _clean_payload(
        {
            "cdp_url": _require_non_empty(cdp_url, "cdp_url"),
            "session_id": session_id,
            "session_name": session_name,
        }
    )
    return BackendCoreClient().request("POST", "/tabs/import", payload)


def tab_import_status(job_id: str) -> dict[str, Any]:
    """Return status for a tab import job."""
    encoded_job_id = urllib.parse.quote(_require_non_empty(job_id, "job_id"), safe="")
    return BackendCoreClient().request("GET", f"/tabs/import/{encoded_job_id}")


def tab_search(
    query: str,
    session_id: str | None = None,
    limit: int = 10,
    mode: str = "hybrid",
) -> dict[str, Any]:
    """Search indexed tabs through Backend Core."""
    if limit < 1:
        raise ValueError("limit must be greater than zero")
    payload = _clean_payload(
        {
            "query": _require_non_empty(query, "query"),
            "session_id": session_id,
            "top_k": limit,
            "mode": _require_non_empty(mode, "mode"),
        }
    )
    return BackendCoreClient().request("POST", "/search", payload)


def tab_cluster(session_id: str) -> dict[str, Any]:
    """Cluster tabs for a Backend Core session."""
    return BackendCoreClient().request(
        "POST",
        "/cluster",
        {"session_id": _require_non_empty(session_id, "session_id")},
    )


def tab_open(
    urls: list[str] | None = None,
    session_id: str | None = None,
    cdp_url: str = DEFAULT_CDP_URL,
) -> dict[str, Any]:
    """Open URLs in the attached local browser through Backend Core."""
    if urls:
        cleaned_urls = _require_urls(urls)
    elif session_id:
        cleaned_urls = None
    else:
        raise ValueError("at least one URL or session_id is required")
    payload = _clean_payload(
        {
            "urls": cleaned_urls,
            "session_id": session_id,
            "cdp_url": _require_non_empty(cdp_url, "cdp_url"),
        }
    )
    return BackendCoreClient().request("POST", "/tabs/open", payload)


def tab_export(
    session_id: str,
    export_format: str = "markdown",
) -> dict[str, Any]:
    """Export organized tabs for a Backend Core session."""
    payload = {
        "session_id": _require_non_empty(session_id, "session_id"),
        "format": _require_non_empty(export_format, "export_format"),
    }
    return BackendCoreClient().request("POST", "/export", payload)


TOOL_FUNCTIONS: dict[str, Callable[..., dict[str, Any]]] = {
    "tab_import_from_browser": tab_import_from_browser,
    "tab_import_status": tab_import_status,
    "tab_search": tab_search,
    "tab_cluster": tab_cluster,
    "tab_open": tab_open,
    "tab_export": tab_export,
}


def handle_stdio_message(message: dict[str, Any]) -> dict[str, Any]:
    """Handle a single JSON stdio tool-call message."""
    tool_name = message.get("tool") or message.get("name") or message.get("method")
    arguments = message.get("arguments", message.get("params", {}))
    if tool_name not in TOOL_FUNCTIONS:
        raise ValueError(f"unknown tab tool: {tool_name}")
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be a JSON object")
    return {"result": TOOL_FUNCTIONS[tool_name](**arguments)}


def main(
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
) -> None:
    """Run a small JSON-lines stdio adapter for local MCP experiments."""
    for line in input_stream:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
            response = handle_stdio_message(message)
        except Exception as error:
            response = {"error": redact_configured_secrets(str(error))}
        output_stream.write(json.dumps(response, sort_keys=True) + "\n")
        output_stream.flush()


if __name__ == "__main__":
    main()
