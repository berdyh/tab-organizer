"""Shared browser-origin (CORS) policy for the three FastAPI services.

All three services used to run ``allow_origins=["*"]`` together with
``allow_credentials=True``. That combination turned every unauthenticated read
endpoint into a cross-site exfiltration primitive: any page the user happened to
have open while the stack was running could ``fetch('http://localhost:8080/...')``,
walk the session list and pull back stored page CONTENT -- including tabs
imported from the user's logged-in browser session.

None of backend-core, ai-engine or browser-engine has a legitimate browser-side
caller. The Streamlit UI talks to them from the *server* side with the
``requests`` library (``services/web-ui/src/api/client.py``); the user's browser
only ever reaches Streamlit on :8089. The policy is therefore an explicit
allowlist of the Web UI origin, and credentials are never allowed.

``CORS_ALLOWED_ORIGINS`` (comma-separated) overrides the default for operators
who front the stack with a different UI origin. It is a deliberate,
operator-only escape hatch in the same spirit as
``SCRAPE_ALLOW_PRIVATE_NETWORKS``: setting it to ``*`` re-opens the hole.
"""

from __future__ import annotations

import logging
import os

from services.observability import log_event

CORS_ORIGINS_ENV = "CORS_ALLOWED_ORIGINS"

# The Streamlit Web UI, on the two loopback spellings a browser may use. Both
# are distinct origins to a browser, so both must be listed.
DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:8089",
    "http://127.0.0.1:8089",
)


def allowed_origins() -> list[str]:
    """Resolve the browser origins allowed to read these services' responses."""
    raw = os.getenv(CORS_ORIGINS_ENV, "")
    configured = [origin.strip().rstrip("/") for origin in raw.split(",")]
    configured = [origin for origin in configured if origin]
    if not configured:
        return list(DEFAULT_ALLOWED_ORIGINS)
    if "*" in configured:
        # Dropping allow_credentials blunts this but does not close it: the
        # exfiltration chain reads unauthenticated endpoints and never needed
        # credentials. Say so at startup rather than failing closed -- an
        # operator who typed "*" gets to keep it, but not silently.
        log_event(
            "cors.wildcard_configured",
            level=logging.WARNING,
            env_var=CORS_ORIGINS_ENV,
            detail=(
                "any website the user visits can read this service's "
                "responses, including captured page text"
            ),
        )
    return configured
