"""Authentication service entrypoint."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

if __package__ in (None, ""):
    # Allow running as a stand-alone script (`python services/auth/main.py`)
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from app.api import router as api_router  # type: ignore
    from app.auth_queue import AuthenticationQueue  # type: ignore
    from app.credential_store import SecureCredentialStore  # type: ignore
    from app.dependencies import get_state  # type: ignore
    from app.detector import AuthenticationDetector  # type: ignore
    from app.domain_mapper import DomainAuthMapper  # type: ignore
    from app.interactive import InteractiveAuthenticator  # type: ignore
    from app.logging import configure_logging  # type: ignore
    from app.models import (  # type: ignore
        AuthDetectionResponse,
        AuthSession,
        AuthTaskResponse,
        AuthenticationRequirement,
        AuthenticationTask,
        CredentialStoreRequest,
        DomainAuthMapping,
        InteractiveAuthRequest,
        OAuthAuthRequest,
        OAuthConfig,
        SessionRequest,
        URLAnalysisRequest,
    )
    from app.oauth import OAuthFlowHandler  # type: ignore
else:
    from .app.api import router as api_router
    from .app.auth_queue import AuthenticationQueue
    from .app.credential_store import SecureCredentialStore
    from .app.dependencies import get_state
    from .app.detector import AuthenticationDetector
    from .app.domain_mapper import DomainAuthMapper
    from .app.interactive import InteractiveAuthenticator
    from .app.logging import configure_logging
    from .app.models import (
        AuthDetectionResponse,
        AuthSession,
        AuthTaskResponse,
        AuthenticationRequirement,
        AuthenticationTask,
        CredentialStoreRequest,
        DomainAuthMapping,
        InteractiveAuthRequest,
        OAuthAuthRequest,
        OAuthConfig,
        SessionRequest,
        URLAnalysisRequest,
    )
    from .app.oauth import OAuthFlowHandler

from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait


configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown tasks for the FastAPI app."""
    state = get_state()  # Initializes shared components and starts the queue
    try:
        yield
    finally:
        state.auth_queue.stop_processing()


app = FastAPI(
    title="Authentication Service",
    description="Handles website authentication requirements and credential management",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(api_router)


__all__ = [
    "app",
    "AuthenticationDetector",
    "SecureCredentialStore",
    "DomainAuthMapper",
    "InteractiveAuthenticator",
    "OAuthFlowHandler",
    "AuthenticationQueue",
    "AuthenticationRequirement",
    "AuthenticationTask",
    "AuthSession",
    "OAuthConfig",
    "URLAnalysisRequest",
    "CredentialStoreRequest",
    "AuthDetectionResponse",
    "InteractiveAuthRequest",
    "OAuthAuthRequest",
    "SessionRequest",
    "AuthTaskResponse",
    "async_playwright",
    "Browser",
    "BrowserContext",
    "Page",
    "webdriver",
    "WebDriverWait",
]
