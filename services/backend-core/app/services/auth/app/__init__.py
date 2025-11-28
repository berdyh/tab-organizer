"""Authentication service package."""

from .auth_queue import AuthenticationQueue
from .credential_store import SecureCredentialStore
from .detector import AuthenticationDetector
from .domain_mapper import DomainAuthMapper
from .interactive import InteractiveAuthenticator
from .logging import configure_logging
from .models import *  # noqa: F401,F403 - re-export models for convenience
from .oauth import OAuthFlowHandler
from .state import AuthServiceState, build_state

__all__ = [
    "AuthenticationQueue",
    "SecureCredentialStore",
    "AuthenticationDetector",
    "DomainAuthMapper",
    "InteractiveAuthenticator",
    "configure_logging",
    "OAuthFlowHandler",
    "AuthServiceState",
    "build_state",
]
