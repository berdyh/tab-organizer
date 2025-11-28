"""Application state for the authentication service."""

from dataclasses import dataclass

from .auth_queue import AuthenticationQueue
from .credential_store import SecureCredentialStore
from .detector import AuthenticationDetector
from .domain_mapper import DomainAuthMapper
from .interactive import InteractiveAuthenticator
from .oauth import OAuthFlowHandler


@dataclass
class AuthServiceState:
    """Container for shared auth service components."""

    detector: AuthenticationDetector
    credential_store: SecureCredentialStore
    domain_mapper: DomainAuthMapper
    interactive_auth: InteractiveAuthenticator
    oauth_handler: OAuthFlowHandler
    auth_queue: AuthenticationQueue


def build_state() -> AuthServiceState:
    """Build the shared application state."""
    detector = AuthenticationDetector()
    credential_store = SecureCredentialStore()
    domain_mapper = DomainAuthMapper()
    interactive_auth = InteractiveAuthenticator()
    oauth_handler = OAuthFlowHandler()
    auth_queue = AuthenticationQueue(authenticator=interactive_auth)

    auth_queue.start_processing()

    return AuthServiceState(
        detector=detector,
        credential_store=credential_store,
        domain_mapper=domain_mapper,
        interactive_auth=interactive_auth,
        oauth_handler=oauth_handler,
        auth_queue=auth_queue,
    )


__all__ = ["AuthServiceState", "build_state"]
