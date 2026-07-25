"""Authentication module."""

from .detector import AuthDetectionResult, AuthDetector
from .queue import AuthQueue, AuthRequest, CredentialStore, CredentialStoreError

__all__ = [
    "AuthDetector",
    "AuthDetectionResult",
    "AuthQueue",
    "AuthRequest",
    "CredentialStore",
    "CredentialStoreError",
]
