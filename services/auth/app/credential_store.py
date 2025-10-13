"""Secure credential storage utilities."""

import base64
import json
from typing import Any, Dict, List, Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import structlog


logger = structlog.get_logger()


class SecureCredentialStore:
    """Handles secure storage and retrieval of authentication credentials."""

    def __init__(self, master_password: str = "default_master_key") -> None:
        self.master_password = master_password.encode()
        self._fernet = self._create_fernet_key()
        self.credentials_store: Dict[str, bytes] = {}

    def _create_fernet_key(self) -> Fernet:
        """Create Fernet encryption key from master password."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"stable_salt_for_demo",
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_password))
        return Fernet(key)

    def store_credentials(self, domain: str, credentials: Dict[str, Any]) -> bool:
        """Store encrypted credentials for a domain."""
        try:
            credentials_json = json.dumps(credentials)
            encrypted_credentials = self._fernet.encrypt(credentials_json.encode())
            self.credentials_store[domain] = encrypted_credentials
            logger.info("Credentials stored securely", domain=domain)
            return True
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to store credentials", domain=domain, error=str(exc))
            return False

    def retrieve_credentials(self, domain: str) -> Optional[Dict[str, Any]]:
        """Retrieve and decrypt credentials for a domain."""
        try:
            if domain not in self.credentials_store:
                return None

            encrypted_credentials = self.credentials_store[domain]
            decrypted_json = self._fernet.decrypt(encrypted_credentials).decode()
            credentials = json.loads(decrypted_json)
            logger.info("Credentials retrieved", domain=domain)
            return credentials
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to retrieve credentials", domain=domain, error=str(exc))
            return None

    def delete_credentials(self, domain: str) -> bool:
        """Delete stored credentials for a domain."""
        try:
            if domain in self.credentials_store:
                del self.credentials_store[domain]
                logger.info("Credentials deleted", domain=domain)
                return True
            return False
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to delete credentials", domain=domain, error=str(exc))
            return False

    def list_stored_domains(self) -> List[str]:
        """List all domains with stored credentials."""
        return list(self.credentials_store.keys())


__all__ = ["SecureCredentialStore"]
