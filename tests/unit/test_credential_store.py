"""Unit tests for fail-closed credential storage (browser-engine auth)."""

import sys

import pytest

sys.path.insert(0, "/app")

from cryptography.fernet import Fernet

from services.browser_engine.app.auth.queue import (
    AuthQueue,
    CredentialStore,
    CredentialStoreError,
)


def _no_keyring(monkeypatch):
    """Force the keyring path to be unavailable in-process."""
    monkeypatch.setattr(CredentialStore, "_load_key_from_keyring", lambda self: None)


class TestCredentialStoreFailClosed:
    def test_fail_closed_when_no_key_and_no_keyring(self, monkeypatch):
        monkeypatch.delenv("CREDENTIAL_ENCRYPTION_KEY", raising=False)
        _no_keyring(monkeypatch)

        store = CredentialStore()

        assert store.is_ready is False
        with pytest.raises(CredentialStoreError) as exc_info:
            store.store("example.com", "form", {"username": "a", "password": "b"})

        error = exc_info.value
        assert error.code == "credential_store_unconfigured"
        assert error.cause
        assert error.fix
        assert set(error.to_dict()) == {"code", "cause", "fix"}
        # Nothing was persisted.
        assert store.has_credentials("example.com") is False

    def test_invalid_key_fails_closed(self, monkeypatch):
        monkeypatch.delenv("CREDENTIAL_ENCRYPTION_KEY", raising=False)
        _no_keyring(monkeypatch)

        store = CredentialStore(encryption_key="not-a-valid-fernet-key")

        assert store.is_ready is False
        with pytest.raises(CredentialStoreError) as exc_info:
            store.store("example.com", "form", {"x": "y"})
        assert exc_info.value.code == "credential_key_invalid"

    def test_round_trip_with_explicit_key(self, monkeypatch):
        _no_keyring(monkeypatch)
        key = Fernet.generate_key().decode()

        store = CredentialStore(encryption_key=key)

        assert store.is_ready is True
        creds = {"username": "alice", "password": "s3cret"}
        store.store("example.com", "form", creds)
        assert store.has_credentials("example.com") is True
        assert store.retrieve("example.com") == creds

    def test_round_trip_with_env_key(self, monkeypatch):
        _no_keyring(monkeypatch)
        key = Fernet.generate_key().decode()
        monkeypatch.setenv("CREDENTIAL_ENCRYPTION_KEY", key)

        store = CredentialStore()

        assert store.is_ready is True
        store.store("example.com", "form", {"a": "b"})
        assert store.retrieve("example.com") == {"a": "b"}

    def test_keyring_key_enables_storage(self, monkeypatch):
        monkeypatch.delenv("CREDENTIAL_ENCRYPTION_KEY", raising=False)
        key = Fernet.generate_key().decode()
        monkeypatch.setattr(CredentialStore, "_load_key_from_keyring", lambda self: key)

        store = CredentialStore()

        assert store.is_ready is True
        store.store("example.com", "form", {"a": "b"})
        assert store.retrieve("example.com") == {"a": "b"}


class TestAuthQueueFailClosed:
    @pytest.mark.asyncio
    async def test_provide_credentials_surfaces_fail_closed(self, monkeypatch):
        monkeypatch.delenv("CREDENTIAL_ENCRYPTION_KEY", raising=False)
        _no_keyring(monkeypatch)

        queue = AuthQueue()
        await queue.request_auth("https://example.com/login", "form")

        with pytest.raises(CredentialStoreError):
            await queue.provide_credentials("example.com", {"username": "a"})

        # The pending request is preserved — no partial/corrupt state.
        assert queue.get_pending_count() == 1

    @pytest.mark.asyncio
    async def test_provide_credentials_round_trip(self, monkeypatch):
        _no_keyring(monkeypatch)
        key = Fernet.generate_key().decode()
        monkeypatch.setenv("CREDENTIAL_ENCRYPTION_KEY", key)

        queue = AuthQueue()
        await queue.request_auth("https://example.com/login", "form")

        stored = await queue.provide_credentials(
            "example.com", {"username": "alice", "password": "s3cret"}
        )

        assert stored is True
        assert queue.get_pending_count() == 0
        assert queue.get_credentials("https://example.com/login") == {
            "username": "alice",
            "password": "s3cret",
        }
