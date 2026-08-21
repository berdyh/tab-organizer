"""A configured bearer token must be a credential, not merely non-empty.

`ensure_service_token` treats an already-set env/.env value as the record that a
human chose it, and never clobbers it -- CLAUDE.md's "an already-set env var is
consent". That precedence is correct. What it lacked was any test of the value's
ADEQUACY: "non-empty" answered "is this present", never "is this a credential".

The gap was observed, not imagined. An author's `.env` carried
`BACKEND_CALLBACK_TOKEN=:` -- one colon guarding `POST /api/v1/ingest/v1`, the
content write path -- and it was accepted. It also broke SEC-27's redaction
audit, which asserts no configured token value appears in any response body: a
one-character token matches every JSON response containing a colon, so a
green suite reported a credential leak.

These are deployment-shape assertions about the tooling, so they live here and
NOT in the frozen black-box suite, which freezes service behaviour.
"""

import pytest

from scripts import cli


TOKEN_ENVS = [
    "AI_ENGINE_API_TOKEN",
    "BACKEND_CALLBACK_TOKEN",
    "BACKEND_AGENT_API_TOKEN",
    "BROWSER_ENGINE_API_TOKEN",
]


@pytest.mark.parametrize("env_name", TOKEN_ENVS)
def test_one_character_token_is_refused(env_name, monkeypatch):
    """The exact value that was found in a real .env."""
    monkeypatch.setenv(env_name, ":")

    with pytest.raises(SystemExit) as exit_info:
        cli.ensure_service_token(env_name)

    message = str(exit_info.value)
    assert "service_token_too_short" in message
    assert env_name in message
    # The refusal must say how to resolve it, not merely that it refused.
    assert "fix:" in message
    # It must never echo a credential back, even a bad one, but it may say how
    # long the value was.
    assert "1-character" in message


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_token_is_not_treated_as_configured(value, monkeypatch, tmp_path):
    """Blank means unset, which mints a token -- it is not a short-token error.

    Blanking the variable is the documented remediation for a bad token, so it
    has to lead to a freshly minted one rather than the refusal.
    """
    monkeypatch.setenv("BACKEND_CALLBACK_TOKEN", value)
    monkeypatch.setattr(cli, "_read_service_token_store", lambda: {})
    written = {}
    monkeypatch.setattr(cli, "_write_service_token_store", written.update)

    token = cli.ensure_service_token("BACKEND_CALLBACK_TOKEN")

    assert len(token) >= cli.MIN_SERVICE_TOKEN_LENGTH
    assert written["BACKEND_CALLBACK_TOKEN"] == token


def test_short_but_plausible_token_is_still_refused(monkeypatch):
    """The floor is a length, not a blocklist of obviously-silly values."""
    monkeypatch.setenv("BACKEND_AGENT_API_TOKEN", "hunter2")

    with pytest.raises(SystemExit) as exit_info:
        cli.ensure_service_token("BACKEND_AGENT_API_TOKEN")
    assert "service_token_too_short" in str(exit_info.value)


def test_adequate_configured_token_is_returned_unchanged(monkeypatch):
    """Consent is still honoured: a real token the operator chose is kept.

    The point of the check is to refuse values that are not credentials, never
    to start clobbering operator-chosen ones.
    """
    chosen = "a" * cli.MIN_SERVICE_TOKEN_LENGTH
    monkeypatch.setenv("BACKEND_CALLBACK_TOKEN", chosen)

    assert cli.ensure_service_token("BACKEND_CALLBACK_TOKEN") == chosen


def test_minted_tokens_clear_the_floor_the_check_enforces(monkeypatch):
    """Whatever the tooling mints must satisfy its own rule.

    A floor above the minted length would make a fresh install fail on its
    first run, which is the one regression this check could plausibly cause.
    """
    monkeypatch.delenv("BACKEND_CALLBACK_TOKEN", raising=False)
    monkeypatch.setattr(cli, "_read_service_token_store", lambda: {})
    monkeypatch.setattr(cli, "_write_service_token_store", lambda store: None)

    minted = cli.ensure_service_token("BACKEND_CALLBACK_TOKEN")

    assert len(minted) >= cli.MIN_SERVICE_TOKEN_LENGTH


def test_persisted_store_value_is_also_checked(monkeypatch):
    """A bad value must not slip in through the persisted store either.

    `data/service-tokens.json` is a file a human can edit, so it is a second
    door onto the same trust boundary.
    """
    monkeypatch.delenv("BROWSER_ENGINE_API_TOKEN", raising=False)
    monkeypatch.setattr(
        cli, "_read_service_token_store", lambda: {"BROWSER_ENGINE_API_TOKEN": ":"}
    )
    monkeypatch.setattr(cli, "_write_service_token_store", lambda store: None)

    with pytest.raises(SystemExit) as exit_info:
        cli.ensure_service_token("BROWSER_ENGINE_API_TOKEN")
    assert "service_token_too_short" in str(exit_info.value)
