"""Regression tests for `scripts/cli.py check-provider` (SPEC-provider-routing.md R1).

Before this fix, an unset AI_PROVIDER made `check-provider` silently fall back
to "openrouter" -- printing an availability line for a provider nobody chose,
and with `--generate` issuing a real billed OpenRouter request nobody
selected. That is the exact silent-default failure R1 forbids, so this must
fail closed instead, the same way `cmd_host_ai` does.
"""

import argparse

import pytest

from scripts import cli


def _check_args(**overrides):
    values = {
        "provider": None,
        "model": None,
        "generate": False,
        "prompt": "Reply with OK.",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_cmd_check_provider_fails_closed_when_ai_provider_not_set(monkeypatch):
    monkeypatch.setattr(cli, "load_env_file", lambda: None)
    monkeypatch.delenv("AI_PROVIDER", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_check_provider(_check_args())

    message = str(exc_info.value)
    assert "provider_not_selected" in message
    assert "AI_PROVIDER" in message
    assert "configure-provider" in message


def test_cmd_check_provider_never_reaches_generate_when_provider_not_set(monkeypatch):
    """Even with --generate, refusing must happen before any provider (billed
    or not) is ever constructed or called."""
    monkeypatch.setattr(cli, "load_env_file", lambda: None)
    monkeypatch.delenv("AI_PROVIDER", raising=False)

    called = []
    monkeypatch.setattr(
        "services.ai_engine.app.core.llm_client.LLMClient.generate",
        lambda self, *a, **k: called.append(True),
    )

    with pytest.raises(SystemExit):
        cli.cmd_check_provider(_check_args(generate=True))

    assert not called


def test_cmd_check_provider_accepts_explicit_provider_flag(monkeypatch):
    """The legitimate route: an explicit --provider IS the deliberate choice
    (R3) and must keep working with no AI_PROVIDER in the environment."""
    monkeypatch.setattr(cli, "load_env_file", lambda: None)
    monkeypatch.delenv("AI_PROVIDER", raising=False)

    provider = cli._require_explicit_provider(
        "claude_code", "AI_PROVIDER", "LLM", "--provider"
    )
    assert provider == "claude_code"


def test_cmd_check_provider_honors_ai_provider_env_as_consent(monkeypatch):
    """AI_PROVIDER already set in the environment (or .env, via load_env_file)
    IS the record of consent (R3) -- it must not be treated the same as an
    unset value."""
    monkeypatch.setattr(cli, "load_env_file", lambda: None)
    monkeypatch.setenv("AI_PROVIDER", "ollama")

    provider = cli._require_explicit_provider(None, "AI_PROVIDER", "LLM", "--provider")
    assert provider == "ollama"
