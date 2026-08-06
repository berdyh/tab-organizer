"""Regression tests for the interactive init helper."""

import argparse
from pathlib import Path

import pytest

from scripts import init


def test_configure_ollama_writes_runtime_env_names(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "AI_PROVIDER=openrouter",
                "EMBEDDING_PROVIDER=openrouter",
                "LLM_MODEL=",
                "EMBEDDING_MODEL=",
                "EMBEDDING_DIMENSIONS=1024",
                "OLLAMA_HOST=",
            ]
        )
        + "\n"
    )
    monkeypatch.setattr(init, "ENV_FILE", env_file)

    args = argparse.Namespace(
        ollama_llm="llama3.2:3b",
        ollama_embedding="nomic-embed-text",
        ollama_mode="docker",
        profile="none",
    )

    init.configure_ollama(args)

    env_text = env_file.read_text()
    assert "AI_PROVIDER=ollama" in env_text
    assert "EMBEDDING_PROVIDER=ollama" in env_text
    assert "OLLAMA_HOST=http://ollama:11434" in env_text
    assert "EMBEDDING_MODEL=nomic-embed-text" in env_text
    assert "EMBEDDING_DIMENSIONS=" in env_text
    assert "EMBEDDING_DIMENSIONS=1024" not in env_text
    assert "LLM_PROVIDER=" not in env_text
    assert "OLLAMA_URL=" not in env_text


def test_claude_embedding_provider_argument_is_supported():
    args = init.parse_args(
        [
            "--provider",
            "claude",
            "--claude-embedding-provider",
            "openai",
        ]
    )

    assert args.claude_embedding_provider == "openai"


def test_subscription_provider_arguments_are_supported():
    args = init.parse_args(
        [
            "--provider",
            "codex_acp",
            "--subscription-embedding-provider",
            "ollama",
            "--codex-acp-command",
            "/usr/local/bin/acpx",
        ]
    )

    assert args.provider == "codex_acp"
    assert args.subscription_embedding_provider == "ollama"
    assert args.codex_acp_command == "/usr/local/bin/acpx"


@pytest.mark.parametrize(
    "provider",
    ["deepseek", "claude_code", "codex_cli", "codex_acp", "anthropic"],
)
def test_claude_embedding_provider_rejects_llm_only_providers(provider):
    """Every LLM-only provider must be refused as an embedding provider.

    `openrouter` was in this list until 2026-08-05 and has been REMOVED, not
    because the rule changed but because it never belonged: it does serve
    embeddings (POST /v1/embeddings, verified by live call -- see the
    correction note in config/ai_models.yaml). The earlier entry cited a scan
    of openrouter's /v1/models chat listing, which cannot see that surface.

    The five that remain are genuinely LLM-only: none has an embedding adapter
    in `services/ai-engine/app/providers/`.
    """
    args = argparse.Namespace(
        claude_llm="claude-3-5-sonnet-latest",
        claude_embedding_provider=provider,
        claude_embedding=None,
        anthropic_key=None,
    )

    with pytest.raises(SystemExit) as exc_info:
        init.configure_claude(args)

    message = str(exc_info.value)
    assert provider in message
    assert "does not support embeddings" in message
    # The message lists what actually works, derived from the catalog. Assert
    # on a real embedding provider rather than a hardcoded name, so this test
    # tracks the catalog instead of freezing a stale belief about it.
    assert "ollama" in message
    # The refused provider must not appear in the list of what *does* work.
    # This used to assert on `openrouter` by name, which quietly became wrong
    # when the catalog was corrected; asserting on the provider under test
    # tracks the catalog instead of freezing a belief about one provider.
    assert provider not in message.split("Supported providers:")[-1]


def test_configure_claude_clears_stale_embedding_dimensions(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "AI_PROVIDER=ollama",
                "EMBEDDING_PROVIDER=ollama",
                "LLM_MODEL=llama3.2:3b",
                "EMBEDDING_MODEL=nomic-embed-text",
                "EMBEDDING_DIMENSIONS=768",
            ]
        )
        + "\n"
    )
    monkeypatch.setattr(init, "ENV_FILE", env_file)

    # Switch to an embedding model with DIFFERENT dimensions than the 768 above,
    # so a stale value would be caught. bge-m3 is 1024. The provider must be one
    # that genuinely embeds; ollama is the local default.
    args = argparse.Namespace(
        claude_llm="claude-3-5-sonnet-latest",
        claude_embedding_provider="ollama",
        claude_embedding="bge-m3",
        anthropic_key=None,
    )

    init.configure_claude(args)

    env_text = env_file.read_text()
    assert "AI_PROVIDER=anthropic" in env_text
    assert "EMBEDDING_PROVIDER=ollama" in env_text
    assert "EMBEDDING_MODEL=bge-m3" in env_text
    # The stale 768 from nomic-embed-text must not survive the switch to a
    # 1024-dim model: a dimension that disagrees with the model makes ai-engine
    # refuse to write to LanceDB, and that surfaces at index time, after the
    # scrape has already run.
    assert "EMBEDDING_DIMENSIONS=768" not in env_text
    # It is cleared to BLANK rather than rewritten to 1024, deliberately: a
    # blank value resolves from the model catalog at runtime, so the pair can
    # never drift apart. Pinning a number here is what allows them to disagree.
    assert "EMBEDDING_DIMENSIONS=\n" in env_text or env_text.rstrip().endswith(
        "EMBEDDING_DIMENSIONS="
    )


def test_main_refuses_to_auto_select_provider_when_noninteractive(tmp_path, monkeypatch):
    """A non-interactive `init.py` run with no --provider must never silently
    write AI_PROVIDER=ollama. That line is later read (R1/R3) as proof a human
    deliberately chose it; an unattended default forges that record just like
    the tty-gate in cli.py's configure-provider is written to prevent.
    """
    assert not init.sys.stdin.isatty()  # sanity: this is exactly the scenario under test

    env_file = tmp_path / ".env"
    template_file = tmp_path / ".env.example"
    template_file.write_text("AI_PROVIDER=\nEMBEDDING_PROVIDER=\n")
    monkeypatch.setattr(init, "ENV_FILE", env_file)
    monkeypatch.setattr(init, "ENV_TEMPLATE", template_file)
    monkeypatch.setattr(init, "require_docker", lambda: None)
    monkeypatch.setattr(init, "ensure_logs_dir", lambda: None)

    with pytest.raises(SystemExit) as exc_info:
        init.main([])

    message = str(exc_info.value)
    assert "will not choose a provider on your behalf" in message
    assert "--provider" in message
    assert "AI_PROVIDER=ollama" not in env_file.read_text()


def test_update_env_vars_applies_every_key_in_one_pass(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("AI_PROVIDER=openrouter\nEMBEDDING_PROVIDER=openrouter\n")
    monkeypatch.setattr(init, "ENV_FILE", env_file)

    init.update_env_vars(
        {
            "AI_PROVIDER": "claude_code",
            "LLM_MODEL": "sonnet",
            "EMBEDDING_PROVIDER": "ollama",
        }
    )

    env_text = env_file.read_text()
    assert "AI_PROVIDER=claude_code" in env_text
    assert "LLM_MODEL=sonnet" in env_text
    assert "EMBEDDING_PROVIDER=ollama" in env_text


def test_update_env_vars_never_leaves_a_partial_file_on_write_failure(tmp_path, monkeypatch):
    """`cli.py configure-provider` writes AI_PROVIDER/LLM_MODEL/EMBEDDING_PROVIDER/
    EMBEDDING_MODEL/EMBEDDING_DIMENSIONS together through this function.
    scripts/MODULE.md treats a partially-written .env as a hard constraint to
    avoid, so an IO failure partway through the write must leave the ORIGINAL
    file completely untouched, never a mix of old and new keys.
    """
    env_file = tmp_path / ".env"
    original = "AI_PROVIDER=openrouter\nEMBEDDING_PROVIDER=openrouter\n"
    env_file.write_text(original)
    monkeypatch.setattr(init, "ENV_FILE", env_file)

    real_write_text = Path.write_text

    def failing_write_text(self, *args, **kwargs):
        if self.name.endswith(".tmp"):
            raise OSError("simulated disk full")
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", failing_write_text)

    with pytest.raises(OSError):
        init.update_env_vars({"AI_PROVIDER": "claude_code", "EMBEDDING_PROVIDER": "ollama"})

    assert env_file.read_text() == original


def test_configure_codex_acp_writes_subscription_runtime_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "AI_PROVIDER=openrouter",
                "EMBEDDING_PROVIDER=openrouter",
                "LLM_MODEL=openai/gpt-4o-mini",
                "EMBEDDING_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2:free",
                "EMBEDDING_DIMENSIONS=1024",
                "CODEX_ACP_COMMAND=acpx",
            ]
        )
        + "\n"
    )
    monkeypatch.setattr(init, "ENV_FILE", env_file)

    args = argparse.Namespace(
        subscription_llm="codex-acp-default",
        subscription_embedding_provider="ollama",
        subscription_embedding="nomic-embed-text",
        claude_code_command=None,
        codex_cli_command=None,
        codex_acp_command="/usr/local/bin/acpx",
    )

    init.configure_subscription_cli(args, "codex_acp")

    env_text = env_file.read_text()
    assert "AI_PROVIDER=codex_acp" in env_text
    assert "EMBEDDING_PROVIDER=ollama" in env_text
    assert "LLM_MODEL=codex-acp-default" in env_text
    assert "EMBEDDING_MODEL=nomic-embed-text" in env_text
    assert "EMBEDDING_DIMENSIONS=" in env_text
    assert "EMBEDDING_DIMENSIONS=1024" not in env_text
    assert "CODEX_ACP_COMMAND=/usr/local/bin/acpx" in env_text
