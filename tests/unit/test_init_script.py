"""Regression tests for the interactive init helper."""

import argparse

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
    ["deepseek", "claude_code", "codex_cli", "codex_acp", "openrouter", "anthropic"],
)
def test_claude_embedding_provider_rejects_llm_only_providers(provider):
    """Every LLM-only provider must be refused as an embedding provider.

    `openrouter` belongs in this list: it serves NO embedding models at all
    (verified against the live API 2026-08-04 -- none of the 338 catalogued
    models has an embedding modality, and the three embedding IDs previously
    listed for it did not exist). It was long assumed to be the cloud embedding
    fallback, so indexing failed even with a valid API key.
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
    assert "openrouter" not in message.split("Supported providers:")[-1]


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
    # that genuinely embeds -- openrouter serves no embedding models at all.
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
