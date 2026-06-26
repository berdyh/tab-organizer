"""CLI regressions for host-run AI Engine routing."""

import argparse

from scripts import cli


def _host_ai_args(**overrides):
    values = {
        "provider": "codex_acp",
        "embedding_provider": "ollama",
        "llm_model": None,
        "embedding_model": None,
        "ollama_host": None,
        "claude_code_command": None,
        "codex_cli_command": None,
        "codex_acp_command": None,
        "host": "0.0.0.0",
        "port": 8090,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_ensure_host_ai_token_reuses_configured_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_ENGINE_API_TOKEN", "configured-token")
    monkeypatch.setattr(cli, "HOST_AI_TOKEN_FILE", tmp_path / "host-ai-token")

    assert cli.ensure_host_ai_token() == "configured-token"
    assert not (tmp_path / "host-ai-token").exists()


def test_host_ai_rewrites_docker_ollama_host_and_sets_token(monkeypatch):
    calls = []

    monkeypatch.setattr(cli, "load_env_file", lambda: None)
    monkeypatch.setattr(cli, "ensure_host_ai_token", lambda: "host-token")
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama:11434")
    monkeypatch.setattr(
        cli,
        "run_command",
        lambda cmd, env=None, **kwargs: calls.append({"cmd": cmd, "env": env}),
    )

    cli.cmd_host_ai(_host_ai_args())

    assert calls
    env = calls[0]["env"]
    assert env["AI_PROVIDER"] == "codex_acp"
    assert env["EMBEDDING_PROVIDER"] == "ollama"
    assert env["OLLAMA_HOST"] == "http://localhost:11434"
    assert env["AI_ENGINE_API_TOKEN"] == "host-token"
    assert env["BACKEND_CALLBACK_TOKEN"] == "host-token"


def test_start_host_ai_sets_container_url_token_and_disables_ai_container(monkeypatch):
    calls = []

    monkeypatch.setattr(cli, "load_env_file", lambda: None)
    monkeypatch.setattr(cli, "ensure_host_ai_token", lambda: "host-token")
    monkeypatch.setattr(
        cli,
        "docker_compose",
        lambda *args, profiles=None, env=None: calls.append(
            {"args": args, "profiles": profiles, "env": env}
        ),
    )

    cli.cmd_start(
        argparse.Namespace(
            dev=False,
            build=False,
            detach=True,
            host_ai=True,
            host_ai_url="http://host.docker.internal:8090",
        )
    )

    assert calls[0]["args"] == (
        "up",
        "-d",
        "--scale",
        "ai-engine=0",
    )
    assert calls[0]["profiles"] == ["default"]
    assert calls[0]["env"]["AI_ENGINE_URL"] == "http://host.docker.internal:8090"
    assert calls[0]["env"]["AI_ENGINE_API_TOKEN"] == "host-token"
    assert calls[0]["env"]["BACKEND_CALLBACK_TOKEN"] == "host-token"


def test_start_populates_service_tokens_for_default_stack(monkeypatch):
    calls = []

    monkeypatch.setattr(cli, "load_env_file", lambda: None)
    monkeypatch.setattr(cli, "ensure_host_ai_token", lambda: "service-token")
    monkeypatch.delenv("AI_ENGINE_URL", raising=False)
    monkeypatch.setattr(
        cli,
        "docker_compose",
        lambda *args, profiles=None, env=None: calls.append(
            {"args": args, "profiles": profiles, "env": env}
        ),
    )

    cli.cmd_start(
        argparse.Namespace(
            dev=False,
            build=False,
            detach=True,
            host_ai=False,
            host_ai_url="http://host.docker.internal:8090",
        )
    )

    assert calls[0]["args"] == ("up", "-d")
    assert calls[0]["profiles"] == ["default"]
    assert calls[0]["env"]["AI_ENGINE_API_TOKEN"] == "service-token"
    assert calls[0]["env"]["BACKEND_CALLBACK_TOKEN"] == "service-token"
    assert "AI_ENGINE_URL" not in calls[0]["env"]


def test_service_env_replaces_blank_env_tokens_from_dotenv(monkeypatch):
    monkeypatch.setenv("AI_ENGINE_API_TOKEN", "")
    monkeypatch.setenv("BACKEND_CALLBACK_TOKEN", "")
    monkeypatch.setattr(cli, "ensure_host_ai_token", lambda: "generated-token")

    env = cli.service_env_with_tokens()

    assert env["AI_ENGINE_API_TOKEN"] == "generated-token"
    assert env["BACKEND_CALLBACK_TOKEN"] == "generated-token"
