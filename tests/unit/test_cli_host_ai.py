"""CLI regressions for host-run AI Engine routing."""

import argparse
import json

from scripts import cli
from scripts.mcp import tabs as mcp_tabs


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
    assert calls[0]["env"]["BACKEND_AGENT_API_TOKEN"] == "host-token"


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
    assert calls[0]["env"]["BACKEND_AGENT_API_TOKEN"] == "service-token"
    assert "AI_ENGINE_URL" not in calls[0]["env"]


def test_service_env_replaces_blank_env_tokens_from_dotenv(monkeypatch):
    monkeypatch.setenv("AI_ENGINE_API_TOKEN", "")
    monkeypatch.setenv("BACKEND_CALLBACK_TOKEN", "")
    monkeypatch.setenv("BACKEND_AGENT_API_TOKEN", "")
    monkeypatch.setattr(cli, "ensure_host_ai_token", lambda: "generated-token")

    env = cli.service_env_with_tokens()

    assert env["AI_ENGINE_API_TOKEN"] == "generated-token"
    assert env["BACKEND_CALLBACK_TOKEN"] == "generated-token"
    assert env["BACKEND_AGENT_API_TOKEN"] == "generated-token"


def test_backend_core_client_sends_agent_token_without_body_leak(monkeypatch):
    calls = []

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"status": "ok"}'

    def fake_urlopen(request, timeout):
        calls.append({"request": request, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr(mcp_tabs.urllib.request, "urlopen", fake_urlopen)

    client = mcp_tabs.BackendCoreClient(
        "http://backend.test",
        "agent-token-value",
        12.0,
    )

    result = client.request(
        "POST",
        "/search",
        payload={"query": "remote tabs", "limit": 5},
    )

    assert result == {"status": "ok"}
    assert calls[0]["timeout"] == 12.0
    request = calls[0]["request"]
    assert request.full_url == "http://backend.test/api/v1/search"
    assert request.get_header("Authorization") == "Bearer agent-token-value"
    assert request.get_header("Content-type") == "application/json"
    assert b"agent-token-value" not in request.data


def test_mcp_tab_tool_wrappers_call_backend_core(monkeypatch):
    calls = []

    class FakeClient:
        def request(self, method, path, payload=None):
            calls.append({"method": method, "path": path, "payload": payload})
            return {"method": method, "path": path}

    monkeypatch.setattr(mcp_tabs, "BackendCoreClient", FakeClient)

    assert mcp_tabs.tab_import_from_browser(
        cdp_url="http://localhost:9222",
        session_id="sess_1",
        session_name="Live Browser",
    ) == {"method": "POST", "path": "/tabs/import"}
    assert mcp_tabs.tab_import_status("job_1") == {
        "method": "GET",
        "path": "/tabs/import/job_1",
    }
    assert mcp_tabs.tab_search(
        "remote tabs",
        session_id="sess_1",
        limit=3,
        mode="keyword",
    ) == {
        "method": "POST",
        "path": "/search",
    }
    assert mcp_tabs.tab_cluster("sess_1") == {
        "method": "POST",
        "path": "/cluster",
    }
    assert mcp_tabs.tab_open(
        ["https://example.com"], cdp_url="http://localhost:9222"
    ) == {"method": "POST", "path": "/tabs/open"}
    assert mcp_tabs.tab_export("sess_1", export_format="markdown") == {
        "method": "POST",
        "path": "/export",
    }

    assert calls == [
        {
            "method": "POST",
            "path": "/tabs/import",
            "payload": {
                "cdp_url": "http://localhost:9222",
                "session_id": "sess_1",
                "session_name": "Live Browser",
            },
        },
        {"method": "GET", "path": "/tabs/import/job_1", "payload": None},
        {
            "method": "POST",
            "path": "/search",
            "payload": {
                "query": "remote tabs",
                "session_id": "sess_1",
                "top_k": 3,
                "mode": "keyword",
            },
        },
        {"method": "POST", "path": "/cluster", "payload": {"session_id": "sess_1"}},
        {
            "method": "POST",
            "path": "/tabs/open",
            "payload": {
                "urls": ["https://example.com"],
                "cdp_url": "http://localhost:9222",
            },
        },
        {
            "method": "POST",
            "path": "/export",
            "payload": {"session_id": "sess_1", "format": "markdown"},
        },
    ]


def test_tabs_cli_subcommands_dispatch_to_mcp_wrappers(monkeypatch, capsys):
    calls = []

    def record(name, result):
        def wrapper(*args, **kwargs):
            calls.append({"name": name, "args": args, "kwargs": kwargs})
            return result

        return wrapper

    monkeypatch.setenv("BACKEND_AGENT_API_TOKEN", "agent-token-value")
    monkeypatch.setattr(
        cli.mcp_tabs,
        "tab_import_from_browser",
        record("import", {"job_id": "job_1", "status": "queued"}),
    )
    monkeypatch.setattr(
        cli.mcp_tabs,
        "tab_import_status",
        record("status", {"job_id": "job_1", "status": "running"}),
    )
    monkeypatch.setattr(
        cli.mcp_tabs,
        "tab_search",
        record("search", {"results": [{"url": "https://example.com"}]}),
    )
    monkeypatch.setattr(
        cli.mcp_tabs,
        "tab_cluster",
        record("cluster", {"clusters": [{"name": "Research"}]}),
    )
    monkeypatch.setattr(
        cli.mcp_tabs,
        "tab_open",
        record("open", {"opened": [{"url": "https://example.com"}]}),
    )

    parser = cli.build_parser()
    commands = [
        [
            "tabs",
            "import",
            "--cdp-url",
            "http://localhost:9222",
            "--session-id",
            "sess_1",
            "--session-name",
            "Live Browser",
        ],
        ["tabs", "status", "job_1"],
        [
            "tabs",
            "search",
            "remote tabs",
            "--session-id",
            "sess_1",
            "--limit",
            "3",
            "--mode",
            "keyword",
        ],
        ["tabs", "cluster", "sess_1"],
        ["tabs", "open", "https://example.com", "--cdp-url", "http://localhost:9222"],
    ]

    for command in commands:
        args = parser.parse_args(command)
        args.func(args)

    output = capsys.readouterr().out

    assert [call["name"] for call in calls] == [
        "import",
        "status",
        "search",
        "cluster",
        "open",
    ]
    assert calls[0]["kwargs"] == {
        "cdp_url": "http://localhost:9222",
        "session_id": "sess_1",
        "session_name": "Live Browser",
    }
    assert calls[1]["args"] == ("job_1",)
    assert calls[2]["kwargs"] == {
        "query": "remote tabs",
        "session_id": "sess_1",
        "limit": 3,
        "mode": "keyword",
    }
    assert calls[3]["args"] == ("sess_1",)
    assert calls[4]["kwargs"] == {
        "urls": ["https://example.com"],
        "session_id": None,
        "cdp_url": "http://localhost:9222",
    }
    assert "agent-token-value" not in output
    assert json.loads(output.splitlines()[0]) == {"job_id": "job_1", "status": "queued"}


def test_tabs_cli_errors_redact_configured_agent_token(monkeypatch, capsys):
    monkeypatch.setenv("BACKEND_AGENT_API_TOKEN", "agent-token-value")

    def raise_token_error(**_kwargs):
        raise RuntimeError("backend rejected agent-token-value")

    monkeypatch.setattr(cli.mcp_tabs, "tab_search", raise_token_error)

    parser = cli.build_parser()
    args = parser.parse_args(["tabs", "search", "remote tabs"])

    try:
        args.func(args)
    except SystemExit as error:
        assert error.code == 1
    else:
        raise AssertionError("expected CLI command to exit on backend error")

    output = capsys.readouterr()
    assert "agent-token-value" not in output.out
    assert "agent-token-value" not in output.err
    assert "<redacted>" in output.err
