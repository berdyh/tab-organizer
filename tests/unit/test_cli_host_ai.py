"""CLI regressions for host-run AI Engine routing."""

import argparse
import json
import re
import subprocess
from pathlib import Path

import pytest

from scripts import cli
from scripts.mcp import tabs as mcp_tabs

ROOT = Path(__file__).resolve().parents[2]


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


def _stub_service_tokens(monkeypatch):
    """Mint a deterministic, per-scope-distinct token without touching disk."""
    for env_name in cli.SERVICE_TOKEN_ENVS:
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setattr(cli, "ensure_service_token", lambda name: f"token-{name}")


def _isolate_token_store(monkeypatch, tmp_path):
    """Point both the new and legacy token stores at a temp directory."""
    monkeypatch.setattr(cli, "HOST_AI_TOKEN_FILE", tmp_path / "host-ai-token")
    monkeypatch.setattr(cli, "SERVICE_TOKEN_FILE", tmp_path / "service-tokens.json")
    for env_name in cli.SERVICE_TOKEN_ENVS:
        monkeypatch.delenv(env_name, raising=False)
    return tmp_path


def test_ensure_host_ai_token_reuses_configured_env(monkeypatch, tmp_path):
    _isolate_token_store(monkeypatch, tmp_path)
    monkeypatch.setenv("AI_ENGINE_API_TOKEN", "configured-token")

    assert cli.ensure_host_ai_token() == "configured-token"
    assert not (tmp_path / "host-ai-token").exists()
    assert not (tmp_path / "service-tokens.json").exists()


def test_service_env_mints_four_distinct_tokens_in_stock_env(monkeypatch, tmp_path):
    """Stock-env regression guard for the collapsed-token privilege bug.

    A single shared value made BACKEND_AGENT_API_TOKEN equal to the token
    browser-engine accepts, so an agent principal reached the scrape, CDP and
    credential control plane. Every scope must get its own value.
    """
    _isolate_token_store(monkeypatch, tmp_path)

    env = cli.service_env_with_tokens()

    tokens = {name: env[name] for name in cli.SERVICE_TOKEN_ENVS}
    assert set(tokens) == {
        "AI_ENGINE_API_TOKEN",
        "BACKEND_CALLBACK_TOKEN",
        "BACKEND_AGENT_API_TOKEN",
        "BROWSER_ENGINE_API_TOKEN",
    }
    assert all(value.strip() for value in tokens.values())
    assert len(set(tokens.values())) == 4, f"tokens are not pairwise distinct: {tokens}"

    # Stable across restarts: a second call reuses the persisted store.
    assert {
        name: cli.service_env_with_tokens()[name] for name in cli.SERVICE_TOKEN_ENVS
    } == tokens


def test_service_env_seeds_ai_token_from_legacy_single_token_file(
    monkeypatch, tmp_path
):
    """Existing installs keep their ai-engine token; the rest are newly minted."""
    _isolate_token_store(monkeypatch, tmp_path)
    (tmp_path / "host-ai-token").write_text("legacy-single-token\n")

    env = cli.service_env_with_tokens()

    assert env["AI_ENGINE_API_TOKEN"] == "legacy-single-token"
    others = {
        name: env[name]
        for name in cli.SERVICE_TOKEN_ENVS
        if name != "AI_ENGINE_API_TOKEN"
    }
    assert "legacy-single-token" not in others.values()
    assert len(set(others.values())) == 3
    assert (tmp_path / "service-tokens.json").exists()


def test_service_env_never_clobbers_user_supplied_tokens(monkeypatch, tmp_path):
    _isolate_token_store(monkeypatch, tmp_path)
    monkeypatch.setenv("BACKEND_AGENT_API_TOKEN", "user-agent-token")

    env = cli.service_env_with_tokens()

    assert env["BACKEND_AGENT_API_TOKEN"] == "user-agent-token"
    assert env["BROWSER_ENGINE_API_TOKEN"] != "user-agent-token"


def test_host_ai_rewrites_docker_ollama_host_and_sets_token(monkeypatch):
    calls = []

    monkeypatch.setattr(cli, "load_env_file", lambda: None)
    _stub_service_tokens(monkeypatch)
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
    assert env["AI_ENGINE_API_TOKEN"] == "token-AI_ENGINE_API_TOKEN"
    assert env["BACKEND_CALLBACK_TOKEN"] == "token-BACKEND_CALLBACK_TOKEN"
    assert env["AI_ENGINE_API_TOKEN"] != env["BACKEND_CALLBACK_TOKEN"]


def test_cmd_host_ai_fails_closed_when_ai_provider_not_set(monkeypatch):
    """host-ai is the documented way to use subscription providers (CLAUDE.md).

    Before this fix it silently injected AI_PROVIDER=claude_code /
    EMBEDDING_PROVIDER=ollama when neither a flag nor an env var supplied one
    -- forging the exact "env var is the record of consent" invariant
    SPEC-provider-routing.md R1/R3 requires. It must refuse instead of ever
    invoking uvicorn with a provider nobody chose.
    """
    calls = []
    monkeypatch.setattr(cli, "load_env_file", lambda: None)
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    _stub_service_tokens(monkeypatch)
    monkeypatch.setattr(
        cli, "run_command", lambda cmd, env=None, **kwargs: calls.append(cmd)
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_host_ai(_host_ai_args(provider=None, embedding_provider=None))

    message = str(exc_info.value)
    assert "provider_not_selected" in message
    assert "AI_PROVIDER" in message
    assert "configure-provider" in message
    assert not calls, "uvicorn must never start with an unselected provider"


def test_cmd_host_ai_fails_closed_when_embedding_provider_not_set(monkeypatch):
    """Same fail-closed rule for EMBEDDING_PROVIDER, isolated from AI_PROVIDER
    by supplying --provider explicitly (real consent) so only the embedding
    gate is under test."""
    calls = []
    monkeypatch.setattr(cli, "load_env_file", lambda: None)
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    _stub_service_tokens(monkeypatch)
    monkeypatch.setattr(
        cli, "run_command", lambda cmd, env=None, **kwargs: calls.append(cmd)
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_host_ai(_host_ai_args(provider="claude_code", embedding_provider=None))

    message = str(exc_info.value)
    assert "provider_not_selected" in message
    assert "EMBEDDING_PROVIDER" in message
    assert not calls


def test_cmd_host_ai_accepts_explicit_provider_flags(monkeypatch):
    """The legitimate route keeps working: an explicit --provider/
    --embedding-provider IS the deliberate choice (R3), with no AI_PROVIDER/
    EMBEDDING_PROVIDER in the environment at all."""
    calls = []
    monkeypatch.setattr(cli, "load_env_file", lambda: None)
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    _stub_service_tokens(monkeypatch)
    monkeypatch.setattr(
        cli,
        "run_command",
        lambda cmd, env=None, **kwargs: calls.append({"cmd": cmd, "env": env}),
    )

    cli.cmd_host_ai(_host_ai_args(provider="codex_acp", embedding_provider="ollama"))

    assert calls
    env = calls[0]["env"]
    assert env["AI_PROVIDER"] == "codex_acp"
    assert env["EMBEDDING_PROVIDER"] == "ollama"


def test_host_ai_parser_default_host_is_not_bound_to_all_interfaces():
    """Source-level guard in the spirit of
    test_runtime_auth_config.py::test_published_ports_are_bound_to_loopback:
    that test only covers docker-compose.yml's published ports, so this is
    the host-ai equivalent for the one bind that runs on the operator's own
    machine, outside Docker."""
    source = (ROOT / "scripts" / "cli.py").read_text()
    match = re.search(
        r'host_ai_parser\.add_argument\(\s*"--host",\s*default=([^,\n]+),',
        source,
    )
    assert match, "could not find host-ai's --host argument definition"
    assert match.group(1).strip() != '"0.0.0.0"', (
        "host-ai's --host default binds every interface, exposing the "
        "host-run AI Engine to the whole LAN"
    )


def test_host_ai_cli_parses_host_default_as_none():
    parser = cli.build_parser()
    args = parser.parse_args(["host-ai"])
    assert args.host is None


def test_resolve_host_ai_bind_host_returns_explicit_value_verbatim():
    assert cli.resolve_host_ai_bind_host("127.0.0.1") == "127.0.0.1"
    assert cli.resolve_host_ai_bind_host("203.0.113.5") == "203.0.113.5"


def test_resolve_host_ai_bind_host_honors_explicit_wide_open_opt_in(capsys):
    # 0.0.0.0 is still allowed when the operator asks for it explicitly --
    # this is an opt-in widening, not the default -- but it must warn loudly.
    assert cli.resolve_host_ai_bind_host("0.0.0.0") == "0.0.0.0"
    assert "0.0.0.0" in capsys.readouterr().err


def test_resolve_host_ai_bind_host_discovers_bridge_gateway_when_unset(monkeypatch):
    monkeypatch.setattr(cli, "discover_docker_bridge_gateway", lambda: "172.21.0.1")
    assert cli.resolve_host_ai_bind_host(None) == "172.21.0.1"


def test_resolve_host_ai_bind_host_fails_loudly_when_bridge_undiscoverable(
    monkeypatch,
):
    monkeypatch.setattr(cli, "discover_docker_bridge_gateway", lambda: None)
    try:
        cli.resolve_host_ai_bind_host(None)
    except SystemExit as error:
        assert "bridge gateway" in str(error)
        assert "0.0.0.0" not in str(error) or "--host" in str(error)
    else:
        raise AssertionError(
            "expected resolve_host_ai_bind_host to fail closed, not guess"
        )


def test_discover_docker_bridge_gateway_parses_matching_network(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:3] == ["docker", "network", "ls"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="bridge\nresiduals_tab-organizer-network\nhost\n",
                stderr="",
            )
        if cmd[:3] == ["docker", "network", "inspect"]:
            assert cmd[3] == "residuals_tab-organizer-network"
            return subprocess.CompletedProcess(cmd, 0, stdout="172.21.0.1\n", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli.discover_docker_bridge_gateway() == "172.21.0.1"
    assert len(calls) == 2


def test_discover_docker_bridge_gateway_returns_none_when_docker_unavailable(
    monkeypatch,
):
    def raise_missing(cmd, **kwargs):
        raise FileNotFoundError("docker not found")

    monkeypatch.setattr(cli.subprocess, "run", raise_missing)

    assert cli.discover_docker_bridge_gateway() is None


def test_discover_docker_bridge_gateway_returns_none_on_ambiguous_match(monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="foo_tab-organizer-network\nbar_tab-organizer-network\n",
            stderr="",
        )

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli.discover_docker_bridge_gateway() is None


def test_discover_docker_bridge_gateway_returns_none_on_garbage_gateway(monkeypatch):
    def fake_run(cmd, **kwargs):
        if cmd[:3] == ["docker", "network", "ls"]:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="tab-organizer-network\n", stderr=""
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="not-an-ip\n", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli.discover_docker_bridge_gateway() is None


def test_cmd_host_ai_binds_discovered_gateway_not_wide_open(monkeypatch):
    calls = []

    monkeypatch.setattr(cli, "load_env_file", lambda: None)
    _stub_service_tokens(monkeypatch)
    monkeypatch.setattr(cli, "discover_docker_bridge_gateway", lambda: "172.22.0.1")
    monkeypatch.setattr(
        cli,
        "run_command",
        lambda cmd, env=None, **kwargs: calls.append({"cmd": cmd, "env": env}),
    )

    cli.cmd_host_ai(_host_ai_args(host=None))

    assert calls
    cmd = calls[0]["cmd"]
    host_index = cmd.index("--host")
    assert cmd[host_index + 1] == "172.22.0.1"
    assert "0.0.0.0" not in cmd


def test_cmd_host_ai_fails_closed_when_bridge_undiscoverable(monkeypatch):
    monkeypatch.setattr(cli, "load_env_file", lambda: None)
    _stub_service_tokens(monkeypatch)
    monkeypatch.setattr(cli, "discover_docker_bridge_gateway", lambda: None)
    monkeypatch.setattr(
        cli,
        "run_command",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("run_command must not be reached")
        ),
    )

    try:
        cli.cmd_host_ai(_host_ai_args(host=None))
    except SystemExit:
        pass
    else:
        raise AssertionError("expected cmd_host_ai to fail closed, not bind 0.0.0.0")


def test_start_host_ai_sets_container_url_token_and_disables_ai_container(monkeypatch):
    calls = []

    monkeypatch.setattr(cli, "load_env_file", lambda: None)
    _stub_service_tokens(monkeypatch)
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
    env = calls[0]["env"]
    minted = [env[name] for name in cli.SERVICE_TOKEN_ENVS]
    assert minted == [f"token-{name}" for name in cli.SERVICE_TOKEN_ENVS]
    assert len(set(minted)) == 4


def test_start_populates_service_tokens_for_default_stack(monkeypatch):
    calls = []

    monkeypatch.setattr(cli, "load_env_file", lambda: None)
    _stub_service_tokens(monkeypatch)
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
    env = calls[0]["env"]
    minted = [env[name] for name in cli.SERVICE_TOKEN_ENVS]
    assert minted == [f"token-{name}" for name in cli.SERVICE_TOKEN_ENVS]
    assert len(set(minted)) == 4
    assert "AI_ENGINE_URL" not in env


def test_service_env_replaces_blank_env_tokens_from_dotenv(monkeypatch):
    _stub_service_tokens(monkeypatch)
    for env_name in cli.SERVICE_TOKEN_ENVS:
        monkeypatch.setenv(env_name, "")

    env = cli.service_env_with_tokens()

    for env_name in cli.SERVICE_TOKEN_ENVS:
        assert env[env_name] == f"token-{env_name}"
    assert len({env[name] for name in cli.SERVICE_TOKEN_ENVS}) == 4


def test_cmd_test_loads_env_file_like_other_stack_commands(monkeypatch):
    """`test` must load .env like `start`/`host-ai`/`check-provider` do.

    cmd_test recreates the stack (docker compose up, for integration/e2e)
    through service_env_with_tokens(), which resolves each token from
    os.environ first and only falls back to the persisted
    data/service-tokens.json store when nothing is configured there.
    Skipping load_env_file() meant an explicit .env token/config override
    was invisible to `test`, so `test --type integration` could recreate the
    stack on different values than `start -d` used for the exact same .env
    -- the two commands would then hold the running stack open on divergent
    tokens. Every other command that calls service_env_with_tokens() calls
    load_env_file() first; `test` was the one place that didn't.
    """
    loaded = []
    monkeypatch.setattr(cli, "load_env_file", lambda: loaded.append(True))
    monkeypatch.setattr(cli, "service_env_with_tokens", lambda: {})
    monkeypatch.setattr(
        cli,
        "docker_compose",
        lambda *args, profiles=None, env=None: None,
    )
    monkeypatch.setattr(
        cli, "wait_for_default_stack", lambda include_web_ui=False: None
    )

    cli.cmd_test(argparse.Namespace(type="unit"))
    assert loaded == [True], "cmd_test(unit) must call load_env_file()"

    loaded.clear()
    cli.cmd_test(argparse.Namespace(type="integration"))
    assert loaded == [True], "cmd_test(integration) must call load_env_file()"


def test_integration_test_waits_for_default_stack(monkeypatch):
    calls = []
    waits = []

    monkeypatch.setattr(cli, "service_env_with_tokens", lambda: {})
    monkeypatch.setattr(
        cli,
        "docker_compose",
        lambda *args, profiles=None, env=None: calls.append(
            {"args": args, "profiles": profiles, "env": env}
        ),
    )
    monkeypatch.setattr(
        cli,
        "wait_for_default_stack",
        lambda include_web_ui=False: waits.append(include_web_ui),
    )

    cli.cmd_test(argparse.Namespace(type="integration"))

    assert calls[0]["args"] == ("up", "-d")
    assert calls[0]["profiles"] == ["default"]
    assert calls[0]["env"]["PLATFORM_MAINTAINER_SIGNUP_CODE"] == "local-maintainer"
    assert calls[1]["args"] == ("run", "--rm", "test-integration")
    assert calls[1]["profiles"] == ["default", "test-integration"]
    assert waits == [False]


def test_e2e_test_waits_for_web_ui(monkeypatch):
    calls = []
    waits = []

    monkeypatch.setattr(cli, "service_env_with_tokens", lambda: {})
    monkeypatch.setattr(
        cli,
        "docker_compose",
        lambda *args, profiles=None, env=None: calls.append(
            {"args": args, "profiles": profiles, "env": env}
        ),
    )
    monkeypatch.setattr(
        cli,
        "wait_for_default_stack",
        lambda include_web_ui=False: waits.append(include_web_ui),
    )

    cli.cmd_test(argparse.Namespace(type="e2e"))

    assert calls[0]["args"] == ("up", "-d")
    assert calls[1]["args"] == ("run", "--rm", "test-e2e")
    assert calls[1]["profiles"] == ["default", "test-e2e"]
    assert waits == [True]


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
    monkeypatch.setattr(
        cli.mcp_tabs,
        "tab_export",
        record("export", {"content": "# Export"}),
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
        ["tabs", "export", "sess_1", "--format", "markdown"],
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
        "export",
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
    assert calls[5]["kwargs"] == {
        "session_id": "sess_1",
        "export_format": "markdown",
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
