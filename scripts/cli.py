#!/usr/bin/env python3
"""Tab Organizer CLI - Unified management tool."""

import argparse
import asyncio
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
DOCKER_COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"
HOST_AI_TOKEN_FILE = PROJECT_ROOT / "data" / "host-ai-token"
DEFAULT_STACK_HEALTHCHECKS = (
    ("Backend Core", "http://localhost:8080/health"),
    ("AI Engine", "http://localhost:8090/health"),
    ("Browser Engine", "http://localhost:8083/health"),
)
WEB_UI_HEALTHCHECK = ("Web UI", "http://localhost:8089/_stcore/health")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mcp import tabs as mcp_tabs


def run_command(
    cmd: list[str],
    check: bool = True,
    capture: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run a shell command."""
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        check=check,
        capture_output=capture,
        text=True,
        env=env,
    )


def docker_compose(
    *args: str,
    profiles: list[str] = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run docker compose command."""
    cmd = ["docker", "compose", "-f", str(DOCKER_COMPOSE_FILE)]
    
    if profiles:
        for profile in profiles:
            cmd.extend(["--profile", profile])
    
    cmd.extend(args)
    return run_command(cmd, env=env)


def load_env_file() -> None:
    """Load simple KEY=VALUE pairs from .env without overriding the shell."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return

    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


def ensure_host_ai_token() -> str:
    """Return a local shared token for container-to-host AI Engine calls."""
    token = os.getenv("AI_ENGINE_API_TOKEN", "").strip()
    if token:
        return token

    if HOST_AI_TOKEN_FILE.exists():
        token = HOST_AI_TOKEN_FILE.read_text().strip()
        if token:
            return token

    HOST_AI_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    HOST_AI_TOKEN_FILE.write_text(f"{token}\n")
    HOST_AI_TOKEN_FILE.chmod(0o600)
    return token


def set_env_default_if_blank(env: dict[str, str], key: str, value: str) -> None:
    """Set an env default when a copied .env left the key blank."""
    if not env.get(key, "").strip():
        env[key] = value


def service_env_with_tokens() -> dict[str, str]:
    """Return compose env with shared service auth tokens populated."""
    env = os.environ.copy()
    token = ensure_host_ai_token()
    set_env_default_if_blank(env, "AI_ENGINE_API_TOKEN", token)
    set_env_default_if_blank(env, "BACKEND_CALLBACK_TOKEN", token)
    set_env_default_if_blank(env, "BACKEND_AGENT_API_TOKEN", token)
    return env


def wait_for_http_health(name: str, url: str, timeout_seconds: int = 60) -> None:
    """Wait until a local service health endpoint accepts requests."""
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 500:
                    return
                last_error = f"HTTP {response.status}"
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = str(error)
        time.sleep(1)
    raise RuntimeError(f"{name} did not become ready at {url}: {last_error}")


def wait_for_default_stack(include_web_ui: bool = False) -> None:
    """Wait for services that integration/e2e tests call immediately."""
    healthchecks = list(DEFAULT_STACK_HEALTHCHECKS)
    if include_web_ui:
        healthchecks.append(WEB_UI_HEALTHCHECK)
    for name, url in healthchecks:
        wait_for_http_health(name, url)


def cmd_start(args):
    """Start all services."""
    load_env_file()

    profiles = ["default"]
    if args.dev:
        profiles = ["dev"]
    
    extra_args = []
    if args.build:
        extra_args.append("--build")
    if args.detach:
        extra_args.append("-d")

    compose_env = service_env_with_tokens()
    if args.host_ai:
        compose_env["AI_ENGINE_URL"] = args.host_ai_url
        extra_args.extend(["--scale", "ai-engine=0"])
    
    docker_compose("up", *extra_args, profiles=profiles, env=compose_env)
    
    if args.detach:
        print("\nServices started.")
        print("   Web UI:         http://localhost:8089")
        print("   Backend API:    http://localhost:8080")
        if args.host_ai:
            print(f"   AI Engine:      host-run at {args.host_ai_url}")
        else:
            print("   AI Engine:      http://localhost:8090")
        print("   Browser Engine: http://localhost:8083")
        print("   Ollama:         http://localhost:11434")
        print("   LanceDB:        embedded in AI Engine (volume: lancedb-data)")


def cmd_host_ai(args):
    """Run the AI engine on the host so it can use authenticated local CLIs."""
    load_env_file()

    env = os.environ.copy()
    provider = args.provider or os.getenv("AI_PROVIDER") or "claude_code"
    embedding_provider = (
        args.embedding_provider or os.getenv("EMBEDDING_PROVIDER") or "ollama"
    )
    env["AI_PROVIDER"] = provider
    env["EMBEDDING_PROVIDER"] = embedding_provider
    env["AI_ENGINE_API_TOKEN"] = ensure_host_ai_token()
    set_env_default_if_blank(env, "BACKEND_CALLBACK_TOKEN", env["AI_ENGINE_API_TOKEN"])
    set_env_default_if_blank(
        env, "VECTOR_DB_PATH", str(PROJECT_ROOT / "data" / "lancedb-host")
    )
    set_env_default_if_blank(
        env,
        "AGENT_CLI_WORKDIR",
        str(PROJECT_ROOT / "data" / "agent-cli-workdir"),
    )

    if embedding_provider == "ollama":
        ollama_host = args.ollama_host or env.get("OLLAMA_HOST", "")
        if not ollama_host or "://ollama:" in ollama_host or "host.docker.internal" in ollama_host:
            ollama_host = "http://localhost:11434"
        env["OLLAMA_HOST"] = ollama_host
    if args.llm_model:
        env["LLM_MODEL"] = args.llm_model
    if args.embedding_model:
        env["EMBEDDING_MODEL"] = args.embedding_model
    if args.claude_code_command:
        env["CLAUDE_CODE_COMMAND"] = args.claude_code_command
    if args.codex_cli_command:
        env["CODEX_CLI_COMMAND"] = args.codex_cli_command
    if args.codex_acp_command:
        env["CODEX_ACP_COMMAND"] = args.codex_acp_command

    print(
        "Host AI engine mode uses your local CLI auth state. "
        "Start Docker with './scripts/cli.py start -d --host-ai' in another terminal."
    )
    run_command(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "services.ai_engine.app.main:app",
            "--host",
            args.host,
            "--port",
            str(args.port),
        ],
        env=env,
    )


def cmd_check_provider(args):
    """Run a local provider availability check and optional generation smoke."""
    load_env_file()

    from config.config_loader import get_ai_config
    from services.ai_engine.app.core.llm_client import LLMClient, LLMConfig

    ai_config = get_ai_config()
    provider = args.provider or os.getenv("AI_PROVIDER") or "openrouter"
    model = (
        args.model
        or os.getenv("LLM_MODEL")
        or ai_config.get_default_model(provider, "llm")
        or ""
    )

    probe_client = LLMClient(
        LLMConfig(
            provider="openrouter",
            model=ai_config.get_default_model("openrouter", "llm") or "",
        )
    )
    available = probe_client.is_provider_runtime_available(provider, "llm")
    print(f"{provider}: {'available' if available else 'not available'}")

    if args.generate:
        client = LLMClient(LLMConfig(provider=provider, model=model))
        result = asyncio.run(client.generate(args.prompt))
        print(result)


def print_backend_result(result: dict) -> None:
    """Print Backend Core JSON without exposing configured tokens."""
    text = json.dumps(result, sort_keys=True)
    print(mcp_tabs.redact_configured_secrets(text))


def run_backend_tab_tool(tool, *args, **kwargs) -> None:
    """Run a Backend Core tab tool and print a redacted JSON result."""
    load_env_file()
    set_env_default_if_blank(
        os.environ,
        mcp_tabs.AGENT_TOKEN_ENV,
        ensure_host_ai_token(),
    )
    try:
        print_backend_result(tool(*args, **kwargs))
    except Exception as error:
        message = mcp_tabs.redact_configured_secrets(str(error))
        print(f"Error: {message}", file=sys.stderr)
        raise SystemExit(1) from error


def cmd_tabs_import(args):
    """Import currently open browser tabs through Backend Core."""
    run_backend_tab_tool(
        mcp_tabs.tab_import_from_browser,
        cdp_url=args.cdp_url,
        session_id=args.session_id,
        session_name=args.session_name,
    )


def cmd_tabs_status(args):
    """Show a browser tab import job status."""
    run_backend_tab_tool(mcp_tabs.tab_import_status, args.job_id)


def cmd_tabs_search(args):
    """Search indexed browser tabs through Backend Core."""
    run_backend_tab_tool(
        mcp_tabs.tab_search,
        query=args.query,
        session_id=args.session_id,
        limit=args.limit,
        mode=args.mode,
    )


def cmd_tabs_cluster(args):
    """Cluster indexed browser tabs through Backend Core."""
    run_backend_tab_tool(mcp_tabs.tab_cluster, args.session_id)


def cmd_tabs_open(args):
    """Open URLs in the attached local browser through Backend Core."""
    run_backend_tab_tool(
        mcp_tabs.tab_open,
        urls=args.urls,
        session_id=args.session_id,
        cdp_url=args.cdp_url,
    )


def cmd_tabs_export(args):
    """Export organized tabs for a Backend Core session."""
    run_backend_tab_tool(
        mcp_tabs.tab_export,
        session_id=args.session_id,
        export_format=args.format,
    )


def cmd_stop(args):
    """Stop all services."""
    extra_args = []
    if args.volumes:
        extra_args.append("-v")
    
    docker_compose("down", *extra_args, profiles=["default", "dev"])
    print("Services stopped")


def cmd_restart(args):
    """Restart services."""
    services = args.services if args.services else []
    docker_compose("restart", *services, profiles=["default"])
    print("Services restarted")


def cmd_status(args):
    """Show service status."""
    docker_compose("ps", profiles=["default", "dev"])


def cmd_logs(args):
    """Show service logs."""
    extra_args = []
    if args.follow:
        extra_args.append("-f")
    if args.tail:
        extra_args.extend(["--tail", str(args.tail)])
    
    services = [args.service] if args.service else []
    docker_compose("logs", *extra_args, *services, profiles=["default"])


def cmd_test(args):
    """Run tests."""
    test_type = args.type or "unit"
    test_env = service_env_with_tokens()
    test_profiles = [f"test-{test_type}"]

    print(f"Running {test_type} tests...")

    if test_type == "all":
        for suite in ("unit", "integration", "e2e"):
            cmd_test(argparse.Namespace(type=suite))
        return

    if test_type in ("integration", "e2e"):
        # Start dependencies first. Provide the maintainer bootstrap code only
        # for local test runs; the published stack has no default maintainer code.
        set_env_default_if_blank(
            test_env, "PLATFORM_MAINTAINER_SIGNUP_CODE", "local-maintainer"
        )
        docker_compose("up", "-d", profiles=["default"], env=test_env)
        wait_for_default_stack(include_web_ui=test_type == "e2e")
        test_profiles = ["default", f"test-{test_type}"]

    docker_compose(
        "run",
        "--rm",
        f"test-{test_type}",
        profiles=test_profiles,
        env=test_env,
    )


def cmd_models(args):
    """Manage Ollama models."""
    if args.list:
        print("Available models in Ollama:")
        run_command(["docker", "exec", "tab-organizer-ollama", "ollama", "list"])
    elif args.pull:
        print(f"Pulling model: {args.pull}")
        run_command(["docker", "exec", "tab-organizer-ollama", "ollama", "pull", args.pull])
    else:
        print("Use --list to show models or --pull <model> to download a model")


def cmd_init(args):
    """Initialize the project."""
    print("Initializing Tab Organizer...")
    
    # Create .env from example if not exists
    env_file = PROJECT_ROOT / ".env"
    env_example = PROJECT_ROOT / ".env.example"
    
    if not env_file.exists() and env_example.exists():
        import shutil
        shutil.copy(env_example, env_file)
        print("Created .env from .env.example")
    
    # Build images
    if args.build:
        print("Building Docker images...")
        docker_compose("build", profiles=["default"])
    
    # Pull Ollama models
    if args.models:
        print("Starting Ollama...")
        docker_compose("up", "-d", "ollama", profiles=["default"])
        
        import time
        time.sleep(5)  # Wait for Ollama to start
        
        print("Pulling default models...")
        # Get default models from config
        sys.path.append(str(Path(__file__).parent.parent))
        from config.config_loader import get_ai_config
        ai_config = get_ai_config()
        
        default_llm = ai_config.get_default_model("ollama", "llm")
        default_embedding = ai_config.get_default_model("ollama", "embedding")
        
        run_command(["docker", "exec", "tab-organizer-ollama", "ollama", "pull", default_llm], check=False)
        run_command(["docker", "exec", "tab-organizer-ollama", "ollama", "pull", default_embedding], check=False)
    
    print("\nInitialization complete.")
    print("   Run './scripts/cli.py start -d' to start services")


def cmd_clean(args):
    """Clean up Docker resources."""
    print("Cleaning up...")
    
    # Stop services
    docker_compose("down", "-v", "--remove-orphans", profiles=["default", "dev"])
    
    if args.images:
        print("Removing images...")
        run_command([
            "docker", "images", "-q", "tab-organizer-*"
        ], check=False)
        # Remove project images
        result = run_command(
            ["docker", "images", "--filter", "reference=tab-organizer-*", "-q"],
            capture=True,
            check=False,
        )
        if result.stdout.strip():
            image_ids = result.stdout.strip().split("\n")
            run_command(["docker", "rmi", "-f"] + image_ids, check=False)
    
    print("Cleanup complete")


def cmd_shell(args):
    """Open a shell in a service container."""
    service = args.service
    run_command([
        "docker", "exec", "-it", f"tab-organizer-{service}", "/bin/bash"
    ], check=False)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Tab Organizer CLI - Unified management tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s init --build --models    Initialize project with images and models
  %(prog)s start -d                 Start all services in background
  %(prog)s host-ai --provider claude_code
  %(prog)s start -d --host-ai       Start Docker services wired to host AI
  %(prog)s start --build            Rebuild and start services
  %(prog)s stop                     Stop all services
  %(prog)s logs -f web-ui           Follow web-ui logs
  %(prog)s test --type unit         Run unit tests
  %(prog)s tabs search "query"      Search indexed browser tabs
  %(prog)s models --pull <model>   Pull a model
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # start
    start_parser = subparsers.add_parser("start", help="Start services")
    start_parser.add_argument("--build", "-b", action="store_true", help="Build images before starting")
    start_parser.add_argument("--detach", "-d", action="store_true", help="Run in background")
    start_parser.add_argument("--dev", action="store_true", help="Use development profile")
    start_parser.add_argument(
        "--host-ai",
        action="store_true",
        help="Route backend/browser/web containers to a host-run AI engine",
    )
    start_parser.add_argument(
        "--host-ai-url",
        default="http://host.docker.internal:8090",
        help="AI engine URL visible from containers when --host-ai is used",
    )
    start_parser.set_defaults(func=cmd_start)

    # host-ai
    host_ai_parser = subparsers.add_parser(
        "host-ai",
        help="Run AI engine on the host for subscription CLI providers",
    )
    host_ai_parser.add_argument(
        "--provider",
        choices=["claude_code", "codex_cli", "codex_acp", "openrouter", "ollama"],
        help="LLM provider to run in the host AI engine; defaults to AI_PROVIDER or claude_code",
    )
    host_ai_parser.add_argument("--llm-model", help="Override LLM_MODEL")
    host_ai_parser.add_argument(
        "--embedding-provider",
        help="Embedding provider to pair with the LLM provider; defaults to EMBEDDING_PROVIDER or ollama",
    )
    host_ai_parser.add_argument("--embedding-model", help="Override EMBEDDING_MODEL")
    host_ai_parser.add_argument(
        "--ollama-host",
        help="Host URL for Ollama embeddings in host-ai mode; defaults to http://localhost:11434 when .env points at Docker-only ollama",
    )
    host_ai_parser.add_argument(
        "--claude-code-command",
        help="Override CLAUDE_CODE_COMMAND, for example an absolute claude path",
    )
    host_ai_parser.add_argument(
        "--codex-cli-command",
        help="Override CODEX_CLI_COMMAND, for example an absolute codex path",
    )
    host_ai_parser.add_argument(
        "--codex-acp-command",
        help="Override CODEX_ACP_COMMAND, for example an absolute acpx path",
    )
    host_ai_parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    host_ai_parser.add_argument("--port", type=int, default=8090, help="Bind port")
    host_ai_parser.set_defaults(func=cmd_host_ai)

    # check-provider
    check_parser = subparsers.add_parser(
        "check-provider",
        help="Check local LLM provider availability and optional smoke generation",
    )
    check_parser.add_argument(
        "--provider",
        choices=[
            "openrouter",
            "ollama",
            "openai",
            "anthropic",
            "claude_code",
            "codex_cli",
            "codex_acp",
            "deepseek",
            "gemini",
        ],
        help="Provider to check; defaults to AI_PROVIDER",
    )
    check_parser.add_argument("--model", help="Override model for the check")
    check_parser.add_argument(
        "--generate",
        action="store_true",
        help="Run a real generation request after availability succeeds",
    )
    check_parser.add_argument(
        "--prompt",
        default="Reply with OK.",
        help="Prompt used by --generate",
    )
    check_parser.set_defaults(func=cmd_check_provider)

    # tabs
    tabs_parser = subparsers.add_parser(
        "tabs",
        help="Import, search, cluster, and open browser tabs via Backend Core",
    )
    tab_subparsers = tabs_parser.add_subparsers(
        dest="tab_command",
        help="Tab commands",
        required=True,
    )

    tabs_import_parser = tab_subparsers.add_parser(
        "import",
        help="Import currently open browser tabs from a local CDP endpoint",
    )
    tabs_import_parser.add_argument(
        "--cdp-url",
        default=mcp_tabs.DEFAULT_CDP_URL,
        help="Local Chrome DevTools Protocol URL",
    )
    tabs_import_parser.add_argument(
        "--session-id",
        help="Existing Backend Core session ID to import into",
    )
    tabs_import_parser.add_argument(
        "--session-name",
        help="Name for a new Backend Core session when no session ID is provided",
    )
    tabs_import_parser.set_defaults(func=cmd_tabs_import)

    tabs_status_parser = tab_subparsers.add_parser(
        "status",
        help="Show a tab import job status",
    )
    tabs_status_parser.add_argument("job_id", help="Backend Core tab import job ID")
    tabs_status_parser.set_defaults(func=cmd_tabs_status)

    tabs_search_parser = tab_subparsers.add_parser(
        "search",
        help="Search indexed browser tabs",
    )
    tabs_search_parser.add_argument("query", help="Search query")
    tabs_search_parser.add_argument("--session-id", help="Restrict search to a session")
    tabs_search_parser.add_argument(
        "--mode",
        choices=["hybrid", "semantic", "keyword"],
        default="hybrid",
        help="Search mode",
    )
    tabs_search_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum result count",
    )
    tabs_search_parser.set_defaults(func=cmd_tabs_search)

    tabs_cluster_parser = tab_subparsers.add_parser(
        "cluster",
        help="Cluster indexed tabs for a session",
    )
    tabs_cluster_parser.add_argument("session_id", help="Backend Core session ID")
    tabs_cluster_parser.set_defaults(func=cmd_tabs_cluster)

    tabs_open_parser = tab_subparsers.add_parser(
        "open",
        help="Open URLs in the attached local browser",
    )
    tabs_open_parser.add_argument("urls", nargs="*", help="URLs to open")
    tabs_open_parser.add_argument(
        "--session-id",
        help="Open all URLs from an existing Backend Core session",
    )
    tabs_open_parser.add_argument(
        "--cdp-url",
        default=mcp_tabs.DEFAULT_CDP_URL,
        help="Local Chrome DevTools Protocol URL",
    )
    tabs_open_parser.set_defaults(func=cmd_tabs_open)

    tabs_export_parser = tab_subparsers.add_parser(
        "export",
        help="Export organized tabs for a session",
    )
    tabs_export_parser.add_argument("session_id", help="Backend Core session ID")
    tabs_export_parser.add_argument(
        "--format",
        choices=["markdown", "json", "html", "obsidian"],
        default="markdown",
        help="Export format",
    )
    tabs_export_parser.set_defaults(func=cmd_tabs_export)
    
    # stop
    stop_parser = subparsers.add_parser("stop", help="Stop services")
    stop_parser.add_argument("--volumes", "-v", action="store_true", help="Remove volumes")
    stop_parser.set_defaults(func=cmd_stop)
    
    # restart
    restart_parser = subparsers.add_parser("restart", help="Restart services")
    restart_parser.add_argument("services", nargs="*", help="Services to restart")
    restart_parser.set_defaults(func=cmd_restart)
    
    # status
    status_parser = subparsers.add_parser("status", help="Show service status")
    status_parser.set_defaults(func=cmd_status)
    
    # logs
    logs_parser = subparsers.add_parser("logs", help="Show service logs")
    logs_parser.add_argument("service", nargs="?", help="Service name")
    logs_parser.add_argument("--follow", "-f", action="store_true", help="Follow logs")
    logs_parser.add_argument("--tail", "-n", type=int, help="Number of lines")
    logs_parser.set_defaults(func=cmd_logs)
    
    # test
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("--type", "-t", choices=["unit", "integration", "e2e", "all"], help="Test type")
    test_parser.set_defaults(func=cmd_test)
    
    # models
    models_parser = subparsers.add_parser("models", help="Manage Ollama models")
    models_parser.add_argument("--list", "-l", action="store_true", help="List models")
    models_parser.add_argument("--pull", "-p", metavar="MODEL", help="Pull a model")
    models_parser.set_defaults(func=cmd_models)
    
    # init
    init_parser = subparsers.add_parser("init", help="Initialize project")
    init_parser.add_argument("--build", "-b", action="store_true", help="Build images")
    init_parser.add_argument("--models", "-m", action="store_true", help="Pull default models")
    init_parser.set_defaults(func=cmd_init)
    
    # clean
    clean_parser = subparsers.add_parser("clean", help="Clean up resources")
    clean_parser.add_argument("--images", "-i", action="store_true", help="Remove images")
    clean_parser.set_defaults(func=cmd_clean)
    
    # shell
    shell_parser = subparsers.add_parser("shell", help="Open shell in container")
    shell_parser.add_argument("service", choices=["backend", "ai", "browser", "ui", "ollama"])
    shell_parser.set_defaults(func=cmd_shell)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()
