#!/usr/bin/env python3
"""Unified command line interface for Tab Organizer operations.

This tool replaces the legacy shell scripts with a single entry point that
handles lifecycle management, diagnostics, and test execution.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Iterable, List
from urllib import request, error as url_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_TRACK_FILE = PROJECT_ROOT / ".docker-compose-files"
DEFAULT_COMPOSE_FILES = ["docker-compose.yml"]
TEST_COMPOSE_FILE = "docker-compose.test.yml"


def run_command(
    command: List[str],
    *,
    cwd: Path | None = None,
    capture_output: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Execute a shell command with consistent defaults."""
    cwd = cwd or PROJECT_ROOT
    print(f"→ {' '.join(command)}")
    return subprocess.run(
        command,
        cwd=str(cwd),
        check=check,
        capture_output=capture_output,
        text=True,
    )


def update_env_var(key: str, value: str, env_file: Path | None = None) -> None:
    """Insert or overwrite a key/value pair in the project's .env file."""
    env_file = env_file or PROJECT_ROOT / ".env"
    if not env_file.exists():
        return

    lines = env_file.read_text().splitlines()
    updated = False
    for idx, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[idx] = f"{key}={value}"
            updated = True
            break
    if not updated:
        lines.append(f"{key}={value}")

    env_file.write_text("\n".join(lines) + "\n")


def ensure_env_file() -> None:
    """Ensure the runtime .env file exists."""
    env_file = PROJECT_ROOT / ".env"
    example = PROJECT_ROOT / ".env.example"

    if env_file.exists():
        return
    if not example.exists():
        raise SystemExit("Missing .env and .env.example files. Run scripts/init.py first.")
    env_file.write_text(example.read_text())
    print("Created .env from .env.example; update it with your configuration.")


def ensure_logs_dir() -> None:
    """Create the logs directory if needed."""
    (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)


def detect_gpu() -> bool:
    """Heuristically detect if nvidia-smi can see a GPU."""
    try:
        run_command(["nvidia-smi"], capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def detect_local_ollama(timeout: float = 2.0) -> bool:
    """Check whether a local Ollama instance is reachable."""
    try:
        request.urlopen("http://127.0.0.1:11434/api/tags", timeout=timeout)
        return True
    except (url_error.URLError, url_error.HTTPError):
        return False


def choose_profile(preference: str) -> List[str]:
    """Resolve docker compose profile arguments."""
    if preference == "auto":
        return ["--profile", "gpu"] if detect_gpu() else ["--profile", "cpu"]
    if preference in {"gpu", "cpu"}:
        return ["--profile", preference]
    return []


def read_active_compose_files() -> List[str]:
    """Return the compose files that were active during the last start."""
    if not COMPOSE_TRACK_FILE.exists():
        return []
    return [line.strip() for line in COMPOSE_TRACK_FILE.read_text().splitlines() if line.strip()]


def save_active_compose_files(files: Iterable[str]) -> None:
    """Persist the compose files used for the running stack."""
    COMPOSE_TRACK_FILE.write_text("\n".join(files) + "\n")


def compose_arguments(files: Iterable[str]) -> List[str]:
    """Build docker compose CLI arguments for the provided files."""
    args: List[str] = []
    for compose_file in files:
        args.extend(["-f", compose_file])
    return args


def require_docker() -> None:
    """Verify that docker and docker compose are available."""
    try:
        run_command(["docker", "--version"], capture_output=True)
        run_command(["docker", "compose", "version"], capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise SystemExit("Docker Compose V2 is required. Install or update Docker before continuing.")


def cmd_start(args: argparse.Namespace) -> None:
    """Start the application stack."""
    require_docker()
    ensure_env_file()
    ensure_logs_dir()

    compose_files = list(DEFAULT_COMPOSE_FILES)
    ollama_mode = args.ollama_mode
    profile_flags: List[str] = []

    use_local = detect_local_ollama() if ollama_mode == "auto" else ollama_mode == "local"
    if use_local:
        update_env_var("OLLAMA_URL", "http://localhost:11434")
        compose_files.append("docker-compose.local-ollama.yml")
        print("Using locally running Ollama on http://localhost:11434")
    else:
        update_env_var("OLLAMA_URL", "http://ollama:11434")
        profile_flags = choose_profile(args.profile)
        print("Using dockerized Ollama service (profile: {})".format(profile_flags[1] if profile_flags else "default"))

    compose_args = compose_arguments(compose_files)

    if args.pull:
        run_command(["docker", "compose", *compose_args, *profile_flags, "pull"])

    if args.build:
        run_command(["docker", "compose", *compose_args, *profile_flags, "build"])

    up_args = ["docker", "compose", *compose_args, *profile_flags, "up", "-d"]
    if args.quiet_pull:
        up_args.append("--quiet-pull")
    run_command(up_args)

    save_active_compose_files(compose_files)
    print("Stack is up. Access the web UI at http://localhost:8089")


def cmd_stop(args: argparse.Namespace) -> None:
    """Stop the running application stack."""
    require_docker()
    compose_files = read_active_compose_files() or DEFAULT_COMPOSE_FILES
    compose_args = compose_arguments(compose_files)

    down_command = ["docker", "compose", *compose_args, "down"]
    if args.volumes:
        down_command.append("-v")
    run_command(down_command)

    if COMPOSE_TRACK_FILE.exists():
        COMPOSE_TRACK_FILE.unlink()

    print("Services stopped successfully.")


def cmd_restart(args: argparse.Namespace) -> None:
    """Restart the application stack."""
    cmd_stop(argparse.Namespace(volumes=False))
    cmd_start(args)


def cmd_status(_: argparse.Namespace) -> None:
    """Display docker compose status."""
    require_docker()
    compose_files = read_active_compose_files() or DEFAULT_COMPOSE_FILES
    compose_args = compose_arguments(compose_files)
    run_command(["docker", "compose", *compose_args, "ps"])


def cmd_logs(args: argparse.Namespace) -> None:
    """Stream logs from the stack."""
    require_docker()
    compose_files = read_active_compose_files() or DEFAULT_COMPOSE_FILES
    compose_args = compose_arguments(compose_files)
    command = ["docker", "compose", *compose_args, "logs", "-f"]
    if args.service:
        command.append(args.service)
    run_command(command, check=False)


def ensure_test_directories() -> None:
    """Create directories used during test collection."""
    for dirname in ("test-results", "coverage", "test-reports", "logs"):
        (PROJECT_ROOT / dirname).mkdir(parents=True, exist_ok=True)


UNIT_TEST_SERVICES = [
    "url-input-unit-test",
    "auth-unit-test",
    "scraper-unit-test",
    "analyzer-unit-test",
    "clustering-unit-test",
    "export-unit-test",
    "session-unit-test",
    "monitoring-unit-test",
    "api-gateway-unit-test",
    "web-ui-unit-test",
]

INTEGRATION_TEST_SERVICES = [
    "url-input-integration-test",
    "auth-integration-test",
    "scraper-integration-test",
    "analyzer-integration-test",
    "clustering-integration-test",
    "export-integration-test",
    "session-integration-test",
    "monitoring-integration-test",
    "api-gateway-integration-test",
]


def compose_test_command(*args: str) -> List[str]:
    """Build a docker compose command targeting the test compose file."""
    return ["docker", "compose", "-f", TEST_COMPOSE_FILE, *args]


def run_docker_compose(args: List[str]) -> None:
    """Helper that wraps docker compose with project root cwd."""
    run_command(args, cwd=PROJECT_ROOT)


def run_test_suite(test_type: str) -> None:
    """Execute dockerized test suites."""
    ensure_test_directories()

    if test_type == "unit":
        for service in UNIT_TEST_SERVICES:
            run_docker_compose(compose_test_command("up", "--build", "--abort-on-container-exit", service))
    elif test_type == "integration":
        run_docker_compose(compose_test_command("up", "-d", "test-qdrant", "test-ollama"))
        run_docker_compose(
            compose_test_command(
                "up",
                "--build",
                "--abort-on-container-exit",
                *INTEGRATION_TEST_SERVICES,
            )
        )
    elif test_type == "e2e":
        run_docker_compose(compose_test_command("up", "-d", "test-qdrant", "test-ollama", "test-api-gateway", "test-web-ui"))
        run_docker_compose(compose_test_command("up", "--build", "--abort-on-container-exit", "e2e-test-runner"))
    elif test_type == "performance":
        run_docker_compose(compose_test_command("up", "-d", "test-qdrant", "test-ollama", "test-api-gateway"))
        run_docker_compose(compose_test_command("up", "--abort-on-container-exit", "load-test-runner"))
    elif test_type == "all":
        run_test_suite("unit")
        run_test_suite("integration")
        run_test_suite("e2e")
    else:
        raise SystemExit(f"Unknown test suite: {test_type}")


def collect_test_artifacts() -> None:
    """Copy test artifacts from running containers to the host."""
    try:
        result = run_command(
            ["docker", "ps", "-a", "--filter", "name=test", "--format", "{{.Names}}"],
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        return

    containers = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    for container in containers:
        run_command(["docker", "cp", f"{container}:/app/test-results/.", "test-results/"], check=False)
        run_command(["docker", "cp", f"{container}:/app/coverage/.", "coverage/"], check=False)


def generate_test_reports() -> None:
    """Trigger report generation container."""
    run_docker_compose(compose_test_command("up", "--build", "--no-deps", "test-report-aggregator"))


def cmd_test(args: argparse.Namespace) -> None:
    """Run dockerized test suites."""
    require_docker()
    test_type = args.type

    print(textwrap.dedent(f"""
        Running tests
        -------------
        Type: {test_type}
        Cleanup: {not args.skip_cleanup}
    """).strip())

    try:
        run_test_suite(test_type)
        if not args.skip_artifacts:
            collect_test_artifacts()
            generate_test_reports()
    finally:
        if not args.skip_cleanup:
            run_docker_compose(compose_test_command("down", "-v"))


def cmd_models(_: argparse.Namespace) -> None:
    """Display the currently recommended Ollama models."""
    models_file = PROJECT_ROOT / "config" / "models.json"
    if not models_file.exists():
        raise SystemExit("config/models.json is missing; cannot display model recommendations.")

    data = json.loads(models_file.read_text())
    llm_models = data.get("llm_models", {})
    embedding_models = data.get("embedding_models", {})

    print("LLM models:")
    for key, meta in llm_models.items():
        provider = meta.get("provider", "Unknown")
        description = meta.get("description", "")
        size = meta.get("size", "n/a")
        print(f"  - {key} ({provider}, {size}): {description}")

    print("\nEmbedding models:")
    for key, meta in embedding_models.items():
        provider = meta.get("provider", "Unknown")
        description = meta.get("description", "")
        size = meta.get("size", "n/a")
        print(f"  - {key} ({provider}, {size}): {description}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tab Organizer CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start", help="Start the application stack")
    start_parser.set_defaults(func=cmd_start)
    start_parser.add_argument(
        "--profile",
        choices=["auto", "cpu", "gpu", "none"],
        default="auto",
        help="Docker compose profile for Ollama. 'auto' picks gpu when available.",
    )
    start_parser.add_argument(
        "--ollama-mode",
        choices=["auto", "local", "docker"],
        default="auto",
        help="Prefer a locally running Ollama instance or the dockerized service.",
    )
    start_parser.add_argument("--pull", action="store_true", help="Run docker compose pull before starting.")
    start_parser.add_argument("--build", action="store_true", help="Run docker compose build before starting.")
    start_parser.add_argument("--quiet-pull", action="store_true", help="Pass --quiet-pull to docker compose up.")

    stop_parser = subparsers.add_parser("stop", help="Stop the application stack")
    stop_parser.set_defaults(func=cmd_stop)
    stop_parser.add_argument("--volumes", action="store_true", help="Remove volumes when stopping.")

    restart_parser = subparsers.add_parser("restart", help="Restart the application stack")
    restart_parser.set_defaults(func=cmd_restart)
    restart_parser.add_argument(
        "--profile",
        choices=["auto", "cpu", "gpu", "none"],
        default="auto",
    )
    restart_parser.add_argument(
        "--ollama-mode",
        choices=["auto", "local", "docker"],
        default="auto",
    )
    restart_parser.add_argument("--pull", action="store_true")
    restart_parser.add_argument("--build", action="store_true")
    restart_parser.add_argument("--quiet-pull", action="store_true")

    status_parser = subparsers.add_parser("status", help="Show docker compose status")
    status_parser.set_defaults(func=cmd_status)

    logs_parser = subparsers.add_parser("logs", help="Tail service logs")
    logs_parser.set_defaults(func=cmd_logs)
    logs_parser.add_argument("service", nargs="?", help="Optional service name to filter logs.")

    test_parser = subparsers.add_parser("test", help="Run the dockerized test workflow")
    test_parser.set_defaults(func=cmd_test)
    test_parser.add_argument(
        "--type",
        choices=["unit", "integration", "e2e", "performance", "all"],
        default="all",
        help="Select which suite to execute.",
    )
    test_parser.add_argument("--skip-cleanup", action="store_true", help="Leave test containers running after completion.")
    test_parser.add_argument("--skip-artifacts", action="store_true", help="Skip artifact collection and report aggregation.")

    models_parser = subparsers.add_parser("models", help="Print model recommendations from config/models.json")
    models_parser.set_defaults(func=cmd_models)

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "profile", None) == "none":
        args.profile = ""

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    except SystemExit as exc:
        raise exc
    except Exception as exc:  # pragma: no cover - guard rail
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
