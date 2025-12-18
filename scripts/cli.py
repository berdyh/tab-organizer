#!/usr/bin/env python3
"""Tab Organizer CLI - Unified management tool."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
DOCKER_COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"


def run_command(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a shell command."""
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        check=check,
        capture_output=capture,
        text=True,
    )


def docker_compose(*args: str, profiles: list[str] = None) -> subprocess.CompletedProcess:
    """Run docker compose command."""
    cmd = ["docker", "compose", "-f", str(DOCKER_COMPOSE_FILE)]
    
    if profiles:
        for profile in profiles:
            cmd.extend(["--profile", profile])
    
    cmd.extend(args)
    return run_command(cmd)


def cmd_start(args):
    """Start all services."""
    profiles = ["default"]
    if args.dev:
        profiles = ["dev"]
    
    extra_args = []
    if args.build:
        extra_args.append("--build")
    if args.detach:
        extra_args.append("-d")
    
    docker_compose("up", *extra_args, profiles=profiles)
    
    if args.detach:
        print("\n✅ Services started!")
        print("   Web UI:         http://localhost:8089")
        print("   Backend API:    http://localhost:8080")
        print("   AI Engine:      http://localhost:8090")
        print("   Browser Engine: http://localhost:8083")
        print("   Qdrant:         http://localhost:6333")
        print("   Ollama:         http://localhost:11434")


def cmd_stop(args):
    """Stop all services."""
    extra_args = []
    if args.volumes:
        extra_args.append("-v")
    
    docker_compose("down", *extra_args, profiles=["default", "dev"])
    print("✅ Services stopped")


def cmd_restart(args):
    """Restart services."""
    services = args.services if args.services else []
    docker_compose("restart", *services, profiles=["default"])
    print("✅ Services restarted")


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
    profile = f"test-{test_type}"
    
    print(f"Running {test_type} tests...")
    
    if test_type in ("integration", "e2e"):
        # Start dependencies first
        docker_compose("up", "-d", profiles=["default"])
    
    docker_compose("run", "--rm", f"test-{test_type}", profiles=[profile])


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
        print("✅ Created .env from .env.example")
    
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
        run_command(["docker", "exec", "tab-organizer-ollama", "ollama", "pull", "llama3.2"], check=False)
        run_command(["docker", "exec", "tab-organizer-ollama", "ollama", "pull", "nomic-embed-text"], check=False)
    
    print("\n✅ Initialization complete!")
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
    
    print("✅ Cleanup complete")


def cmd_shell(args):
    """Open a shell in a service container."""
    service = args.service
    run_command([
        "docker", "exec", "-it", f"tab-organizer-{service}", "/bin/bash"
    ], check=False)


def main():
    parser = argparse.ArgumentParser(
        description="Tab Organizer CLI - Unified management tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s init --build --models    Initialize project with images and models
  %(prog)s start -d                 Start all services in background
  %(prog)s start --build            Rebuild and start services
  %(prog)s stop                     Stop all services
  %(prog)s logs -f web-ui           Follow web-ui logs
  %(prog)s test --type unit         Run unit tests
  %(prog)s models --pull llama3.2   Pull a model
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # start
    start_parser = subparsers.add_parser("start", help="Start services")
    start_parser.add_argument("--build", "-b", action="store_true", help="Build images before starting")
    start_parser.add_argument("--detach", "-d", action="store_true", help="Run in background")
    start_parser.add_argument("--dev", action="store_true", help="Use development profile")
    start_parser.set_defaults(func=cmd_start)
    
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
    test_parser.add_argument("--type", "-t", choices=["unit", "integration", "e2e"], help="Test type")
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
    shell_parser.add_argument("service", choices=["backend", "ai", "browser", "ui", "qdrant", "ollama"])
    shell_parser.set_defaults(func=cmd_shell)
    
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
