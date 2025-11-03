#!/usr/bin/env python3
"""Interactive environment bootstrapper for Tab Organizer.

This script prepares the project for first-time use by configuring the .env
file, selecting model providers, and pulling required docker images.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple
from urllib import request, error as url_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
ENV_TEMPLATE = PROJECT_ROOT / ".env.example"


OLLAMA_LLM_OPTIONS: List[Tuple[str, str]] = [
    ("qwen3:1.7b", "Ultra-efficient with thinking mode (~1.1GB download)"),
    ("phi4:3.8b", "GPU-optimized, strong coding skills (~2.3GB download)"),
    ("gemma3n:e4b", "Balanced Google multimodal model (~2.2GB download)"),
    ("qwen3:8b", "High-quality reasoning (~4.7GB download)"),
]

OLLAMA_EMBED_OPTIONS: List[Tuple[str, str]] = [
    ("nomic-embed-text", "Best general-purpose embeddings (~274MB)"),
    ("mxbai-embed-large", "Highest quality embeddings (~669MB)"),
    ("all-minilm", "Lightweight and fast (~90MB)"),
]

CLAUDE_LLM_OPTIONS: List[Tuple[str, str]] = [
    ("claude-3-5-sonnet-20241022", "Balanced Claude 3.5 Sonnet (recommended)"),
    ("claude-3-5-haiku-20241022", "Cost-efficient Claude 3.5 Haiku"),
    ("claude-3-opus-20240229", "Highest reasoning performance Claude 3 Opus"),
]

CLAUDE_EMBED_OPTIONS: List[Tuple[str, str]] = [
    ("claude-embedding-1", "Anthropic Claude Embedding v1 (3k context)"),
    ("claude-embedding-1-large", "Anthropic Claude Embedding v1 Large (long context)"),
]


def run_command(command: List[str], *, check: bool = True, capture_output: bool = False) -> subprocess.CompletedProcess:
    """Execute a shell command in the project root."""
    print(f"→ {' '.join(command)}")
    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        check=check,
        capture_output=capture_output,
        text=True,
    )


def update_env_var(key: str, value: str) -> None:
    """Insert or update keys in the environment file."""
    if not ENV_FILE.exists():
        return

    lines = ENV_FILE.read_text().splitlines()
    updated = False
    for idx, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[idx] = f"{key}={value}"
            updated = True
            break
    if not updated:
        lines.append(f"{key}={value}")

    ENV_FILE.write_text("\n".join(lines) + "\n")


def ensure_env_file() -> None:
    """Guarantee that .env exists by copying the template when necessary."""
    if ENV_FILE.exists():
        return
    if not ENV_TEMPLATE.exists():
        raise SystemExit("Missing .env template; cannot initialize environment.")
    ENV_FILE.write_text(ENV_TEMPLATE.read_text())
    print("Created .env from .env.example. Update sensitive values after this setup.")


def ensure_logs_dir() -> None:
    """Create the logs directory that docker-compose expects."""
    (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)


def require_docker() -> None:
    """Validate docker and docker compose availability."""
    try:
        run_command(["docker", "--version"], capture_output=True)
        run_command(["docker", "compose", "version"], capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise SystemExit("Docker Compose V2 is required. Please install or update Docker.")


def detect_local_ollama(timeout: float = 2.0) -> bool:
    """Check if Ollama is already running on the host."""
    try:
        request.urlopen("http://127.0.0.1:11434/api/tags", timeout=timeout)
        return True
    except (url_error.URLError, url_error.HTTPError):
        return False


def detect_gpu() -> bool:
    """Best-effort GPU detection using nvidia-smi."""
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def resolve_profile(preference: str) -> str:
    """Translate requested profile into a concrete docker compose profile."""
    if preference == "auto":
        return "gpu" if detect_gpu() else "cpu"
    if preference in {"gpu", "cpu"}:
        return preference
    return ""


def prompt_choice(message: str, options: List[Tuple[str, str]], default_index: int = 0) -> str:
    """Prompt the user to pick an option, falling back to defaults when non-interactive."""
    if not sys.stdin.isatty():
        return options[default_index][0]

    print(message)
    for idx, (value, description) in enumerate(options, start=1):
        default_marker = " (default)" if idx - 1 == default_index else ""
        print(f"  {idx}. {value} — {description}{default_marker}")

    while True:
        response = input("Select an option by number (press Enter for default): ").strip()
        if not response:
            return options[default_index][0]
        if response.isdigit():
            choice = int(response)
            if 1 <= choice <= len(options):
                return options[choice - 1][0]
        print("Invalid selection. Please enter a valid number.")


def prompt_text(message: str, *, required: bool = False, default: str | None = None) -> str:
    """Prompt for free-form text."""
    if not sys.stdin.isatty():
        if required and default is None:
            raise SystemExit(f"{message} is required in non-interactive mode.")
        return default or ""

    while True:
        response = input(f"{message}: ").strip()
        if response:
            return response
        if not required:
            return default or ""
        print("This value is required.")


def configure_ollama(args: argparse.Namespace) -> Dict[str, str]:
    """Configure the environment to use Ollama-based models."""
    print(
        textwrap.dedent(
            """
            Configuring Ollama provider
            --------------------------
            Selecting Ollama means large models (2–5GB for LLMs, 90–700MB for embeddings)
            will be downloaded the first time you start the stack. Ensure you have
            sufficient disk space and bandwidth.
            """
        ).strip()
    )

    llm_model = args.ollama_llm or prompt_choice("Choose an Ollama LLM model:", OLLAMA_LLM_OPTIONS, default_index=0)
    embedding_model = args.ollama_embedding or prompt_choice(
        "Choose an Ollama embedding model:", OLLAMA_EMBED_OPTIONS, default_index=0
    )

    update_env_var("LLM_PROVIDER", "ollama")
    update_env_var("EMBEDDING_PROVIDER", "ollama")
    update_env_var("OLLAMA_MODEL", llm_model)
    update_env_var("OLLAMA_EMBEDDING_MODEL", embedding_model)
    update_env_var("OLLAMA_ENABLED", "true")

    mode = args.ollama_mode
    if mode == "auto":
        mode = "local" if detect_local_ollama() else "docker"

    if mode == "local":
        update_env_var("OLLAMA_URL", "http://localhost:11434")
        print("Detected local Ollama instance; dockerized Ollama will be skipped.")
        use_local = True
    else:
        update_env_var("OLLAMA_URL", "http://ollama:11434")
        use_local = False
        print("Ollama will run inside Docker. The first start may take several minutes while models download.")

    compose_profile = resolve_profile(args.profile)

    return {"provider": "ollama", "use_local": str(use_local), "compose_profile": compose_profile}


def configure_claude(args: argparse.Namespace) -> Dict[str, str]:
    """Configure the environment to use Anthropic Claude models."""
    print(
        textwrap.dedent(
            """
            Configuring Claude provider
            ---------------------------
            Claude uses Anthropic's hosted APIs. You will need an ANTHROPIC_API_KEY
            with access to the selected models. No large local downloads are required.
            """
        ).strip()
    )

    llm_model = args.claude_llm or prompt_choice(
        "Choose a Claude LLM model:", CLAUDE_LLM_OPTIONS, default_index=0
    )
    embedding_model = args.claude_embedding or prompt_choice(
        "Choose a Claude embedding model:", CLAUDE_EMBED_OPTIONS, default_index=0
    )
    api_key = args.anthropic_key or prompt_text(
        "Enter your ANTHROPIC_API_KEY (leave blank to keep existing value)", required=False, default=""
    )

    update_env_var("LLM_PROVIDER", "anthropic")
    update_env_var("EMBEDDING_PROVIDER", "anthropic")
    update_env_var("CLAUDE_MODEL", llm_model)
    update_env_var("CLAUDE_EMBEDDING_MODEL", embedding_model)
    update_env_var("OLLAMA_ENABLED", "false")
    update_env_var("OLLAMA_URL", "")
    if api_key:
        update_env_var("ANTHROPIC_API_KEY", api_key)

    print("Configured project to use Claude APIs. Ensure services know how to read the new provider settings.")
    return {"provider": "claude", "use_local": "false", "compose_profile": ""}


def perform_docker_tasks(provider_info: Dict[str, str], args: argparse.Namespace) -> None:
    """Pull and build docker images according to the selected provider."""
    if args.skip_pull and args.skip_build:
        return

    profile = provider_info.get("compose_profile") or ""
    profile_args = ["--profile", profile] if profile and profile != "none" else []

    if not args.skip_pull:
        run_command(["docker", "compose", *profile_args, "pull"])
    if not args.skip_build:
        run_command(["docker", "compose", *profile_args, "build"])


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project bootstrap utility")
    parser.add_argument(
        "--provider",
        choices=["ollama", "claude"],
        help="Preselect the preferred model provider. Default prompts interactively.",
    )
    parser.add_argument(
        "--ollama-mode",
        choices=["auto", "local", "docker"],
        default="auto",
        help="Prefer a local or dockerized Ollama installation.",
    )
    parser.add_argument(
        "--profile",
        choices=["auto", "cpu", "gpu", "none"],
        default="auto",
        help="Docker compose profile to enable when preparing Ollama images.",
    )
    parser.add_argument("--ollama-llm", help="Explicit Ollama LLM model selection.")
    parser.add_argument("--ollama-embedding", help="Explicit Ollama embedding model selection.")
    parser.add_argument("--claude-llm", help="Explicit Claude LLM selection.")
    parser.add_argument("--claude-embedding", help="Explicit Claude embedding selection.")
    parser.add_argument("--anthropic-key", help="Anthropic API key (Claude provider).")
    parser.add_argument("--skip-pull", action="store_true", help="Skip docker compose pull.")
    parser.add_argument("--skip-build", action="store_true", help="Skip docker compose build.")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    require_docker()
    ensure_env_file()
    ensure_logs_dir()

    provider = args.provider
    if provider is None:
        provider = "ollama" if sys.stdin.isatty() else "ollama"
        if sys.stdin.isatty():
            response = prompt_text("Use Ollama for local models? (yes/no)", default="yes")
            provider = "ollama" if response.lower() in {"y", "yes", ""} else "claude"

    if provider == "ollama":
        provider_info = configure_ollama(args)
    else:
        provider_info = configure_claude(args)

    perform_docker_tasks(provider_info, args)

    print("Initialization complete. Next steps:")
    print("  1. Review and adjust .env as needed.")
    print("  2. Start the stack with ./scripts/cli.py start")
    print("  3. Run tests with ./scripts/cli.py test --type all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
