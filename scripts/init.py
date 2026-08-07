#!/usr/bin/env python3
"""Interactive environment bootstrapper for Tab Organizer.

This script prepares the project for first-time use by configuring the .env
file, selecting model providers, and pulling required docker images.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple
from urllib import request, error as url_error

# Import configuration loader
sys.path.append(str(Path(__file__).parent.parent))
from config.config_loader import get_ai_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
ENV_TEMPLATE = PROJECT_ROOT / ".env.example"

# .env carries provider API keys and all four service bearer tokens, so it is
# created owner-only -- the same rule `scripts/cli.py` already applies to
# `data/service-tokens.json`.
SECRET_FILE_MODE = 0o600



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


def _target_env_mode() -> int:
    """The mode .env should end up with: never wider than owner-only.

    Preserving the existing mode was the first version of this fix, and it
    remediated nobody. The bug being fixed IS what produced today's 0644 files
    (`Path.write_text` under the stock 022 umask, whose mode `replace` then
    carried onto .env), so "a mode the operator deliberately set" is
    indistinguishable from "the mode our own bug left behind" -- and every
    existing install would have kept its group/world-readable secrets forever.
    A `cp .env.example .env` produces the same 0644.

    So a wider-than-0600 mode is narrowed, once, out loud. Anything already at
    or below 0600 is left exactly as it is (an operator running 0400 keeps it).
    """
    if not ENV_FILE.exists():
        return SECRET_FILE_MODE
    current = ENV_FILE.stat().st_mode & 0o777
    if current & 0o077:
        print(
            f"Narrowing {ENV_FILE.name} from {current:04o} to {SECRET_FILE_MODE:04o}: "
            "it holds provider API keys and all four service bearer tokens, and was "
            "readable by other local accounts."
        )
        return SECRET_FILE_MODE
    return current


def _write_env_file_atomically(text: str) -> None:
    """Replace .env's contents in one atomic operation (temp file + rename).

    `Path.replace` is an atomic rename on the same filesystem (POSIX and
    Windows both guarantee this), so a reader -- or a crash/IO error hitting
    this process -- only ever sees the old file in full or the new file in
    full, never a truncated or half-written one.

    The temp file is created 0600 and .env's existing mode is restored onto it
    before the rename. `Path.write_text` would create the temp file with the
    process umask (0644 under the stock 022), and `replace` carries the TEMP
    file's mode onto the destination -- so an owner-only .env silently became
    group/world-readable on the next write. .env holds provider API keys and
    all four service bearer tokens, so that is a credential disclosure, not a
    cosmetic permission drift. Widening is what this prevents; a mode the
    operator deliberately set is preserved as-is.
    """
    tmp_path = ENV_FILE.with_name(ENV_FILE.name + ".tmp")
    mode = _target_env_mode()
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, SECRET_FILE_MODE)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.chmod(tmp_path, mode)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    tmp_path.replace(ENV_FILE)


def update_env_vars(pairs: Dict[str, str]) -> None:
    """Insert or update multiple keys in one read-modify-write pass.

    Callers that need to change several related keys together (for example
    AI_PROVIDER + LLM_MODEL + EMBEDDING_PROVIDER + EMBEDDING_MODEL +
    EMBEDDING_DIMENSIONS, as `cli.py configure-provider` does) must apply all
    of them in the same pass: calling `update_env_var` once per key opens,
    reads, and rewrites the whole file N separate times, so an IO failure (or
    a crash) between calls leaves `.env` with only some of the keys updated --
    a torn write across an otherwise-atomic-looking selection. Batching into
    one in-memory edit plus one atomic write removes that window entirely.
    """
    if not ENV_FILE.exists():
        return

    lines = ENV_FILE.read_text().splitlines()
    seen: set[str] = set()
    for idx, line in enumerate(lines):
        for key, value in pairs.items():
            if line.startswith(f"{key}="):
                # EVERY assignment of the key is rewritten, not just the first.
                # A .env may legally carry a key twice, and every consumer
                # (docker compose, python-dotenv, `source`) takes the LAST one
                # -- so stopping at the first match let `configure-provider`
                # report that it had written a new provider while the stack
                # kept booting on the stale duplicate below it.
                lines[idx] = f"{key}={value}"
                seen.add(key)
                break
    for key, value in pairs.items():
        if key not in seen:
            lines.append(f"{key}={value}")

    _write_env_file_atomically("\n".join(lines) + "\n")


def update_env_var(key: str, value: str) -> None:
    """Insert or update a single key in the environment file."""
    update_env_vars({key: value})


def update_embedding_model_env(embedding_model: str) -> None:
    """Set embedding model and clear stale dimension overrides."""
    update_env_vars({"EMBEDDING_MODEL": embedding_model, "EMBEDDING_DIMENSIONS": ""})


def ensure_env_file() -> None:
    """Guarantee that .env exists by copying the template when necessary."""
    if ENV_FILE.exists():
        return
    if not ENV_TEMPLATE.exists():
        raise SystemExit("Missing .env template; cannot initialize environment.")
    # Created owner-only from the start: the very next thing a stock install
    # does is write API keys and four bearer tokens into this file, and a
    # 0644 .env would have handed them to every local account.
    fd = os.open(ENV_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, SECRET_FILE_MODE)
    with os.fdopen(fd, "w") as handle:
        handle.write(ENV_TEMPLATE.read_text())
    os.chmod(ENV_FILE, SECRET_FILE_MODE)
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
    
    # Get AI configuration
    ai_config = get_ai_config()
    
    # Get available Ollama models
    ollama_llm_models = ai_config.get_provider_models("ollama", "llm")
    ollama_embed_models = ai_config.get_provider_models("ollama", "embedding")
    
    # Convert to choice format for prompt
    llm_options = [(model, ai_config.format_model_description(model)) for model in ollama_llm_models]
    embed_options = [(model, ai_config.format_model_description(model)) for model in ollama_embed_models]
    
    llm_model = args.ollama_llm or prompt_choice("Choose an Ollama LLM model:", llm_options, default_index=0)
    embedding_model = args.ollama_embedding or prompt_choice(
        "Choose an Ollama embedding model:", embed_options, default_index=0
    )

    update_env_var("AI_PROVIDER", "ollama")
    update_env_var("EMBEDDING_PROVIDER", "ollama")
    update_env_var("LLM_MODEL", llm_model)
    update_embedding_model_env(embedding_model)

    mode = args.ollama_mode
    if mode == "auto":
        mode = "local" if detect_local_ollama() else "docker"

    if mode == "local":
        update_env_var("OLLAMA_HOST", "http://localhost:11434")
        print("Detected local Ollama instance; dockerized Ollama will be skipped.")
        use_local = True
    else:
        update_env_var("OLLAMA_HOST", "http://ollama:11434")
        use_local = False
        print("Ollama will run inside Docker. The first start may take several minutes while models download.")

    compose_profile = resolve_profile(args.profile)

    return {"provider": "ollama", "use_local": str(use_local), "compose_profile": compose_profile}


def get_embedding_provider_options(ai_config) -> List[Tuple[str, str]]:
    """Return embedding-capable providers that have at least one embedding model."""
    options = []
    for provider in ai_config.get_all_providers():
        if not ai_config.is_provider_supported(provider, "embeddings"):
            continue
        if not ai_config.get_provider_models(provider, "embedding"):
            continue
        options.append((provider, ai_config.get_provider_config(provider).get("type", "")))
    return options


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
    
    # Get AI configuration
    ai_config = get_ai_config()
    
    # Get available Claude models
    claude_llm_models = ai_config.get_provider_models("anthropic", "llm")
    
    # Convert to choice format for prompt
    llm_options = [(model, ai_config.format_model_description(model)) for model in claude_llm_models]
    
    llm_model = args.claude_llm or prompt_choice(
        "Choose a Claude LLM model:", llm_options, default_index=0
    )
    
    # Claude doesn't support embeddings, so we need to choose another provider
    print("\nClaude does not provide embeddings. Selecting embedding provider...")
    embed_provider_options = get_embedding_provider_options(ai_config)
    if not embed_provider_options:
        raise SystemExit("No embedding-capable providers with embedding models are configured.")

    supported_embedding_providers = [provider for provider, _ in embed_provider_options]
    if args.claude_embedding_provider:
        if args.claude_embedding_provider not in supported_embedding_providers:
            supported = ", ".join(supported_embedding_providers)
            raise SystemExit(
                f"--claude-embedding-provider '{args.claude_embedding_provider}' does not support embeddings "
                f"or has no embedding models. Supported providers: {supported}."
            )
        embed_provider = args.claude_embedding_provider
    else:
        embed_provider = prompt_choice("Choose embedding provider:", embed_provider_options, default_index=0)
    
    # Get models for selected embedding provider
    embed_models = ai_config.get_provider_models(embed_provider, "embedding")
    if not embed_models:
        raise SystemExit(f"Provider '{embed_provider}' has no embedding models configured.")
    embed_options = [(model, ai_config.format_model_description(model)) for model in embed_models]
    embedding_model = args.claude_embedding or prompt_choice(
        f"Choose a {embed_provider} embedding model:", embed_options, default_index=0
    )
    
    api_key = args.anthropic_key or prompt_text(
        "Enter your ANTHROPIC_API_KEY (leave blank to keep existing value)", required=False, default=""
    )
    
    if api_key:
        update_env_var("ANTHROPIC_API_KEY", api_key)

    update_env_var("AI_PROVIDER", "anthropic")
    update_env_var("EMBEDDING_PROVIDER", embed_provider)
    update_env_var("LLM_MODEL", llm_model)
    update_embedding_model_env(embedding_model)

    return {"provider": "claude", "use_local": "false", "compose_profile": ""}


def _select_embedding_provider(ai_config, requested_provider: str | None) -> str:
    embed_provider_options = get_embedding_provider_options(ai_config)
    if not embed_provider_options:
        raise SystemExit("No embedding-capable providers with embedding models are configured.")

    supported_embedding_providers = [provider for provider, _ in embed_provider_options]
    if requested_provider:
        if requested_provider not in supported_embedding_providers:
            supported = ", ".join(supported_embedding_providers)
            raise SystemExit(
                f"Embedding provider '{requested_provider}' does not support embeddings "
                f"or has no embedding models. Supported providers: {supported}."
            )
        return requested_provider

    default_index = supported_embedding_providers.index("ollama") if "ollama" in supported_embedding_providers else 0
    return prompt_choice("Choose embedding provider:", embed_provider_options, default_index=default_index)


def _select_embedding_model(ai_config, provider: str, requested_model: str | None) -> str:
    embed_models = ai_config.get_provider_models(provider, "embedding")
    if not embed_models:
        raise SystemExit(f"Provider '{provider}' has no embedding models configured.")
    if requested_model:
        return requested_model
    embed_options = [(model, ai_config.format_model_description(model)) for model in embed_models]
    return prompt_choice(f"Choose a {provider} embedding model:", embed_options, default_index=0)


def configure_openrouter(args: argparse.Namespace) -> Dict[str, str]:
    """Configure OpenRouter as the HTTP LLM/embedding provider."""
    print(
        textwrap.dedent(
            """
            Configuring OpenRouter provider
            --------------------------------
            OpenRouter uses HTTPS API calls and requires OPENROUTER_API_KEY. Use
            claude_code, codex_cli, or codex_acp when you want local subscription
            CLI routing instead of API-key routing.
            """
        ).strip()
    )

    ai_config = get_ai_config()
    llm_models = ai_config.get_provider_models("openrouter", "llm")
    embed_models = ai_config.get_provider_models("openrouter", "embedding")
    llm_options = [(model, ai_config.format_model_description(model)) for model in llm_models]
    embed_options = [(model, ai_config.format_model_description(model)) for model in embed_models]

    llm_model = args.openrouter_llm or prompt_choice("Choose an OpenRouter LLM model:", llm_options, default_index=0)
    embedding_model = args.openrouter_embedding or prompt_choice(
        "Choose an OpenRouter embedding model:", embed_options, default_index=0
    )
    api_key = args.openrouter_key or prompt_text(
        "Enter your OPENROUTER_API_KEY (leave blank to keep existing value)",
        required=False,
        default="",
    )
    if api_key:
        update_env_var("OPENROUTER_API_KEY", api_key)

    update_env_var("AI_PROVIDER", "openrouter")
    update_env_var("EMBEDDING_PROVIDER", "openrouter")
    update_env_var("LLM_MODEL", llm_model)
    update_embedding_model_env(embedding_model)

    return {"provider": "openrouter", "use_local": "false", "compose_profile": ""}


def configure_subscription_cli(args: argparse.Namespace, provider: str) -> Dict[str, str]:
    """Configure local subscription CLI providers for LLM routing."""
    labels = {
        "claude_code": "Claude Code print mode",
        "codex_cli": "Codex CLI one-shot exec",
        "codex_acp": "Codex ACP harness",
    }
    print(
        textwrap.dedent(
            f"""
            Configuring {labels[provider]}
            {'-' * (14 + len(labels[provider]))}
            This provider uses local CLI subscription login state and does not
            require an Anthropic or OpenAI API key for LLM calls. Embeddings
            still need an embedding-capable provider such as Ollama or OpenRouter.
            The stock Docker AI image does not include these CLIs; run the AI
            engine on the host or use a custom image with the authenticated CLI.
            """
        ).strip()
    )

    ai_config = get_ai_config()
    llm_models = ai_config.get_provider_models(provider, "llm")
    llm_options = [(model, ai_config.format_model_description(model)) for model in llm_models]
    llm_model = args.subscription_llm or prompt_choice(
        f"Choose a {provider} LLM model:", llm_options, default_index=0
    )

    embed_provider = _select_embedding_provider(ai_config, args.subscription_embedding_provider)
    embedding_model = _select_embedding_model(
        ai_config,
        embed_provider,
        args.subscription_embedding,
    )

    update_env_var("AI_PROVIDER", provider)
    update_env_var("EMBEDDING_PROVIDER", embed_provider)
    update_env_var("LLM_MODEL", llm_model)
    update_embedding_model_env(embedding_model)

    if provider == "claude_code" and args.claude_code_command:
        update_env_var("CLAUDE_CODE_COMMAND", args.claude_code_command)
    if provider == "codex_cli" and args.codex_cli_command:
        update_env_var("CODEX_CLI_COMMAND", args.codex_cli_command)
    if provider == "codex_acp" and args.codex_acp_command:
        update_env_var("CODEX_ACP_COMMAND", args.codex_acp_command)

    return {"provider": provider, "use_local": "true", "compose_profile": ""}


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
        choices=["ollama", "claude", "openrouter", "claude_code", "codex_cli", "codex_acp"],
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
    parser.add_argument("--claude-embedding-provider", help="Embedding provider to pair with Anthropic Claude.")
    parser.add_argument("--claude-embedding", help="Explicit Claude embedding selection.")
    parser.add_argument("--anthropic-key", help="Anthropic API key (Claude provider).")
    parser.add_argument("--openrouter-llm", help="Explicit OpenRouter LLM selection.")
    parser.add_argument("--openrouter-embedding", help="Explicit OpenRouter embedding selection.")
    parser.add_argument("--openrouter-key", help="OpenRouter API key.")
    parser.add_argument("--subscription-llm", help="Explicit local subscription CLI LLM selection.")
    parser.add_argument(
        "--subscription-embedding-provider",
        help="Embedding provider to pair with claude_code, codex_cli, or codex_acp.",
    )
    parser.add_argument("--subscription-embedding", help="Explicit subscription-mode embedding model.")
    parser.add_argument("--claude-code-command", help="Command for Claude Code print mode.")
    parser.add_argument("--codex-cli-command", help="Command for Codex CLI exec mode.")
    parser.add_argument("--codex-acp-command", help="Command for Codex ACP/acpx mode.")
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
        if not sys.stdin.isatty():
            # SPEC-provider-routing.md R3: AI_PROVIDER/EMBEDDING_PROVIDER in
            # .env is read elsewhere as proof a human deliberately chose a
            # provider. A non-interactive run with no --provider has no such
            # proof, so it must refuse rather than silently write "ollama" --
            # that would forge the exact consent record R3 requires.
            raise SystemExit(
                "init.py will not choose a provider on your behalf: this "
                "session is not interactive and --provider was not given. "
                "Pass --provider explicitly (ollama, claude, openrouter, "
                "claude_code, codex_cli, codex_acp) -- that flag IS the "
                "deliberate choice (SPEC-provider-routing.md R3). Nothing "
                "was written to .env."
            )
        response = prompt_text("Use Ollama for local models? (yes/no)", default="yes")
        provider = "ollama" if response.lower() in {"y", "yes", ""} else "claude"

    if provider == "ollama":
        provider_info = configure_ollama(args)
    elif provider == "claude":
        provider_info = configure_claude(args)
    elif provider == "openrouter":
        provider_info = configure_openrouter(args)
    else:
        provider_info = configure_subscription_cli(args, provider)

    perform_docker_tasks(provider_info, args)

    print("Initialization complete. Next steps:")
    print("  1. Review and adjust .env as needed.")
    print("  2. Start the stack with ./scripts/cli.py start")
    print("  3. Run tests with ./scripts/cli.py test --type all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
