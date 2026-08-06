#!/usr/bin/env python3
"""Tab Organizer CLI - Unified management tool."""

import argparse
import asyncio
import ipaddress
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
SERVICE_TOKEN_FILE = PROJECT_ROOT / "data" / "service-tokens.json"
# One independent bearer token per trust scope. They must never be equal: the
# agent principal must not be able to open browser-engine's CDP/credential
# control plane, and the callback principal must not be able to open the
# ai-engine. `service-tokens.json` keeps them STABLE across restarts.
SERVICE_TOKEN_ENVS = (
    "AI_ENGINE_API_TOKEN",
    "BACKEND_CALLBACK_TOKEN",
    "BACKEND_AGENT_API_TOKEN",
    "BROWSER_ENGINE_API_TOKEN",
)
# The single-token store this replaced. It seeds AI_ENGINE_API_TOKEN only, so
# an existing install keeps its ai-engine token instead of rotating silently.
LEGACY_TOKEN_SEED_ENV = "AI_ENGINE_API_TOKEN"
DEFAULT_STACK_HEALTHCHECKS = (
    ("Backend Core", "http://localhost:8080/health"),
    ("AI Engine", "http://localhost:8090/health"),
    ("Browser Engine", "http://localhost:8083/health"),
)
WEB_UI_HEALTHCHECK = ("Web UI", "http://localhost:8089/_stcore/health")
# docker-compose.yml's `networks:` key, before Compose namespaces it as
# "<project>_tab-organizer-network" (project defaults to the checkout's
# directory name, so it differs per worktree/clone). `host-ai` discovers this
# network's gateway address to bind to instead of hardcoding a bridge IP.
COMPOSE_BRIDGE_NETWORK_SUFFIX = "tab-organizer-network"

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


def _read_service_token_store() -> dict[str, str]:
    """Return the persisted per-scope token map (empty when absent/corrupt)."""
    if not SERVICE_TOKEN_FILE.exists():
        return {}
    try:
        data = json.loads(SERVICE_TOKEN_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(key): str(value).strip()
        for key, value in data.items()
        if isinstance(value, str) and value.strip()
    }


def _write_service_token_store(store: dict[str, str]) -> None:
    """Persist the per-scope token map with owner-only permissions."""
    SERVICE_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    SERVICE_TOKEN_FILE.write_text(json.dumps(store, indent=2, sort_keys=True) + "\n")
    SERVICE_TOKEN_FILE.chmod(0o600)


def _legacy_single_token() -> str:
    """Read the pre-split single-token file used to seed one scope."""
    if not HOST_AI_TOKEN_FILE.exists():
        return ""
    try:
        return HOST_AI_TOKEN_FILE.read_text().strip()
    except OSError:
        return ""


def ensure_service_token(env_name: str) -> str:
    """Return the stable local bearer token for one service trust scope.

    Precedence: an explicit environment/.env value wins (never clobbered), then
    the persisted `data/service-tokens.json` entry, then — for
    `AI_ENGINE_API_TOKEN` only — the legacy single-token `data/host-ai-token`
    file so existing installs do not rotate that token during the split. A new
    independent `secrets.token_urlsafe(32)` value is minted otherwise. Every
    scope gets its OWN value: sharing one token across scopes collapses the
    agent, callback, ai-engine and browser-engine principals into one.
    """
    configured = os.getenv(env_name, "").strip()
    if configured:
        return configured

    store = _read_service_token_store()
    persisted = store.get(env_name, "").strip()
    if persisted:
        return persisted

    token = ""
    if env_name == LEGACY_TOKEN_SEED_ENV:
        token = _legacy_single_token()
    if not token:
        token = secrets.token_urlsafe(32)

    store[env_name] = token
    _write_service_token_store(store)
    return token


def ensure_host_ai_token() -> str:
    """Return the AI Engine token for container-to-host AI Engine calls."""
    return ensure_service_token("AI_ENGINE_API_TOKEN")


def set_env_default_if_blank(env: dict[str, str], key: str, value: str) -> None:
    """Set an env default when a copied .env left the key blank."""
    if not env.get(key, "").strip():
        env[key] = value


def service_env_with_tokens() -> dict[str, str]:
    """Return compose env with one INDEPENDENT auth token per service scope.

    These four values must stay pairwise distinct. A single shared value would
    make the agent token also open browser-engine's scrape/CDP/credential
    endpoints (they accept `BROWSER_ENGINE_API_TOKEN`), collapsing four trust
    scopes into one. Blank/unset keys are filled; user-provided values in the
    environment or `.env` are never overwritten.
    """
    env = os.environ.copy()
    for env_name in SERVICE_TOKEN_ENVS:
        set_env_default_if_blank(env, env_name, ensure_service_token(env_name))
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


def discover_docker_bridge_gateway() -> str | None:
    """Return this checkout's Docker bridge network gateway IP, or None.

    `host-ai` binds its uvicorn here instead of 0.0.0.0. Docker resolves the
    `host.docker.internal` extra_hosts entry every container gets to this
    same per-network gateway address (not a fixed constant -- Compose
    allocates each project's own subnet, so never hardcode e.g. 172.17.0.1,
    which is only the *default* bridge's address and may not even be the
    network this project's containers are attached to). Binding to the
    discovered gateway keeps the host-run AI Engine reachable from containers
    while nothing else on the LAN can route to it.
    """
    try:
        listed = subprocess.run(
            ["docker", "network", "ls", "--format", "{{.Name}}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if listed.returncode != 0:
        return None

    candidates = [
        name
        for name in (line.strip() for line in listed.stdout.splitlines())
        if name == COMPOSE_BRIDGE_NETWORK_SUFFIX
        or name.endswith(f"_{COMPOSE_BRIDGE_NETWORK_SUFFIX}")
    ]
    if len(candidates) != 1:
        # Zero: stack never started (network not created yet). More than
        # one: an ambiguous match we should not guess between either --
        # both fail closed via the None return.
        return None

    try:
        inspected = subprocess.run(
            [
                "docker",
                "network",
                "inspect",
                candidates[0],
                "--format",
                "{{range .IPAM.Config}}{{.Gateway}}{{end}}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if inspected.returncode != 0:
        return None

    gateway = inspected.stdout.strip()
    try:
        ipaddress.ip_address(gateway)
    except ValueError:
        return None
    return gateway


def resolve_host_ai_bind_host(requested: str | None) -> str:
    """Resolve the host `host-ai`'s uvicorn should bind.

    An explicit `--host` always wins verbatim, including "0.0.0.0" for an
    operator who deliberately wants LAN reach (the AI Engine is still
    token-authenticated either way -- this binding choice is defense in
    depth, not the only control). Left unset, this must never fall back to
    0.0.0.0: every other published port in this repo is loopback-only for
    exactly the reason a host-run AI Engine would otherwise be reachable from
    the whole LAN. Discovery failing must fail loudly rather than guess a
    default, so a misconfigured host never silently binds wide open.
    """
    if requested is not None:
        if requested == "0.0.0.0":
            print(
                "WARNING: --host 0.0.0.0 exposes the host AI Engine to the "
                "whole network (defense in depth still applies via "
                "AI_ENGINE_API_TOKEN, but this widens the attack surface "
                "beyond Docker on purpose).",
                file=sys.stderr,
            )
        return requested

    gateway = discover_docker_bridge_gateway()
    if gateway is None:
        raise SystemExit(
            "Could not determine the Docker bridge gateway to bind host-ai "
            f"to (looked for a Docker network named "
            f"'{COMPOSE_BRIDGE_NETWORK_SUFFIX}' or "
            f"'<project>_{COMPOSE_BRIDGE_NETWORK_SUFFIX}'). Run "
            "'./scripts/cli.py start -d --host-ai' once first so Compose "
            "creates the network, or pass --host explicitly (127.0.0.1 for "
            "host-only access, or the address 'docker network inspect "
            f"{COMPOSE_BRIDGE_NETWORK_SUFFIX}' reports for container "
            "access)."
        )
    return gateway


def _require_explicit_provider(
    explicit: str | None, env_var: str, role_label: str, flag_name: str
) -> str:
    """Return an explicitly-chosen provider; refuse rather than invent one.

    SPEC-provider-routing.md R1/R3: `AI_PROVIDER`/`EMBEDDING_PROVIDER` have no
    default anywhere, and an env var already set (in `.env` or the shell) IS
    the record of deliberate consent -- so is an explicit CLI flag. What is
    NOT consent is silently substituting a hardcoded provider ("claude_code",
    "ollama", "openrouter", ...) when neither was given: that forges the exact
    consent record R1/R3 exist to require, the same failure mode
    `_choose_provider_interactively` above refuses to commit for
    configure-provider. Every caller here (`host-ai`, `check-provider`) must
    fail closed the same way instead of picking a provider on the user's
    behalf.
    """
    value = (explicit or os.getenv(env_var, "")).strip()
    if value:
        return value
    raise SystemExit(
        f"code: provider_not_selected\n"
        f"cause: {env_var} is not set. This command does not pick a "
        f"{role_label} provider for you.\n"
        f"fix: Run ./scripts/cli.py configure-provider, or pass {flag_name} "
        f"explicitly, or set {env_var} in .env. Preferred: claude_code, "
        f"codex_cli or gemini_cli (uses your subscription). Metered: openrouter, openai, "
        f"gemini (requires an API key and consent). Local: ollama (free, "
        f"requires models pulled first)."
    )


def cmd_host_ai(args):
    """Run the AI engine on the host so it can use authenticated local CLIs."""
    load_env_file()

    env = os.environ.copy()
    provider = _require_explicit_provider(args.provider, "AI_PROVIDER", "LLM", "--provider")
    embedding_provider = _require_explicit_provider(
        args.embedding_provider, "EMBEDDING_PROVIDER", "embedding", "--embedding-provider"
    )
    env["AI_PROVIDER"] = provider
    env["EMBEDDING_PROVIDER"] = embedding_provider
    env["AI_ENGINE_API_TOKEN"] = ensure_service_token("AI_ENGINE_API_TOKEN")
    set_env_default_if_blank(
        env, "BACKEND_CALLBACK_TOKEN", ensure_service_token("BACKEND_CALLBACK_TOKEN")
    )
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
    if args.gemini_cli_command:
        env["GEMINI_CLI_COMMAND"] = args.gemini_cli_command

    bind_host = resolve_host_ai_bind_host(args.host)
    print(
        "Host AI engine mode uses your local CLI auth state. "
        "Start Docker with './scripts/cli.py start -d --host-ai' in another terminal."
    )
    print(f"   Binding to {bind_host}:{args.port} (see --host for other options)")
    run_command(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "services.ai_engine.app.main:app",
            "--host",
            bind_host,
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
    provider = _require_explicit_provider(args.provider, "AI_PROVIDER", "LLM", "--provider")
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


def _probe_client(ai_config):
    """Build a throwaway LLMClient purely to reach its shared runtime probes.

    The provider/model passed here are never used to talk to a provider --
    configure-provider only calls get_provider_runtime_state(). Both llm_config
    and embedding_config are passed explicitly (rather than left to default)
    so this does not depend on LLMClient's env-derived defaults, which are the
    exact lines SPEC-provider-routing.md R1/R2 are changing concurrently in
    services/ai-engine/app/core/llm_client.py.
    """
    from services.ai_engine.app.core.llm_client import (
        EmbeddingConfig,
        LLMClient,
        LLMConfig,
    )

    return LLMClient(
        LLMConfig(
            provider="ollama", model=ai_config.get_default_model("ollama", "llm") or ""
        ),
        EmbeddingConfig(
            provider="ollama",
            model=ai_config.get_default_model("ollama", "embedding") or "",
        ),
    )


def probe_llm_provider(provider: str) -> dict:
    """Probe real LLM availability for a provider.

    Isolated as its own module-level function so tests can monkeypatch this
    probe boundary instead of depending on what CLIs/keys happen to be
    installed/set on the machine running the test.
    """
    from config.config_loader import get_ai_config

    ai_config = get_ai_config()
    return _probe_client(ai_config).get_provider_runtime_state(provider, "llm")


def probe_embedding_provider(provider: str) -> dict:
    """Probe real embedding availability for a provider. See probe_llm_provider."""
    from config.config_loader import get_ai_config

    ai_config = get_ai_config()
    return _probe_client(ai_config).get_provider_runtime_state(provider, "embeddings")


def probe_ollama_installed_models(base_url: str, timeout: float = 2.0) -> set[str] | None:
    """Return the set of pulled Ollama model names, or None if unreachable.

    Implemented independently of services/ai-engine's LLMClient (rather than
    reaching into its private _ollama_installed_models helper) so
    configure-provider keeps working no matter how the concurrent R1/R2/R4
    refactor of that module lands.
    """
    try:
        with urllib.request.urlopen(
            f"{base_url.rstrip('/')}/api/tags", timeout=timeout
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    installed: set[str] = set()
    for item in payload.get("models", []) or []:
        if not isinstance(item, dict):
            continue
        for key in ("name", "model"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                name = value.strip()
                installed.add(name)
                if name.endswith(":latest"):
                    installed.add(name[: -len(":latest")])
    return installed


def pull_ollama_model(base_url: str, model: str, timeout: float = 1800.0) -> None:
    """Pull an Ollama model via its HTTP API, printing streamed status lines."""
    request_obj = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/pull",
        data=json.dumps({"model": model}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request_obj, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            status = event.get("status")
            if status:
                print(f"  ollama pull {model}: {status}")


def _ollama_base_url() -> str:
    """Resolve the Ollama endpoint the same way init.py/host-ai do: env override first."""
    return os.getenv("OLLAMA_HOST") or "http://localhost:11434"


def _provider_cost_model(ai_config, provider: str) -> str:
    return ai_config.get_provider_config(provider).get("cost_model", "unknown")


def _ordered_llm_candidates(ai_config) -> list[str]:
    """LLM providers worth probing, in catalog preference order.

    `routing.llm_preference_order` covers the preferred subscription CLIs;
    `routing.explicit_opt_in_required` covers every other provider a user can
    still deliberately choose (ollama, openrouter, openai, gemini, anthropic,
    deepseek). Concatenating them (deduped, first occurrence wins) reproduces
    the catalog's ascending `preference` numbers without hardcoding a second
    copy of that order here.
    """
    routing = ai_config.config.get("routing", {})
    ordered: list[str] = []
    for provider in list(routing.get("llm_preference_order", [])) + list(
        routing.get("explicit_opt_in_required", [])
    ):
        if provider not in ordered:
            ordered.append(provider)
    return ordered


def _probe_available_llm_providers(ai_config) -> list[tuple[str, dict]]:
    """Return (provider, probe_state) pairs for LLM providers verified usable.

    A provider whose binary/key/server probe fails is never returned here, so
    it can never reach the prompt_choice() menu or be written to .env.
    """
    available: list[tuple[str, dict]] = []
    for provider in _ordered_llm_candidates(ai_config):
        if not ai_config.is_provider_supported(provider, "llm"):
            continue
        state = probe_llm_provider(provider)
        if state.get("available"):
            print(f"  {provider}: available ({_provider_cost_model(ai_config, provider)})")
            available.append((provider, state))
        else:
            print(f"  {provider}: not available -- {state.get('reason') or 'unknown reason'}")
    return available


def _probe_available_embedding_providers(ai_config) -> list[tuple[str, dict]]:
    """Return (provider, probe_state) pairs for embedding providers verified usable.

    Filtered strictly by the catalog's `supports.embeddings` first -- a
    provider that cannot embed (the subscription CLIs, anthropic, deepseek)
    is never even probed, let alone offered, regardless of what a probe
    mock might return.

    The exclusion list is deliberately not written down here. This docstring
    used to name openrouter as the example of a provider that "serves none",
    which was false; because the filter itself reads the catalog, correcting
    the catalog fixed the behaviour and left the comment lying. Ask
    `is_provider_supported`, not this paragraph.
    """
    available: list[tuple[str, dict]] = []
    for provider in ai_config.get_all_providers():
        if not ai_config.is_provider_supported(provider, "embeddings"):
            continue
        if not ai_config.get_provider_models(provider, "embedding"):
            continue
        state = probe_embedding_provider(provider)
        if state.get("available"):
            print(f"  {provider}: available ({_provider_cost_model(ai_config, provider)})")
            available.append((provider, state))
        else:
            print(f"  {provider}: not available -- {state.get('reason') or 'unknown reason'}")
    return available


def _choose_provider_interactively(
    prompt_choice_fn,
    prompt_message: str,
    options: list[tuple[str, str]],
    default_index: int,
    available_names: list[str],
    flag_name: str,
) -> str:
    """Prompt for a provider choice; refuse to guess when non-interactive.

    AI_PROVIDER/EMBEDDING_PROVIDER in .env is read elsewhere as proof a human
    deliberately chose it (SPEC-provider-routing.md R3: "the env var is the
    record of consent"). `scripts/init.py`'s prompt_choice() silently returns
    `options[default_index]` when stdin is not a tty -- correct there, since
    init.py's defaults were always allowed under the old contract. It is
    wrong here: a
    piped/CI/non-interactive configure-provider run with no explicit flag
    would then write a provider nobody chose, and the resulting .env line
    would be indistinguishable from a real decision to every later reader,
    including the fail-closed checks (R1/R3) that trust it. So -- unlike
    model selection below, a within-provider detail -- provider selection
    fails closed instead of auto-picking.
    """
    if sys.stdin.isatty():
        return prompt_choice_fn(prompt_message, options, default_index=default_index)
    raise SystemExit(
        f"configure-provider will not choose a provider on your behalf: this "
        f"session is not interactive and {flag_name} was not given. Pass "
        f"{flag_name} explicitly -- that flag IS the deliberate choice "
        f"(SPEC-provider-routing.md R3) -- or run configure-provider attached "
        f"to a terminal. Verified available: {', '.join(available_names)}. "
        f"Nothing was written to .env."
    )


def _ensure_ollama_model_pulled(model: str, args: argparse.Namespace) -> str:
    """Verify an Ollama model is actually pulled, offering to pull it if not.

    WI0-B1 found the Ollama container running with zero models pulled, which
    is indistinguishable from working until the first request fails. Refuses
    to hand back a model that is not verifiably present.
    """
    base_url = _ollama_base_url()
    installed = probe_ollama_installed_models(base_url)
    if installed is None:
        raise SystemExit(
            f"Ollama is not reachable at {base_url}. Start it "
            f"(`docker compose up -d ollama`, or run Ollama locally) and retry "
            f"configure-provider."
        )
    if model in installed:
        return model

    print(f"Ollama model '{model}' is not pulled at {base_url}.")
    should_pull = bool(getattr(args, "yes", False))
    if not should_pull and sys.stdin.isatty():
        response = input(f"Pull '{model}' now? [y/N] ").strip().lower()
        should_pull = response in {"y", "yes"}

    if not should_pull:
        raise SystemExit(
            f"Refusing to write an unpulled Ollama model ('{model}'). Pull it first: "
            f"`docker exec tab-organizer-ollama ollama pull {model}` (or "
            f"`ollama pull {model}` if running Ollama on the host), then retry. "
            f"Or rerun with --yes to pull automatically."
        )

    print(f"Pulling {model} ...")
    pull_ollama_model(base_url, model)
    installed = probe_ollama_installed_models(base_url)
    if not installed or model not in installed:
        raise SystemExit(
            f"Pull did not leave '{model}' visible at {base_url}; refusing to write it."
        )
    return model


def cmd_configure_provider(args):
    """Probe real provider availability and write a verified selection to .env.

    This is the asking the service itself cannot do
    (docs/SPEC-provider-routing.md R6): probe which subscription CLIs are on
    PATH and authenticated, which API keys are present, and which Ollama
    models are actually pulled -- then present only what is genuinely usable,
    ordered by the catalog's routing preference, annotated with cost_model.
    Never writes AI_PROVIDER/EMBEDDING_PROVIDER to a value its own probe could
    not verify, and always leaves EMBEDDING_DIMENSIONS blank so it resolves
    from the catalog and cannot drift from EMBEDDING_MODEL.
    """
    from config.config_loader import get_ai_config
    from scripts.init import ensure_env_file, prompt_choice, update_env_vars

    ensure_env_file()
    load_env_file()

    ai_config = get_ai_config()

    print("Probing LLM providers (subscription CLIs first, then opt-in providers)...")
    llm_available = _probe_available_llm_providers(ai_config)
    if not llm_available:
        raise SystemExit(
            "No LLM provider is verified available. Install/authenticate one of:\n"
            "  claude_code -- `claude` on PATH, logged in (subscription)\n"
            "  codex_cli   -- `codex` on PATH, logged in (subscription)\n"
            "  codex_acp   -- `acpx` on PATH (subscription)\n"
            "  gemini_cli  -- `gemini` on PATH AND logged in, i.e.\n"
            "                 ~/.gemini/oauth_creds.json present (subscription).\n"
            "                 `gemini --version` alone is NOT enough: an\n"
            "                 unauthenticated CLI exits 0 there and then blocks\n"
            "                 forever on a browser-login prompt.\n"
            "  ollama      -- reachable local server with a model pulled (free, local)\n"
            "  openrouter / openai / gemini / anthropic / deepseek -- matching API key "
            "set in .env (metered)\n"
            "Nothing was written to .env."
        )

    llm_provider_names = [name for name, _ in llm_available]
    if args.provider:
        if args.provider not in llm_provider_names:
            raise SystemExit(
                f"--provider {args.provider} failed its availability probe (or is "
                f"unknown); refusing to write an unverified provider. Verified "
                f"available: {', '.join(llm_provider_names)}"
            )
        llm_provider = args.provider
    else:
        llm_options = [
            (name, _provider_cost_model(ai_config, name)) for name in llm_provider_names
        ]
        llm_provider = _choose_provider_interactively(
            prompt_choice,
            "Choose an LLM provider (only verified-available options shown):",
            llm_options,
            0,
            llm_provider_names,
            "--provider",
        )

    llm_models = ai_config.get_provider_models(llm_provider, "llm")
    default_llm_model = ai_config.get_default_model(llm_provider, "llm")
    default_index = llm_models.index(default_llm_model) if default_llm_model in llm_models else 0
    # Every menu row carries its route (provider + cost_model), never a bare
    # model name -- see config_loader.format_model_description.
    llm_model_options = [
        (m, ai_config.format_model_description(m)) for m in llm_models
    ]
    llm_model = args.llm_model or prompt_choice(
        f"Choose a {llm_provider} LLM model:", llm_model_options, default_index=default_index
    )

    if llm_provider == "ollama":
        llm_model = _ensure_ollama_model_pulled(llm_model, args)

    print("Probing embedding providers (catalog supports.embeddings only)...")
    embed_available = _probe_available_embedding_providers(ai_config)
    if not embed_available:
        raise SystemExit(
            "No embedding provider is verified available. AI_PROVIDER was not written: "
            "an unset/unverified EMBEDDING_PROVIDER is exactly the silent-default bug "
            "R1/R2 forbid. Pull an Ollama embedding model (e.g. nomic-embed-text) or set "
            "OPENROUTER_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY, then retry.\n"
            "Nothing was written to .env."
        )

    embed_provider_names = [name for name, _ in embed_available]
    if args.embedding_provider:
        if args.embedding_provider not in embed_provider_names:
            raise SystemExit(
                f"--embedding-provider {args.embedding_provider} cannot embed, or failed "
                f"its availability probe; refusing to write an unverified provider. "
                f"Verified available: {', '.join(embed_provider_names)}"
            )
        embedding_provider = args.embedding_provider
    else:
        default_index = (
            embed_provider_names.index("ollama") if "ollama" in embed_provider_names else 0
        )
        embed_options = [
            (name, _provider_cost_model(ai_config, name)) for name in embed_provider_names
        ]
        embedding_provider = _choose_provider_interactively(
            prompt_choice,
            "Choose an embedding provider (only verified-available options shown):",
            embed_options,
            default_index,
            embed_provider_names,
            "--embedding-provider",
        )

    embed_models = ai_config.get_provider_models(embedding_provider, "embedding")
    default_embed_model = ai_config.get_default_model(embedding_provider, "embedding")
    default_index = (
        embed_models.index(default_embed_model) if default_embed_model in embed_models else 0
    )
    embed_model_options = [
        (m, ai_config.format_model_description(m)) for m in embed_models
    ]
    embedding_model = args.embedding_model or prompt_choice(
        f"Choose a {embedding_provider} embedding model:",
        embed_model_options,
        default_index=default_index,
    )

    if embedding_provider == "ollama":
        embedding_model = _ensure_ollama_model_pulled(embedding_model, args)

    # All five keys land in ONE read-modify-write pass (update_env_vars), not
    # four/five sequential ones: a mid-sequence IO failure could otherwise
    # leave .env with e.g. AI_PROVIDER updated but EMBEDDING_PROVIDER stale --
    # a partial write scripts/MODULE.md treats as a hard constraint to avoid.
    # EMBEDDING_DIMENSIONS is written blank in the same pass so it resolves
    # from the catalog and cannot drift from EMBEDDING_MODEL.
    update_env_vars(
        {
            "AI_PROVIDER": llm_provider,
            "LLM_MODEL": llm_model,
            "EMBEDDING_PROVIDER": embedding_provider,
            "EMBEDDING_MODEL": embedding_model,
            "EMBEDDING_DIMENSIONS": "",
        }
    )

    print(
        f"Wrote AI_PROVIDER={llm_provider}, LLM_MODEL={llm_model}, "
        f"EMBEDDING_PROVIDER={embedding_provider}, EMBEDDING_MODEL={embedding_model} to .env."
    )
    print(
        "EMBEDDING_DIMENSIONS left blank -- it resolves from the model catalog at "
        "runtime so it cannot drift from EMBEDDING_MODEL."
    )


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
        ensure_service_token(mcp_tabs.AGENT_TOKEN_ENV),
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
    load_env_file()

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
        choices=[
            "claude_code",
            "codex_cli",
            "gemini_cli",
            "codex_acp",
            "openrouter",
            "ollama",
        ],
        help=(
            "LLM provider to run in the host AI engine; falls back to "
            "AI_PROVIDER in .env if set, otherwise required (this command "
            "never picks a provider for you -- run configure-provider first)"
        ),
    )
    host_ai_parser.add_argument("--llm-model", help="Override LLM_MODEL")
    host_ai_parser.add_argument(
        "--embedding-provider",
        help=(
            "Embedding provider to pair with the LLM provider; falls back "
            "to EMBEDDING_PROVIDER in .env if set, otherwise required"
        ),
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
    host_ai_parser.add_argument(
        "--gemini-cli-command",
        help="Override GEMINI_CLI_COMMAND, for example an absolute gemini path",
    )
    host_ai_parser.add_argument(
        "--host",
        default=None,
        help=(
            "Bind host. Default: auto-discover this checkout's Docker bridge "
            "gateway, so containers reach it via host.docker.internal but "
            "the LAN cannot; pass 0.0.0.0 explicitly to widen that on purpose"
        ),
    )
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
            "gemini_cli",
            "codex_acp",
            "deepseek",
            "gemini",
        ],
        help=(
            "Provider to check; falls back to AI_PROVIDER in .env if set, "
            "otherwise required"
        ),
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

    # configure-provider
    configure_provider_parser = subparsers.add_parser(
        "configure-provider",
        help="Probe real LLM/embedding provider availability and write a verified choice to .env",
    )
    configure_provider_parser.add_argument(
        "--provider",
        help="Preselect an LLM provider; refused if its availability probe fails.",
    )
    configure_provider_parser.add_argument(
        "--llm-model", help="Preselect the LLM model for --provider."
    )
    configure_provider_parser.add_argument(
        "--embedding-provider",
        help="Preselect the embedding provider; refused if it cannot embed or its probe fails.",
    )
    configure_provider_parser.add_argument(
        "--embedding-model", help="Preselect the embedding model for --embedding-provider."
    )
    configure_provider_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Auto-confirm pulling a missing Ollama model instead of prompting.",
    )
    configure_provider_parser.set_defaults(func=cmd_configure_provider)

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
        default=None,
        help=(
            "Local Chrome DevTools Protocol URL. Defaults to the service's "
            "own default (Browser Engine's http://host.docker.internal:9222) "
            "when omitted."
        ),
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
        default=None,
        help=(
            "Local Chrome DevTools Protocol URL. Defaults to the service's "
            "own default (Browser Engine's http://host.docker.internal:9222) "
            "when omitted."
        ),
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
