"""Harness for the frozen security-invariant suite.

Ground rules (frozen with the suite, see MODULE.md and README.md):

* Black-box only. Probes are HTTP requests to a service base URL, or
  observation of a subprocess a service spawned. Two flagged Python-seam
  exceptions (``sec_seam``) touch importable code directly and carry a
  TS-porting rule.
* Managed vs attached. ``SEC_BACKEND_URL`` / ``SEC_AI_URL`` /
  ``SEC_BROWSER_URL`` point the harness at already-running servers (attached
  mode). With none set, the harness runs each FastAPI app in-process through an
  ASGI transport with a fully controlled environment (managed mode). It gets
  that control by importing the Python module (``_load_app``), so managed mode
  works only against this stack. ``SEC_BOOT_*_CMD`` -- harness-launched servers
  in any language -- is PLANNED, NOT IMPLEMENTED; see ``README.md`` for what
  that costs a TS port and when it must land.
* Platform exclusion (plan decision 41). Any request whose path contains
  ``/platform/`` raises ``PlatformPathBlocked`` so the exclusion is enforced.
* Global redaction audit. Every response body is recorded and, at session
  teardown, asserted to contain no configured token value (SEC-27).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import secrets
import socket
import stat
import sys
import tempfile
import threading
import types
from pathlib import Path
from typing import Optional

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# Mode + token wiring
# ---------------------------------------------------------------------------

ATTACHED = bool(
    os.getenv("SEC_BACKEND_URL")
    or os.getenv("SEC_AI_URL")
    or os.getenv("SEC_BROWSER_URL")
)
MODE = "attached" if ATTACHED else "managed"

# Distinct tokens per scope so cross-acceptance is genuinely observable.
TOKEN_ENVS = {
    "ai": "AI_ENGINE_API_TOKEN",
    "callback": "BACKEND_CALLBACK_TOKEN",
    "agent": "BACKEND_AGENT_API_TOKEN",
    "browser": "BROWSER_ENGINE_API_TOKEN",
}

WRONG_TOKEN = "sec-wrong-" + secrets.token_hex(8)

_ORIGINAL_ENV: dict[str, Optional[str]] = {}
_TMP_DIR: Optional[tempfile.TemporaryDirectory] = None


def _set_env(key: str, value: Optional[str]) -> None:
    _ORIGINAL_ENV.setdefault(key, os.environ.get(key))
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


def _install_managed_env() -> None:
    """Generate a controlled per-run environment for in-process apps."""
    global _TMP_DIR
    _TMP_DIR = tempfile.TemporaryDirectory(prefix="sec-suite-")
    root = Path(_TMP_DIR.name)

    for scope, env_name in TOKEN_ENVS.items():
        _set_env(env_name, f"sec-{scope}-{secrets.token_hex(12)}")

    # Deterministic, writable storage for the in-process services.
    _set_env("BACKEND_DB_PATH", str(root / "backend.db"))
    _set_env("VECTOR_DB_PATH", str(root / "lancedb"))
    _set_env("AGENT_CLI_WORKDIR", str(root / "agent-workdir"))

    # Escape hatches and CLI providers default OFF unless a test opts in.
    for key in (
        "AI_ENGINE_ALLOW_UNAUTHENTICATED",
        "SCRAPE_ALLOW_PRIVATE_NETWORKS",
        "CLAUDE_CODE_COMMAND",
        "CODEX_CLI_COMMAND",
        "CODEX_ACP_COMMAND",
        "CLAUDE_CODE_DISABLE_TOOLS",
        "CODEX_CLI_ALLOW_UNTRUSTED_CONTEXT",
        "CODEX_CLI_SANDBOX",
        "AGENT_CLI_TIMEOUT",
        "CLAUDE_CODE_TIMEOUT",
        "CODEX_CLI_TIMEOUT",
    ):
        _set_env(key, None)


def _restore_env() -> None:
    for key, value in _ORIGINAL_ENV.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    _ORIGINAL_ENV.clear()
    global _TMP_DIR
    if _TMP_DIR is not None:
        with contextlib.suppress(Exception):
            _TMP_DIR.cleanup()
        _TMP_DIR = None


# ---------------------------------------------------------------------------
# Fake playwright shim so the browser app imports for validation-only probes.
# ---------------------------------------------------------------------------


def _ensure_playwright_shim() -> None:
    try:
        import playwright.async_api  # noqa: F401

        return
    except Exception:
        pass

    class _FakeTimeout(Exception):
        pass

    def _unavailable(*_args, **_kwargs):
        raise RuntimeError("playwright unavailable in hermetic security harness")

    async_api = types.ModuleType("playwright.async_api")
    async_api.Browser = type("Browser", (), {})
    async_api.Page = type("Page", (), {})
    async_api.TimeoutError = _FakeTimeout
    async_api.async_playwright = _unavailable

    root = types.ModuleType("playwright")
    root.async_api = async_api
    sys.modules.setdefault("playwright", root)
    sys.modules["playwright.async_api"] = async_api


# ---------------------------------------------------------------------------
# In-process app loading
# ---------------------------------------------------------------------------

_APP_IMPORTS = {
    "backend": "services.backend_core.app.main",
    "ai": "services.ai_engine.app.main",
    "browser": "services.browser_engine.app.main",
}

_APP_CACHE: dict[str, object] = {}


def _load_app(service: str):
    if service in _APP_CACHE:
        return _APP_CACHE[service]
    if service == "browser":
        _ensure_playwright_shim()
    import importlib

    module = importlib.import_module(_APP_IMPORTS[service])
    app = getattr(module, "app")
    _APP_CACHE[service] = app
    return app


# ---------------------------------------------------------------------------
# Response audit + platform guard client
# ---------------------------------------------------------------------------


class PlatformPathBlocked(AssertionError):
    """Raised when a probe touches an excluded /platform/ endpoint."""


_RECORDED_BODIES: list[str] = []
_UNSET = object()


class ServiceClient:
    """Thin request wrapper: in-process ASGI or attached HTTP."""

    def __init__(self, service: str, loop: asyncio.AbstractEventLoop):
        self.service = service
        self._loop = loop
        env_url = {
            "backend": "SEC_BACKEND_URL",
            "ai": "SEC_AI_URL",
            "browser": "SEC_BROWSER_URL",
        }[service]
        self.base_url = os.getenv(env_url, f"http://{service}.sec.test")
        token_scope = {"backend": "agent", "ai": "ai", "browser": "browser"}[service]
        self.token = os.environ.get(TOKEN_ENVS[token_scope], "")
        if ATTACHED:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=15.0)
        else:
            app = _load_app(service)
            transport = httpx.ASGITransport(app=app)
            self._client = httpx.AsyncClient(
                transport=transport, base_url=self.base_url, timeout=30.0
            )

    def request(
        self,
        method: str,
        path: str,
        *,
        token=_UNSET,
        json=None,
        params=None,
        headers=None,
    ) -> httpx.Response:
        if "/platform/" in path:
            raise PlatformPathBlocked(path)
        hdrs = dict(headers or {})
        if token is not _UNSET and token is not None:
            hdrs["Authorization"] = f"Bearer {token}"
        response = self._loop.run_until_complete(
            self._client.request(
                method, path, json=json, params=params, headers=hdrs
            )
        )
        with contextlib.suppress(Exception):
            _RECORDED_BODIES.append(response.text)
        return response

    def get(self, path, **kw):
        return self.request("GET", path, **kw)

    def post(self, path, **kw):
        return self.request("POST", path, **kw)

    def delete(self, path, **kw):
        return self.request("DELETE", path, **kw)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._loop.run_until_complete(self._client.aclose())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _managed_env():
    """Install the controlled managed-mode environment only while the security
    suite runs, then restore it.

    Historically the managed env was applied at conftest import time, which
    leaked service tokens (``BROWSER_ENGINE_API_TOKEN`` et al.) into every other
    suite collected in the same pytest invocation. Scoping it to a session
    fixture keeps the mutation contained to ``tests/security`` and guarantees a
    restore, so nothing leaks outside this suite.
    """
    if MODE != "managed":
        yield
        return
    _install_managed_env()
    try:
        yield
    finally:
        _restore_env()


@pytest.fixture(scope="session")
def sec_mode() -> str:
    return MODE


@pytest.fixture(scope="session")
def _event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    with contextlib.suppress(Exception):
        loop.close()


@pytest.fixture(scope="session")
def _clients(_managed_env, _event_loop):
    built: dict[str, ServiceClient] = {}
    yield_map: dict[str, Optional[ServiceClient]] = {}
    for service in ("backend", "ai", "browser"):
        try:
            built[service] = ServiceClient(service, _event_loop)
            yield_map[service] = built[service]
        except Exception as exc:  # import error (e.g. missing dep) -> unavailable
            yield_map[service] = None
            yield_map[f"{service}_error"] = str(exc)  # type: ignore[assignment]
    yield yield_map
    for client in built.values():
        client.close()
    # Global redaction audit (SEC-27).
    tokens = [os.environ.get(env, "") for env in TOKEN_ENVS.values()]
    tokens = [t for t in tokens if t]
    leaked = []
    for body in _RECORDED_BODIES:
        for token in tokens:
            if token and token in body:
                leaked.append(token)
    assert not leaked, "configured token value leaked into an HTTP response body"


def _client_or_skip(clients: dict, service: str) -> ServiceClient:
    client = clients.get(service)
    if client is None:
        reason = clients.get(f"{service}_error", "app unavailable")
        pytest.skip(f"{service} app unavailable in this environment: {reason}")
    return client


@pytest.fixture()
def backend(_clients) -> ServiceClient:
    return _client_or_skip(_clients, "backend")


@pytest.fixture()
def ai(_clients) -> ServiceClient:
    return _client_or_skip(_clients, "ai")


@pytest.fixture()
def browser(_clients) -> ServiceClient:
    return _client_or_skip(_clients, "browser")


class CanaryListener:
    """Loopback TCP listener that counts accepted connections."""

    def __init__(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(16)
        self.port = self._sock.getsockname()[1]
        self.count = 0
        self._stop = False
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while not self._stop:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            self.count += 1
            with contextlib.suppress(Exception):
                conn.close()

    def stop(self):
        self._stop = True
        with contextlib.suppress(Exception):
            self._sock.close()


@pytest.fixture()
def canary_listener():
    listener = CanaryListener()
    try:
        yield listener
    finally:
        listener.stop()


class AgentCLIRecorder:
    """Executable stub that records {argv, env, stdin, cwd} per invocation.

    Behaviour knobs are read from a baked control file rather than the
    environment, because the provider filters the child env down to an
    allowlist that would strip ``STUB_*`` variables.
    """

    def __init__(self, directory: Path):
        self.dir = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.dumps_dir = directory / "dumps"
        self.dumps_dir.mkdir(parents=True, exist_ok=True)
        self.control_path = directory / "control.json"
        self.command = directory / "agent_stub.py"
        self._write_stub()

    def set_control(
        self,
        *,
        sleep: float = 0.0,
        exit_code: int = 0,
        leak: str = "",
        fmt: str = "claude",
    ) -> None:
        import json

        self.control_path.write_text(
            json.dumps(
                {"sleep": sleep, "exit_code": exit_code, "leak": leak, "fmt": fmt}
            ),
            encoding="utf-8",
        )

    def _write_stub(self) -> None:
        script = f'''#!{sys.executable}
import json, os, sys, time, uuid

argv = sys.argv[1:]

# Availability preflight: respond to --version without recording.
if "--version" in argv:
    sys.stdout.write("stub 1.0.0\\n")
    sys.exit(0)

control = {{"sleep": 0.0, "exit_code": 0, "leak": "", "fmt": "claude"}}
try:
    with open({str(self.control_path)!r}, encoding="utf-8") as handle:
        control.update(json.load(handle))
except OSError:
    pass

if control.get("sleep"):
    try:
        time.sleep(float(control["sleep"]))
    except (ValueError, TypeError):
        pass

stdin_text = "" if sys.stdin.isatty() else sys.stdin.read()

record = {{
    "argv": argv,
    "env": dict(os.environ),
    "stdin": stdin_text,
    "cwd": os.getcwd(),
}}
dump_path = os.path.join({str(self.dumps_dir)!r}, uuid.uuid4().hex + ".json")
with open(dump_path, "w", encoding="utf-8") as handle:
    json.dump(record, handle)

exit_code = int(control.get("exit_code", 0))
if exit_code != 0:
    leak = control.get("leak") or ""
    if leak:
        sys.stderr.write(leak + "\\n")
    sys.exit(exit_code)

if control.get("fmt") == "codex":
    sys.stdout.write(
        json.dumps(
            {{"type": "item.completed",
              "item": {{"type": "agent_message", "text": "stub-ok"}}}}
        )
        + "\\n"
    )
else:
    sys.stdout.write(json.dumps({{"result": "stub-ok"}}) + "\\n")
sys.exit(0)
'''
        self.command.write_text(script, encoding="utf-8")
        mode = self.command.stat().st_mode
        self.command.chmod(mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    def dumps(self) -> list[dict]:
        import json

        results = []
        for path in sorted(self.dumps_dir.glob("*.json")):
            with contextlib.suppress(Exception):
                results.append(json.loads(path.read_text(encoding="utf-8")))
        return results

    @property
    def invocation_count(self) -> int:
        return len(list(self.dumps_dir.glob("*.json")))


@pytest.fixture()
def agent_cli_recorder(tmp_path) -> AgentCLIRecorder:
    return AgentCLIRecorder(tmp_path / "recorder")


def switch_llm_provider(ai_client, provider: str, model: Optional[str] = None):
    """Switch the AI Engine LLM provider, skipping if the runtime refuses it."""
    body: dict = {"llm_provider": provider}
    if model:
        body["llm_model"] = model
    response = ai_client.post("/providers/switch", token=ai_client.token, json=body)
    if response.status_code != 200:
        pytest.skip(
            f"provider switch to {provider} unavailable: "
            f"{response.status_code} {response.text[:200]}"
        )
    return response


def ai_generate(ai_client, prompt: str, system: Optional[str] = None):
    body: dict = {"prompt": prompt}
    if system:
        body["system"] = system
    return ai_client.post("/generate", token=ai_client.token, json=body)


@pytest.fixture()
def load_fixture():
    import json

    def _load(name: str) -> dict:
        path = FIXTURES_DIR / "authwalls" / f"{name}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    return _load


# ---------------------------------------------------------------------------
# Collection hooks
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(config, items):
    """Auto-skip sec_managed tests in attached mode (no controllable env)."""
    if MODE != "attached":
        return
    skip = pytest.mark.skip(reason="sec_managed needs harness-controlled env")
    for item in items:
        if "sec_managed" in item.keywords:
            item.add_marker(skip)
