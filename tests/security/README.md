# Security-Invariant Suite (FROZEN)

This suite is **FROZEN** at `SECSUITE_VERSION = "1.0.0"` (see `__init__.py`). It
is the black-box security contract for the Tab Organizer backend. The
TypeScript reimplementation **must pass the same probes** by pointing the
harness env vars at its own servers/boot commands — the test IDs and fixture
files are the contract, and every assertion references observable behavior only
(status codes, response bodies, recorded argv/env/stdin/cwd, canary-connection
counts).

**Excluded:** every `/api/v1/platform/*` (B2B platform) endpoint, per plan
decision 41. The exclusion is enforced — `conftest.py` raises
`PlatformPathBlocked` if any probe requests a path containing `/platform/`.

## Freeze rule

After merge, any semantic edit to `tests/security/` requires a decision-log row
in `docs/MODULE_INDEX.md`'s ledger and a bump of `SECSUITE_VERSION`. See
`MODULE.md` for the full invariant list and the TS-porting rules for the two
`sec_seam` exceptions (SEC-26 MCP tool surface, SEC-39 auth classifier).

## Running

Hermetic subset (CI, inside the test-unit profile):

```bash
docker compose --profile test-unit run --rm test-unit \
  pytest tests/security -m "security and not integration" -q
```

Internet/live-LLM subset (integration lane):

```bash
SEC_ALLOW_NETWORK=1 pytest tests/security -m "security and integration" -q
```

Attached mode (point at a running stack instead of in-process apps):

```bash
SEC_BACKEND_URL=... SEC_AI_URL=... SEC_BROWSER_URL=... \
AI_ENGINE_API_TOKEN=... BACKEND_CALLBACK_TOKEN=... \
BACKEND_AGENT_API_TOKEN=... BROWSER_ENGINE_API_TOKEN=... \
pytest tests/security -q
```

## Modes and markers

- **Managed mode** (default): the harness runs each FastAPI app in-process via
  an ASGI transport with a fully controlled, per-run environment. This is the
  Python harness's boot adapter; the TS port substitutes real servers via the
  `SEC_BOOT_BACKEND_CMD` / `SEC_BOOT_AI_CMD` / `SEC_BOOT_BROWSER_CMD` command
  templates and the same token env contract.
- **Attached mode**: set any `SEC_*_URL`; the harness issues real HTTP to the
  running services and `sec_managed` probes auto-skip (their env can't be
  controlled remotely).
- `@pytest.mark.security` — all tests. `integration` — needs internet/live LLM
  (module-gated on `SEC_ALLOW_NETWORK=1`). `sec_managed` — needs
  harness-controlled env. `sec_seam` — Python-seam exception with a TS-porting
  rule.
