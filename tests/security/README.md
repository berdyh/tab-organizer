# Security-Invariant Suite (FROZEN)

This suite is **FROZEN** at `SECSUITE_VERSION = "1.4.0"` (see `__init__.py`). It
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
`MODULE.md` for the full invariant list and the TS-porting rules for the
`sec_seam` exceptions (SEC-26 MCP tool surface, SEC-39 auth classifier, SEC-40
RAG chat prompt-assembly seam, SEC-41 cluster-label prompt-assembly seam,
SEC-42 agent env-allowlist drift check).

## New in 1.4.0 (wk0 batch B)

Three black-box probes closing self-blind spots the suite had no coverage for:

- **SEC-43** (`test_cors_policy.py`) — no probe issued an `Origin` header
  anywhere, so `allow_origins=["*"]` + `allow_credentials=True` on all three
  services was invisible. Asserts a foreign origin is neither echoed nor
  wildcarded and credentials are never allowed, plus a `sec_managed`
  non-vacuity check that the configured UI origin *is* granted.
- **SEC-44** (`test_route_exposure.py`) — no probe would have caught a
  newly-added unauthenticated route (two credential proxies shipped that way).
  Enumerates the route table from the service's own `/openapi.json` and
  requires every route to answer 401 anonymously unless it is on the reviewed,
  commented `PUBLIC_ROUTES` allowlist. New routes fail until classified.
- **SEC-45** (`test_route_exposure.py`) — `GET /api/v1/urls/{session_id}`
  returned stored metadata verbatim, including the full captured page body.
  Ingests a capture with a known body and asserts the listing carries neither
  that text, a `content` key, nor any unreviewed metadata key.

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
  an ASGI transport with a fully controlled, per-run environment. It gets that
  control by importing the Python module directly (`conftest.py::_load_app`),
  so managed mode works **only** against this Python stack.
- **Attached mode**: set any `SEC_*_URL`; the harness issues real HTTP to the
  running services and `sec_managed` probes auto-skip (their env can't be
  controlled remotely). Attached mode is language-agnostic — it is how the
  TypeScript port runs this suite today, and 173 of 192 probes work there.

### `SEC_BOOT_*_CMD` — PLANNED, NOT IMPLEMENTED

A third mode in which the harness *launches* each service from a command it
controls (`SEC_BOOT_BACKEND_CMD="node dist/server.js --port {port}"`), so it
regains per-test env control against any language. **It does not exist.** The
name appears only in prose here, in `MODULE.md`, and in `conftest.py`'s
docstring; there is no implementation behind it.

What that costs today: the 19 `sec_managed` probes auto-skip in attached mode,
so a TypeScript port can go green having never exercised agent subprocess
hardening (SEC-28..33), credential isolation, prompt-envelope containment, or
URL-safety refusals under controlled config. Those are premise 4 in executable
form.

**When it must land** (revised 2026-08-05, supersedes "before TS facade work
begins"): before the first TypeScript commit that touches credentials, tokens,
or agent subprocesses — in practice around the wk8 rehearsal, not wk1. The
components those 19 probes validate (browser-engine capture, ai-engine
providers) port at or after the wk10 cutover, so building boot mode earlier
would mean validating a TS implementation that does not exist yet.

The two invariants that DO apply from the first facade commit — CORS policy and
token scopes — need no boot mode. They are plain HTTP assertions: point
`SEC_BACKEND_URL` at the TS facade and they run in attached mode.

**Hedge against deferring** (do this while the Python behaviour is verified and
nobody is under cutover pressure): extract each `sec_managed` probe's inputs and
expected refusals into language-neutral JSON fixtures, the pattern SEC-25/42
already use for the agent env allowlist. Boot mode then becomes a runner over
data, and the contract cannot be quietly softened to fit whatever got built —
changing it means editing a fixture in a reviewable diff.
- `@pytest.mark.security` — all tests. `integration` — needs internet/live LLM
  (module-gated on `SEC_ALLOW_NETWORK=1`). `sec_managed` — needs
  harness-controlled env. `sec_seam` — Python-seam exception with a TS-porting
  rule.

## Seam exceptions

Every probe here is meant to be black-box (HTTP status codes, response bodies,
recorded subprocess argv/env/stdin/cwd). Five probes touch importable Python
instead, for the reasons below — listed here so a reader doesn't mistake an
accepted, reasoned exception for an oversight:

- **SEC-26** (`test_credential_isolation.py::test_mcp_tool_surface_has_no_credential_verbs`,
  `sec_seam`) — imports `scripts.mcp.tabs.TOOL_FUNCTIONS` directly. TS-porting
  rule: run the TS MCP server over stdio, issue `tools/list`, apply the same
  name-allowlist and forbidden-verb regex to the returned tool names.
- **SEC-39** (`test_auth_wall_fixtures.py`, `sec_seam`) — feeds Python fixture
  data through a Python auth-classifier function. TS-porting rule: feed the
  same `fixtures/authwalls/*.json` to the TS auth-classifier seam with the
  same assertions.
- **SEC-40** (`test_prompt_envelope.py::test_sec40_rag_chat_prompt_assembly_wraps_retrieved_text`,
  `sec_seam`) — imports `RAGChatbot`, `ClaudeCodeLLMProvider`, and `LLMConfig`
  directly to drive the chat prompt-assembly seam with a fixed poisoned
  "retrieval" result (no live embedding backend involved) through a real CLI
  provider pointed at the hermetic `agent_cli_recorder` stub. This is the
  *gating* counterpart to SEC-34, which is `sec_managed` **and** additionally
  skips without a live embedder, so it can never fail CI on its own.
  TS-porting rule: call the TS port's equivalent chat-prompt-builder
  function(s) directly with the same poisoned fixture and apply the same
  envelope assertions to the output string.
- **SEC-41** (`test_prompt_envelope.py::test_sec41_cluster_label_prompt_assembly_carries_envelope`,
  `sec_seam`) — same pattern as SEC-40, for `TabClusterer.generate_cluster_label`.
  Gating counterpart to SEC-35. TS-porting rule: call the TS port's equivalent
  label-prompt-builder function(s) directly with the same poisoned tab titles
  and apply the same envelope assertions.
- **SEC-42** (`test_credential_isolation.py::test_sec42_env_allowlist_matches_frozen_fixture`,
  `sec_seam`) — imports `AgentCLILLMProvider.ENV_ALLOWLIST` directly to pin it
  against the frozen `fixtures/agent_env_allowlist.json` contract, as a drift
  check. TS-porting rule: once the TS Agent SDK adapter lands, pin its
  equivalent allowlist constant/config against the same fixture file.

**SEC-25 is no longer a seam exception.** It used to import
`AgentCLILLMProvider.ENV_ALLOWLIST` directly; it now asserts the observed
subprocess env is a subset of `fixtures/agent_env_allowlist.json` (a frozen
data contract, not a Python import) — genuinely black-box. SEC-42 above is the
new, explicitly-marked exception that keeps the Python constant honest against
that same fixture.
