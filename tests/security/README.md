# Security-Invariant Suite (FROZEN)

This suite is **FROZEN** at `SECSUITE_VERSION = "1.2.0"` (see `__init__.py`). It
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
