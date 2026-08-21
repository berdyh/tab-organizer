# Security-Invariant Suite (FROZEN)

This suite is **FROZEN** at `SECSUITE_VERSION = "1.8.0"` (see `__init__.py`). It
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
SEC-42 agent env-allowlist drift check, SEC-48 fixture-completeness guard).

## New in 1.7.0 (SEC-45 metadata whitelist widened to six keys)

SEC-45's subset assertion and the service's own
`routes.py::URL_LIST_METADATA_FIELDS` had disagreed ever since
`credential_scope_drop` was added to the service and not to the probe. The
service can emit six keys; this asserted a subset of five. **The first capture
whose fetch was redirected off the origin its credentials belong to would have
failed SEC-45 against the PYTHON stack**, not merely against a port. It never
fired because no probe capture triggers a scope drop — a could-not-fail gap
wearing the costume of a passing test, which is the class this repo has now met
six times.

Found by the TypeScript port, which had to pick one of the two lists to mirror
and could not, because they were not the same list.

Widened the probe rather than narrowing the service. `credential_scope_drop` is
a security SIGNAL, not a leak: it records that a redirect carried the fetch
off-origin so the stored credentials were dropped instead of being sent onward.
Removing it from the listing is what would let a logged-out capture read as an
authenticated one — the exact confusion `auth_used=False` exists to prevent, and
which feeds the decision-37 "authenticated capture ⇒ local embeddings" gate.

Payload verified by reading the producer rather than trusting the comment above
it (`engine.py::_credential_drop_fields`, `CredentialHopTrace`): the value is
`{reason, auth_type, credential_origin_host, dropped_at_host, hops, 
credential_scope_domain, credential_scope_subdomains}` — hostnames via
`urlparse(...).hostname`, an integer hop COUNT, booleans and an auth-type label.
No credentials, no paths, no queries. The comment's "hosts only, never full
URLs" is accurate.

**No assertion was weakened.** It remains a subset assertion against a closed,
reviewed list; a seventh key still fails, and the probe still ingests a capture
carrying an unreviewed `description` key and requires it dropped, so the
deny-by-default property is still executable rather than assumed.

## New in 1.8.0 (T6 — boot mode implemented)

`SEC_BOOT_*_CMD` exists. See "`SEC_BOOT_*_CMD` — IMPLEMENTED at 1.8.0" above
for the mechanism, the per-channel vacuity analysis, and the mutation evidence.
What changed inside the suite:

- `boot.py` (new) — the launcher: free-port allocation, health-gated startup,
  restart on environment drift, append-only child log capture, the
  `sitecustomize` playwright shim, and `BootFailure`.
- `conftest.py` — mode is now resolved **per service** (`SERVICE_MODES`), the
  `sec_managed` skip decision moved from collection time to client-acquisition
  time (SEC-21 picks its service from a fixture at runtime, which a
  collection-time rule cannot see), a boot client that cannot be built raises
  instead of skipping, and boot runs print `executed N/21` plus the nodeid of
  every `sec_managed` probe that did not run.
- `test_credential_isolation.py` — SEC-27 reads `ai.capture_logs(...)` instead
  of `caplog`, and asserts the capture is non-empty.

**No assertion was weakened.** One was added (SEC-27's non-vacuity check, which
also strengthens managed mode) and one observation channel was made
mode-symmetric. Managed mode before and after: **205 passed / 3 skipped / 5
deselected**. Boot mode with all three Python services launched as subprocesses:
**205 passed / 3 skipped / 5 deselected**. Attached mode: unchanged, 21
`sec_managed` skipped.

## New in 1.6.0 (plan decision 44 — the boot-mode hedge, executed)

`SEC_BOOT_*_CMD` is still not implemented and its deferral to ~wk8 still
stands (decision 42). Decision 44 is the hedge against that deferral, and it
is now done: **every `sec_managed` probe's inputs and expected refusals live
in `fixtures/*.json`, not inline in Python.**

| Fixture | Probes |
| --- | --- |
| `agent_subprocess_hardening.json` | SEC-28..33, SEC-46, SEC-47 |
| `credential_isolation.json` | SEC-25, SEC-27 |
| `prompt_envelope.json` | SEC-34, SEC-35 (+ the shared envelope contract SEC-40/41/36 also read) |
| `token_scope_failclosed.json` | SEC-21, SEC-22 |
| `cors_policy.json` | SEC-43 (all three checks; the third is the `sec_managed` one) |
| `url_safety_escape_hatch.json` | SEC-10 |
| `sec_managed_index.json` | the registry SEC-48 reconciles |
| `agent_env_allowlist.json` | unchanged — SEC-25/42, the pattern this generalises |

Grouping is one file per probe FAMILY (the unit that shares staging, shared
constants and a rationale), not one per probe and not one for the whole suite:
a boot-mode runner loads the file for the subsystem it is exercising, and a
reviewer sees a family's contract whole.

- `contracts.py` holds the loader and the assertion runners. They are
  deliberately **strict**: an unrecognised key inside a contract block raises
  instead of being ignored, an empty contract block raises, and
  `assert_expect_keys_consumed` fails if a probe stops reading one of its
  fixture's `expect` keys. A generic runner that silently skips what it does
  not understand is how a frozen contract gets weakened by a typo. Only
  underscore-prefixed keys (`_comment`, `_note`) are ignorable.
- **SEC-48** (`test_fixture_completeness.py`, `sec_seam`) is the guard on the
  hedge. It enumerates the `sec_managed` probes from the suite's own modules'
  markers and reconciles them three ways against `sec_managed_index.json`:
  every probe registered, every entry resolving to a real non-empty contract
  block, and no entry naming a probe that no longer exists. Without it, probe
  N+1 gets added inline next year and boot mode's data set silently stops
  covering the suite.
- No assertion was weakened. Every probe asserts exactly what it asserted at
  1.5.0; this changed **where** the contract lives, not what it says. Verified
  by mutation in both directions for each family (fixture mutated → probe
  fails; service code mutated → probe fails); the outputs are in the ledger
  row.

**Corrected counts.** The 1.5.0 edit added SEC-46/47 without updating the
arithmetic in this file, which still said "175 of 194" and "the 19
`sec_managed` probes" — both were the pre-1.5.0 numbers. At 1.5.0 the real
figures were 196 probes / 21 `sec_managed`; at 1.6.0 they are **213 probes, 21
`sec_managed`, 192 that run in attached mode**. `docs/ARCHITECTURE_PLAN.md`'s
Addendum 2026-08-05 carried the same stale pair; it was corrected in place on
2026-08-07 and now reads 21 `sec_managed` of 213, 192 attached-runnable.

## New in 1.5.0 (gemini_cli adapter)

Two `sec_managed` probes for a blind spot the suite had by construction: every
agent-subprocess probe names its provider. SEC-28/31/32/33 drive `claude_code`
and SEC-29/30 drive `codex_cli`, so a THIRD subscription CLI adapter inherited
only the base-class guarantees that happen to be shared (env allowlist via
SEC-25/42) and none of the per-adapter ones. Adding `gemini_cli` without these
would have shipped an unprobed subprocess.

- **SEC-46** (`test_agent_subprocess_hardening.py`) — the gemini subprocess is
  headless (`-p`), read-only (`--approval-mode plan`) and stdin-free, and the
  guardrail preamble precedes the user block. Each is a one-flag regression:
  without `-p` the CLI starts an interactive session and never returns; `yolo`
  would hand a model driven by page text the write and shell tools; and the
  CLI's login prompt ignores EOF, so nothing may resemble an answer to it.
  This adapter puts the envelope in **argv**, not stdin, so SEC-31's ordering
  invariant is re-checked on the `-p` argument.
- **SEC-47** (same file) — scraped content never reaches gemini by default.
  Same reasoning as SEC-30 for codex: gemini has no tool-free mode, and its
  most restrictive documented approval mode still allows `read_file`,
  `google_web_search` and `web_fetch` (read from the CLI's own bundled
  `policies/read-only.toml`), each an exfiltration channel for injected
  instructions.

Both stage a HOME containing `~/.gemini/oauth_creds.json` and **assert** the
provider switch returned 200 rather than skipping on it, because the adapter
refuses to report available without credentials — an unstaged probe would have
skipped silently and asserted nothing. `AgentCLIRecorder.set_control(fmt=...)`
gained a `"gemini"` format emitting the CLI's real `{"response": ...}`
envelope.

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
  controlled remotely). Attached mode is language-agnostic — **192 of 213**
  probes work there; the **21** `sec_managed` ones auto-skip.
- **Boot mode** (`SEC_BOOT_*_CMD`, new at 1.8.0): the harness *launches* the
  service itself and restarts it whenever a probe changes the environment, so
  all 213 probes run against any language. See below.

Mode is decided **per service** — boot beats attached beats managed — so a
booted TypeScript backend can be probed in the same run as a managed Python
ai-engine.

### `SEC_BOOT_*_CMD` — IMPLEMENTED at 1.8.0 (T6, plan decision 42)

The harness launches each service from a command it owns and gets per-test
environment control back against any language:

```bash
SEC_BOOT_BACKEND_CMD="sh -c 'GATEWAY_PORT={port} exec node server/gateway/src/main.ts'" \
SEC_BOOT_AI_CMD="python -m uvicorn services.ai_engine.app.main:app --host 127.0.0.1 --port {port}" \
  pytest tests/security -m "security and not integration" -q
```

`{port}` is substituted into the command with a free port the harness picked;
the same value is also exported to the child as `SEC_BOOT_PORT` for services
that read their port from the environment. `{host}` is available too.

**Why it exists.** Managed mode gets its environment control by importing the
FastAPI app, which is Python-only. Without boot mode the 21 `sec_managed`
probes auto-skip against any other implementation, so a TypeScript port could
go green having never exercised agent subprocess hardening (SEC-28..33,
46..47), credential isolation, prompt-envelope containment, token fail-closed
defaults, the CORS non-vacuity check, or URL-safety refusals under controlled
config. Those are premise 4 in executable form.

**How environment control is restored.** The child is restarted whenever the
staged environment differs from the one it was launched with. Three rules make
that trustworthy rather than merely convenient:

1. **Restart granularity is per REQUEST, not per test.** SEC-22 stages
   `hatch_absent`, asserts 401, mutates the environment, then asserts the hatch
   really opens — inside one test function. A per-test restart would serve both
   halves from one process and silently collapse the probe into one assertion.
2. **A failed launch RAISES (`BootFailure`), it never skips.** The managed-mode
   "app unavailable → skip" path removes probes silently; a child that dies on a
   bad command, a missing dependency or a port collision must fail the run, not
   delete the coverage boot mode was added to provide. The child's captured
   output is in the message.
3. **A restart resets service state.** Inherent to a process boundary. Probes
   that stage environment → switch provider → act are safe because nothing
   mutates between the switch and the act; a probe needing server state to
   survive an environment change cannot be expressed in boot mode and must say
   so rather than quietly observe a fresh process.

**Observation channels had to cross the boundary too.** A probe that reads a
channel which is dead in boot mode passes having seen nothing, which is worse
than not running. The four channels and where each stands:

| Channel | Probes | Cross-process status |
| --- | --- | --- |
| `agent_cli_recorder.dumps()` | SEC-25, 28..33, 34/35, 46/47 | Already guarded — `assert dumps` fails when the recorder saw nothing. The stub is exec'd by the child and writes to a shared path. |
| `canary_listener.count` | SEC-10 (`sec_managed`) | Already guarded — the `hatch_set` step requires `canary_connections_min`, so an unreachable canary fails instead of reading as "correctly refused". |
| service diagnostics | SEC-27 | **Changed at 1.8.0.** `caplog` sees nothing from a subprocess, so SEC-27 now reads `ai.capture_logs(...)`: the root logger in managed mode, the child's stdout/stderr in boot mode. The capture is additionally asserted **non-empty**, because "the secret is absent from an empty log" asserts nothing. |
| `switch_llm_provider` | agent-subprocess families | Skips when a provider is unavailable. SEC-46/47 already assert the switch returned 200; for the rest, boot runs print `executed N/21 sec_managed probes` and name every probe that did not run, so a silent removal is visible. |

**Restart behaviour is observed, not inferred.** `boot.py` writes a
`=== boot #N ===` marker to the child log on every launch. Counted during a run:
SEC-22 restarts **twice** (once per staged environment, which is what keeps its
two halves from collapsing into one), and SEC-28 restarts **once** across its
two requests (so a provider switch survives to the call that depends on it, and
the environment comparison is not firing indiscriminately). `ENV_NOISE_KEYS` is
load-bearing in one direction only: a key wrongly added is a silent false pass,
a key wrongly omitted only costs a restart.

**Playwright across the boundary.** Managed mode injects a fake
`playwright.async_api` into `sys.modules` before importing browser-engine.
`sys.modules` does not survive exec, so boot mode reinstalls the same shim in
the child through a generated `sitecustomize.py` on `PYTHONPATH`. Without it a
booted browser-engine fails to import for a reason unrelated to any invariant.

**Verified by mutation, in both languages.** Boot-mode-green is satisfiable
vacuously, so the six code-side mutations from the 1.6.0 ledger were re-applied
to a throwaway copy of `services/` and each was observed failing **in boot
mode**: claude `--tools ""`→`"all"` (SEC-28), `ENV_ALLOWLIST` + `ANTHROPIC_API_KEY`
(SEC-25), the ai-engine unset-token 401 → `return` (SEC-21[ai_embed],
SEC-22[hatch_absent]), `http://localhost:8089` dropped from the CORS default
(SEC-43 ×3), `private_scrape_urls_allowed()`→`True` (SEC-10[hatch_absent]), plus
a seventh added for the new channel — `_redact` made a no-op, caught by SEC-27
reading the planted `sk-ant-…` sentinel out of the launched child's stderr. The
same mutation was confirmed still failing in managed mode, so the `caplog`
→ `capture_logs` swap did not weaken it. Finally the **TypeScript** gateway was
booted by the harness and `http://localhost:8089` removed from its
`DEFAULT_ALLOWED_ORIGINS`: SEC-43[backend] failed. That is the whole point of
T6 — the frozen Python suite catching a TypeScript regression.

**Hedge (decision 44) still stands.** Each `sec_managed` probe's inputs and
expected refusals live in `fixtures/*.json`, so boot mode is a runner over data
and softening a contract is a reviewable fixture diff. Boot mode owed the launch
mechanism; that debt is now paid.

## Seam exceptions

Every probe here is meant to be black-box (HTTP status codes, response bodies,
recorded subprocess argv/env/stdin/cwd). Six probes touch importable Python
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
- **SEC-48** (`test_fixture_completeness.py`, `sec_seam`) — imports the suite's
  own test modules to read their pytest markers, which is the only way to ask
  "which `sec_managed` probes exist" without hardcoding the answer it is
  checking. TS-porting rule: enumerate the TS suite's equivalent
  managed-only-tagged cases and apply the same three-way reconciliation against
  the same `fixtures/sec_managed_index.json`.

**SEC-25 is no longer a seam exception.** It used to import
`AgentCLILLMProvider.ENV_ALLOWLIST` directly; it now asserts the observed
subprocess env is a subset of `fixtures/agent_env_allowlist.json` (a frozen
data contract, not a Python import) — genuinely black-box. SEC-42 above is the
new, explicitly-marked exception that keeps the Python constant honest against
that same fixture.
