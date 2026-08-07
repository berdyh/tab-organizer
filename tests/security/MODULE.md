# Security-Invariant Suite Module Card

- purpose: freeze the backend's security invariants as a black-box probe set the
  TypeScript port must also pass.
- product/module functionality: outbound URL safety, CDP local-only, token
  scopes + fail-closed auth, credential isolation, agent subprocess hardening,
  prompt-injection envelope, CI secret scanning, WI0-B6 auth-classifier fixtures.
- scope boundaries: tests + CI only; never modifies service source. Excludes all
  `/api/v1/platform/*` endpoints (plan decision 41), enforced by a conftest
  guard that raises on any `/platform/` path.
- connected modules/submodules: Backend Core, AI Engine, Browser Engine, Ops
  Tooling (`scripts/mcp/tabs.py`), CI workflow.
- allowed change types: only with a `docs/MODULE_INDEX.md` ledger row and a
  `SECSUITE_VERSION` bump (`__init__.py`). The suite is FROZEN.
- special operating rules:
  - Black-box only; six flagged `sec_seam` exceptions touch importable code
    (SEC-26, SEC-39, SEC-40, SEC-41, SEC-42, SEC-48).
  - Managed mode boots services in-process via ASGI with a controlled env. It
    imports the Python module to do that, so it works only against this stack.
    `SEC_BOOT_*_CMD` (harness-launched servers, any language) is **PLANNED, NOT
    IMPLEMENTED** — the name appears in prose only. Until it exists a TS port
    runs 192/213 probes in attached mode and auto-skips the 21 `sec_managed`
    ones, which are exactly the process-boundary guarantees (agent subprocess
    hardening, credential isolation, prompt envelope, token fail-closed
    defaults, the CORS non-vacuity check, URL safety under controlled config).
    Trigger revised 2026-08-05: land it before the first TS commit touching
    credentials, tokens, or agent subprocesses (~wk8), not before facade work.
    CORS and token scopes apply from wk1 but need no boot mode — they run in
    attached mode. (The counts here and in `README.md` were stale at 1.5.0 —
    "173/192" and "19" are pre-SEC-46/47 figures and are still uncorrected in
    `docs/ARCHITECTURE_PLAN.md`'s Addendum 2026-08-05.)
  - **Every `sec_managed` probe's inputs and expected refusals are DATA**
    (`fixtures/*.json`), not inline Python — plan decision 44, the hedge
    against deferring boot mode. A probe body stages and observes; what must
    hold is in the fixture, so boot mode is a runner over data and softening a
    contract is a reviewable fixture diff. `contracts.py`'s runners raise on
    unknown or empty contract blocks and on `expect` keys a probe stops
    reading, so a mistyped assertion cannot become a silent no-op. SEC-48
    (`sec_seam`) reconciles the marker-derived probe list against
    `fixtures/sec_managed_index.json` in both directions, so a probe added
    inline later fails the build instead of quietly falling outside the hedge.
    Adding a `sec_managed` probe means: fixture entry, registry entry, ledger
    row, `SECSUITE_VERSION` bump.
  - Every agent-subprocess probe names its provider, so the suite does NOT
    cover a newly added CLI adapter by construction. SEC-28/31/32/33 are
    claude-only and SEC-29/30 codex-only; SEC-46/47 were added for
    `gemini_cli` on the same basis, and a fifth adapter needs its own pair.
    Both gemini probes stage `~/.gemini/oauth_creds.json` into a temp HOME and
    assert the provider switch returned 200, because that adapter reports
    unavailable without credentials and the probe would otherwise skip
    silently.
  - Distinct per-scope tokens are mandatory in managed mode so cross-acceptance
    is observable. As of SECSUITE 1.3.0 this also mirrors deployment:
    browser-engine's cross-scope ACCEPT fallback was removed (it accepts only
    `BROWSER_ENGINE_API_TOKEN`) and `scripts/cli.py` mints four independent
    tokens, so the `browser_auth_pending` door is load-bearing in the real
    stock configuration, not only under this harness's synthetic env.
    Outbound token *selection* fallbacks (which token a service SENDS
    downstream) are a separate concern and are not frozen here.
  - A session-teardown audit asserts no configured token value appears in any
    recorded response body (SEC-27).
  - The prompt-envelope invariant's GATING probes are SEC-40/41 (hermetic,
    `sec_seam`, drive the prompt-assembly seam directly with no live embedder
    needed); SEC-34/35 are live-stack certification probes only (`sec_managed`
    + an embedding-backend skip stack them so they can never fail CI alone).
  - SEC-44's route table is enumerated from the service's own `/openapi.json`,
    never a hand-maintained list, so an added route is unclassified and fails
    until someone protects it or puts it on `PUBLIC_ROUTES` with a reason. The
    allowlist is also checked for staleness and for still being genuinely
    anonymous-reachable, so it cannot drift into a rubber stamp.
  - SEC-25's contract is data, not Python: it asserts the observed agent
    subprocess env is a subset of `fixtures/agent_env_allowlist.json`. SEC-42
    (`sec_seam`) separately pins `AgentCLILLMProvider.ENV_ALLOWLIST` to equal
    that same fixture, so a Python-side drift from the frozen contract fails
    loudly instead of silently changing what SEC-25 permits.
- invariants: SEC-1..10 (`test_url_safety_scrape.py`), SEC-11..15
  (`test_cdp_local_only.py`), SEC-16..23 (`test_token_scopes.py`), SEC-43
  (`test_cors_policy.py`), SEC-44..45 (`test_route_exposure.py`), SEC-24..27,
  SEC-42 (`test_credential_isolation.py` + `fixtures/agent_env_allowlist.json`),
  SEC-28..33, SEC-46..47 (`test_agent_subprocess_hardening.py`), SEC-34..36, SEC-40..41
  (`test_prompt_envelope.py`), SEC-37..38 (`test_repo_hygiene.py` +
  `.gitleaks.toml` + `secret-scan` CI job), SEC-39 (`test_auth_wall_fixtures.py`
  + `fixtures/authwalls/*.json`), SEC-48 (`test_fixture_completeness.py` +
  `fixtures/sec_managed_index.json`).
- TS-porting rules for the seam exceptions:
  - SEC-26: run the TS MCP server over stdio, issue `tools/list`, apply the same
    name-allowlist and forbidden-verb regex to the returned tool names.
  - SEC-39: feed the same `fixtures/authwalls/*.json` to the TS auth-classifier
    seam; same assertions. Challenge fixtures carry `xfail(strict=False)`; the
    wk0 B6 fix commit deletes those markers as its acceptance criterion.
  - SEC-40/41: call the TS port's equivalent chat/cluster-label prompt-builder
    function(s) directly with the same poisoned fixtures (reused from
    SEC-34/35) and apply the same envelope assertions to the output string.
  - SEC-42: once the TS Agent SDK adapter lands, pin its equivalent allowlist
    constant/config against `fixtures/agent_env_allowlist.json`.
  - SEC-48: enumerate the TS suite's equivalent managed-only-tagged cases and
    reconcile them against the same `fixtures/sec_managed_index.json`, in both
    directions.
- current stubs/placeholders: SEC-9 rebinding flip-flop is only probabilistically
  covered (needs an attacker resolver); host-header/SNI pinning on multi-A-record
  fetches stays covered by Python unit tests outside this frozen suite.
- docs that must stay aligned: `docs/MODULE_INDEX.md` (ledger), `tests/README.md`,
  `docs/TESTING.md`, `.github/workflows/ci-cd.yml`.
- local validation commands/checks:
  `docker compose --profile test-unit run --rm test-unit pytest tests/security -m "security and not integration" -q`.
