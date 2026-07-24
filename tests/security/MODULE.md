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
  - Black-box only; two flagged `sec_seam` exceptions touch importable code.
  - Managed mode boots services in-process via ASGI with a controlled env; the
    TS port substitutes `SEC_BOOT_*_CMD` real servers with the same env
    contract.
  - Distinct per-scope tokens are mandatory in managed mode so cross-acceptance
    is observable; the browser/backend token fallback chains are frozen as
    *permitted when the primary is unset*, not required.
  - A session-teardown audit asserts no configured token value appears in any
    recorded response body (SEC-27).
- invariants: SEC-1..10 (`test_url_safety_scrape.py`), SEC-11..15
  (`test_cdp_local_only.py`), SEC-16..23 (`test_token_scopes.py`), SEC-24..27
  (`test_credential_isolation.py`), SEC-28..33
  (`test_agent_subprocess_hardening.py`), SEC-34..36 (`test_prompt_envelope.py`),
  SEC-37..38 (`test_repo_hygiene.py` + `.gitleaks.toml` + `secret-scan` CI job),
  SEC-39 (`test_auth_wall_fixtures.py` + `fixtures/authwalls/*.json`).
- TS-porting rules for the seam exceptions:
  - SEC-26: run the TS MCP server over stdio, issue `tools/list`, apply the same
    name-allowlist and forbidden-verb regex to the returned tool names.
  - SEC-39: feed the same `fixtures/authwalls/*.json` to the TS auth-classifier
    seam; same assertions. Challenge fixtures carry `xfail(strict=False)`; the
    wk0 B6 fix commit deletes those markers as its acceptance criterion.
- current stubs/placeholders: SEC-9 rebinding flip-flop is only probabilistically
  covered (needs an attacker resolver); host-header/SNI pinning on multi-A-record
  fetches stays covered by Python unit tests outside this frozen suite.
- docs that must stay aligned: `docs/MODULE_INDEX.md` (ledger), `tests/README.md`,
  `docs/TESTING.md`, `.github/workflows/ci-cd.yml`.
- local validation commands/checks:
  `docker compose --profile test-unit run --rm test-unit pytest tests/security -m "security and not integration" -q`.
