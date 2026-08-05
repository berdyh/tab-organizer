# Test Harness Module Card

- purpose: prove module behavior and cross-service contracts.
- product/module functionality: unit, integration, E2E, load scenarios, container test image, shared fixtures.
- scope boundaries: owns validation harness only; does not encode new product direction unless tied to a module contract.
- connected modules/submodules: all service modules, Ops Tooling, CI.
- allowed change types: focused test additions, validation target wiring, stale test removal, fixture cleanup.
- special operating rules: prefer module-local tests before full CI; keep Docker profile commands aligned with `scripts/cli.py` and CI.
- current stubs/placeholders: load/performance suite is manual. The integration/E2E containers run against whatever images are on the machine; `services/ai-engine/requirements.txt` pins `lancedb==0.6.8`, a release that no longer exists on PyPI, so the ai-engine image cannot currently be rebuilt from scratch and `tests/integration/test_api.py::TestAIEngineAPI::test_health_check` fails against an older image (it asserts the `runtime` block this branch added).
- Run integration/E2E through `./scripts/cli.py test`, not by hand. `cmd_test` is the one lifecycle command that does NOT call `load_env_file()`, so it resolves the four service tokens from `data/service-tokens.json` while `cmd_start` resolves them from `.env` first — a stack left running by `start` and one recreated by `test` can end up on different tokens, and a hand-rolled `docker compose run test-integration` will then 401 on every authenticated door. `cmd_test`'s own `up -d` recreates the services with its env, so the runner stays self-consistent.
- `tests/integration/test_api.py::TestIngestV1` and `tests/e2e/test_workflow.py::TestCaptureIngestWorkflow` are the in-situ coverage for `POST /api/v1/ingest/v1` (gap G3); see `services/backend-core/app/api/MODULE.md` for what they do and do not reach. The E2E suite needs `BACKEND_AGENT_API_TOKEN` (backend `/api/v1/search` is agent-scoped) and the integration suite needs all four service tokens — it asserts the foreign ones are REJECTED, so it fails loudly rather than skipping when a token is unset, since an empty header is indistinguishable from a rejected one.
- irrelevant or incomplete code to remove/rework: obsolete fully skipped gateway E2E coverage was removed; keep E2E coverage on the current direct-service workflow.
- docs that must stay aligned: `docs/TESTING.md`, `tests/README.md`, Makefile targets, CI workflow.
- local validation commands/checks: `make test-backend`, `make test-ai`, `make test-browser`, `make test-web`, `make test-ops`, `./scripts/cli.py test --type all`.
