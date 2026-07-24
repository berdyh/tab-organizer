# Test Harness Module Card

- purpose: prove module behavior and cross-service contracts.
- product/module functionality: unit, integration, E2E, load scenarios, container test image, shared fixtures.
- scope boundaries: owns validation harness only; does not encode new product direction unless tied to a module contract.
- connected modules/submodules: all service modules, Ops Tooling, CI.
- allowed change types: focused test additions, validation target wiring, stale test removal, fixture cleanup.
- special operating rules: prefer module-local tests before full CI; keep Docker profile commands aligned with `scripts/cli.py` and CI.
- current stubs/placeholders: load/performance suite is manual.
- irrelevant or incomplete code to remove/rework: obsolete fully skipped gateway E2E coverage was removed; keep E2E coverage on the current direct-service workflow.
- docs that must stay aligned: `docs/TESTING.md`, `tests/README.md`, Makefile targets, CI workflow.
- local validation commands/checks: `make test-backend`, `make test-ai`, `make test-browser`, `make test-web`, `make test-ops`, `./scripts/cli.py test --type all`.
