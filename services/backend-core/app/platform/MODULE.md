# Backend Platform Submodule Card

- purpose: local platform account, company discovery, API token, dashboard, and maintainer support domain.
- product/module functionality: user signup/login/session, B2B token issuance, company search, API request audit, dashboard counters, issue queue.
- scope boundaries: owns platform tables and permissions; does not own Streamlit rendering or non-platform tab workflows.
- connected modules/submodules: Backend API routes, Web UI platform page, Test Harness platform tests.
- allowed change types: persistence split, schema-safe additions, role/permission fixes, API token hardening, focused tests.
- special operating rules: no raw API token disclosure except creation response; keep token hashes parameterized; preserve maintainer bootstrap code enforcement and scope allowlist.
- current stubs/placeholders: seeded companies are demo data for local discovery.
- irrelevant or incomplete code to remove/rework: `store.py` combines schema, auth, tokens, companies, dashboard, and issues; split only in behavior-preserving commits.
- docs that must stay aligned: Backend Core card, Web UI card, `docs/ARCHITECTURE.md`, `docs/AI_CONFIG.md` when platform provider settings change.
- local validation commands/checks: `make test-backend`; focused file `tests/unit/test_platform_backend.py`.
