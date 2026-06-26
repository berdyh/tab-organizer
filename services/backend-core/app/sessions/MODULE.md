# Backend Sessions Submodule Card

- purpose: session lifecycle and persisted tab-processing state.
- product/module functionality: create/list/delete sessions, current session state, URL records, clusters, callback metadata.
- scope boundaries: owns backend session data model; URL normalization belongs in `url_input`, AI content belongs in AI Engine.
- connected modules/submodules: Backend API, URL input, Browser callbacks, export, Test Harness persistence tests.
- allowed change types: persistence fixes, migration-safe schema additions, status bookkeeping, focused tests.
- special operating rules: keep file-backed SQLite behavior and in-memory fallback behavior equivalent where tests require it.
- current stubs/placeholders: none known.
- irrelevant or incomplete code to remove/rework: legacy timestamp usage emits deprecation warnings and can be modernized separately.
- docs that must stay aligned: Backend Core card and `docs/TESTING.md`.
- local validation commands/checks: `make test-backend`; focused file `tests/unit/test_session_persistence.py`.
