# Backend Sessions Submodule Card

- purpose: session lifecycle and persisted tab-processing state.
- product/module functionality: create/list/delete sessions, current session state, URL records, tab import job records, SQLite FTS search rows, clusters, callback metadata.
- scope boundaries: owns backend session data model and keyword search metadata; URL normalization belongs in `url_input`, semantic vectors and chunked content belong in AI Engine.
- connected modules/submodules: Backend API, URL input, Browser callbacks, export, Test Harness persistence tests.
- allowed change types: persistence fixes, migration-safe schema additions, status bookkeeping, focused tests.
- special operating rules: keep file-backed SQLite behavior and in-memory fallback behavior equivalent where tests require it; keep FTS queries parameterized.
- current stubs/placeholders: tab import jobs are durable status records but do not support cancellation/resume.
- irrelevant or incomplete code to remove/rework: legacy timestamp usage emits deprecation warnings and can be modernized separately.
- docs that must stay aligned: Backend Core card and `docs/TESTING.md`.
- local validation commands/checks: `make test-backend`; focused files `tests/unit/test_session_persistence.py` and `tests/unit/test_backend_tab_workflows.py`.
