# Backend Export Submodule Card

- purpose: export processed tab/session data into user-facing formats.
- product/module functionality: Markdown, JSON, HTML, and Obsidian-compatible output.
- scope boundaries: owns rendering/export formatting only; does not own scraping, clustering, or vector storage.
- connected modules/submodules: Backend API, sessions, templates, Web UI export controls.
- allowed change types: format additions, template fixes, output sanitization, tests.
- special operating rules: avoid unsafe HTML output for user-provided content.
- current stubs/placeholders: coverage is currently low compared with other backend submodules.
- irrelevant or incomplete code to remove/rework: none known.
- docs that must stay aligned: README export feature list and `docs/ARCHITECTURE.md`.
- local validation commands/checks: `make test-backend`; add focused export tests before behavior changes.
