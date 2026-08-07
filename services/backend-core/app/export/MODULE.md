# Backend Export Submodule Card

- purpose: export processed tab/session data into user-facing formats.
- product/module functionality: Markdown, JSON, HTML, and Obsidian-compatible output.
- scope boundaries: owns rendering/export formatting only; does not own scraping, clustering, or vector storage.
- connected modules/submodules: Backend API, sessions, templates, Web UI export controls.
- allowed change types: format additions, template fixes, output sanitization, tests.
- special operating rules: avoid unsafe HTML output for user-provided content.
- current stubs/placeholders: coverage was ZERO until 2026-08-07 — there were no export tests at all, which is how the template bug below survived. `tests/unit/test_export_html_template.py` is the first.
- **Template resolution is a two-part invariant, and each part masks the other.** `services/backend-core/Dockerfile` must `COPY templates/`, AND `Exporter._resolve_templates_dir` must find them at the container path (`/app/templates`) rather than by counting parents from `__file__` — five parents is the repo root in a checkout but `/` inside the image. With either half wrong, `export_html`'s try/except silently returns `_generate_basic_html` and the template is dead in every deployment while alive locally. A test asserting only "HTML came back" passes in both broken states; assert on something the TEMPLATE emits (the `<name> - Tab Organizer Export` title suffix). `templates/` is also shipped into the test image for the same reason.
- irrelevant or incomplete code to remove/rework: `export_notion` removed 2026-08-07 (63 lines, never in the `export()` dispatcher, so `format="notion"` raised "Unsupported export format" before and after). `templates/{markdown,json,obsidian}_default.j2` remain unused — `export_markdown`/`export_json`/`export_obsidian` build their output as Python strings and never touch Jinja; only `export.html.j2` is ever loaded.
- docs that must stay aligned: README export feature list and `docs/ARCHITECTURE_PLAN.md`.
- local validation commands/checks: `make test-backend`; add focused export tests before behavior changes.
