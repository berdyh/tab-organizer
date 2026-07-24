# Ops Tooling Module Card

- purpose: command-line lifecycle and local developer operations.
- product/module functionality: Docker Compose start/stop/logs/status/test, host-AI runtime, provider checks, Backend Core tab commands, local MCP tab wrappers, init/model setup, cleanup.
- scope boundaries: orchestrates services but does not implement service-domain behavior.
- connected modules/submodules: Docker Compose, AI Engine, Browser Engine, Backend Core, Test Harness, Shared Config.
- allowed change types: CLI parser fixes, token/bootstrap generation, host-AI environment handling, Backend Core ops wrappers, test wrapper additions, docs.
- special operating rules: never print generated service tokens or `BACKEND_AGENT_API_TOKEN`; preserve host-AI token file permissions and container-to-host routing behavior.
- current stubs/placeholders: performance/load test wrapper is not part of `scripts/cli.py test` today; Textual TUI, rofi/fzf quick-pick, and a full MCP SDK server are isolate-for-later.
- irrelevant or incomplete code to remove/rework: outdated comments or test wrappers that point at old absolute paths.
- docs that must stay aligned: README CLI section, `docs/DEVELOPMENT.md`, `docs/TESTING.md`, `tests/README.md`.
- local validation commands/checks: `make test-ops`; `PYTHONPYCACHEPREFIX=/tmp/tab-organizer-pycache python -m pytest tests/unit/test_cli_host_ai.py -q`; `PYTHONPYCACHEPREFIX=/tmp/tab-organizer-pycache python -m compileall -q scripts config`.
