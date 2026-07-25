# Browser Tabs Submodule Card

- purpose: attach to a user-owned Chromium/Chrome debugging endpoint and import/open tabs.
- product/module functionality: CDP endpoint validation, live tab inventory, readable tab extraction, attached-browser tab opening.
- scope boundaries: owns browser control-plane interaction only; backend persistence and AI indexing contracts belong to connected services.
- connected modules/submodules: Browser routes, Backend Core tab import/open APIs, AI Engine indexing, Ops CLI/MCP tools.
- allowed change types: CDP attach behavior, local endpoint validation, import/open result contracts, focused tests.
- special operating rules: attach-only v1; never close the user's browser/profile; only local CDP endpoints are allowed; imported page URLs still pass scrape URL safety unless explicitly opted in later.
- what "local CDP endpoint" means (two gates, both in `cdp.py`):
  - *input* gate — `validate_cdp_url` accepts only http/https, no credentials/path/query, and a hostname in `LOCAL_CDP_HOSTS` (`localhost`, `127.0.0.1`, `::1`, `host.docker.internal`). IPv6 literals keep their brackets through every netloc rebuild (`_format_netloc`), so `[::1]:9222` never degrades into the different host `::1:9222`.
  - *resolved-address* gate — `resolve_cdp_connect_url` resolves `host.docker.internal` (Chrome's debug port rejects that name in the Host header) and admits an answer only if it falls inside `LOCAL_CDP_NETWORKS`, a deliberate **allowlist**: `127.0.0.0/8`, `10.0.0.0/8`, `172.16.0.0/12` (docker0 bridge), `192.168.0.0/16`, `::1/128`, `fc00::/7`. Do not swap this for `is_private`/`not is_global` — both are True for 6to4 `2002::/16` (which can encode any global IPv4), `198.18.0.0/15`, `192.0.0.0/24`, and `2001:db8::/32`. **Adding a network to that tuple is a security decision.**
- CDP address selection: among allowlisted answers, IPv4 is preferred (resolver order breaks ties inside a family). AF_UNSPEC lookups are RFC 6724-sorted and would otherwise prefer the AAAA record, while the documented `socat ... bind=172.17.0.1` bridge is IPv4-only. Non-allowlisted answers are filtered out and logged (`cdp.resolved_address_filtered`), not treated as poisoning the whole lookup; the connect URL always carries a pinned IP literal, which is what closes DNS rebinding.
- known gap (tracked, not closed here): `connect_over_cdp` fetches `/json/version` from the pinned endpoint and then dials the `webSocketDebuggerUrl` that endpoint returns; that second hop is not re-validated against the allowlist.
- current stubs/placeholders: browser launch mode, rofi/fzf, and TUI flows are isolate-for-later.
- irrelevant or incomplete code to remove/rework: no `ichrome` dependency is used in v1; Playwright CDP attach is the documented contract.
- docs that must stay aligned: Browser Engine card, Backend Core card, Ops Tooling card, README, architecture/testing docs.
- local validation commands/checks: `make test-browser`; focused file `tests/unit/test_browser_tab_harvester.py`.
