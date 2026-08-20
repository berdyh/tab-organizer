# Browser Tabs Submodule Card

- purpose: attach to a user-owned Chromium/Chrome debugging endpoint and import/open tabs.
- product/module functionality: CDP endpoint validation, live tab inventory, readable tab extraction, attached-browser tab opening.
- scope boundaries: owns browser control-plane interaction only; backend persistence and AI indexing contracts belong to connected services.
- connected modules/submodules: Browser routes, Backend Core tab import/open APIs, AI Engine indexing, Ops CLI/MCP tools.
- allowed change types: CDP attach behavior, local endpoint validation, import/open result contracts, focused tests.
- special operating rules: **a tab is imported, skipped, or failed — never
  imported empty.** `_harvest_page` returns `HarvestedTab` or `SkippedTab`, and
  `SkippedTab` carries a reason (`auth_wall`, `blank`, `extraction_failed`) plus
  a detail a human can act on. `TabHarvestResult.total` counts skips, so a
  skipped tab cannot vanish from the arithmetic as well as the payload. Before
  this, a blank tab was imported as `content: ""`, which read as success
  everywhere until the embedding provider rejected the whole batch it travelled
  in. A **sign-in page is skipped, not captured**: its text is the login form
  rather than the page the user has open, and embedding it files the prompt
  under the tab's title so the logged-OUT page answers searches for the
  logged-IN one. The classifier is the one SEC-39 freezes, and it answers "is
  this page ASKING me to sign in" — a page the user is already signed in to
  reports False and is captured normally (verified against real tabs:
  claude.ai, notion, grammarly all classify as not-auth-walled). A
  `bot_challenge` signal is recorded in metadata but never skipped on, because
  those pages carry real content often enough that skipping loses more than it
  saves (replit.com: 0.9 confidence, 3.4k characters of real text). When a tab
  has no DOM text, `app/extraction/fallbacks.py` gets a turn before it is called
  blank. Also: attach-only v1; never close the user's browser/profile; only local CDP endpoints are allowed; imported page URLs still pass scrape URL safety unless explicitly opted in later.
- what "local CDP endpoint" means (three gates, all in `cdp.py`):
  - *input* gate — `validate_cdp_url` accepts only http/https, no credentials/path/query, and a hostname in `LOCAL_CDP_HOSTS` (`localhost`, `127.0.0.1`, `::1`, `host.docker.internal`). IPv6 literals keep their brackets through every netloc rebuild (`_format_netloc`), so `[::1]:9222` never degrades into the different host `::1:9222`.
  - *resolved-address* gate — `resolve_cdp_connect_url` resolves `host.docker.internal` (Chrome's debug port rejects that name in the Host header) and admits an answer only if it falls inside `LOCAL_CDP_NETWORKS`, a deliberate **allowlist**: `127.0.0.0/8`, `10.0.0.0/8`, `172.16.0.0/12` (docker0 bridge), `192.168.0.0/16`, `::1/128`, `fc00::/7`. Do not swap this for `is_private`/`not is_global` — both are True for 6to4 `2002::/16` (which can encode any global IPv4), `198.18.0.0/15`, `192.0.0.0/24`, and `2001:db8::/32`. **Adding a network to that tuple is a security decision.**
  - *advertised-socket* gate (the second hop, previously a tracked gap, now closed) — handing `connect_over_cdp` an http(s) URL pins only the FIRST hop: playwright-core's `urlToWSEndpoint` fetches `/json/version` and dials whatever `webSocketDebuggerUrl` the body contains, following cross-host 3xx redirects and honouring proxy env vars on the way (`server/chromium/chromium.js` + `utils/network.js`, verified identical in the pinned 1.41.0 and in 1.57.0). So `resolve_cdp_ws_endpoint` does that fetch here instead — `follow_redirects=False`, `trust_env=False`, one timeout, a 64 KiB body cap — and `validate_ws_debugger_url` measures the advertised socket against the same rules the connect URL passed: ws/wss only, no credentials, no control characters or whitespace, the exact pinned host (compared as addresses when both parse as one; `127.0.0.1` deliberately does NOT match `localhost`), the exact pinned port, and — checked independently of the host comparison — membership in `LOCAL_CDP_NETWORKS`. The accepted endpoint is re-emitted from parsed parts in canonical form, so the string Node parses cannot spell the host any other way, and Playwright's `ws://` short-circuit means it never re-fetches. Anything else raises `CDPConnectionError(code="cdp_ws_endpoint_not_local")` naming what was advertised, after a `cdp.ws_endpoint_rejected` WARNING. `webSocketDebuggerUrl` is the only field of that response any consumer reads, and nothing here or in Playwright's attach path fetches `/json/list` — target URLs arrive over the validated socket and still pass `_is_importable_page`.
- CDP address selection: among allowlisted answers, IPv4 is preferred (resolver order breaks ties inside a family). AF_UNSPEC lookups are RFC 6724-sorted and would otherwise prefer the AAAA record, while the documented `socat ... bind=172.17.0.1` bridge is IPv4-only. Non-allowlisted answers are filtered out and logged (`cdp.resolved_address_filtered`), not treated as poisoning the whole lookup; the connect URL always carries a pinned IP literal, which is what closes DNS rebinding.
- current stubs/placeholders: browser launch mode, rofi/fzf, and TUI flows are isolate-for-later.
- irrelevant or incomplete code to remove/rework: no `ichrome` dependency is used in v1; Playwright CDP attach is the documented contract.
- docs that must stay aligned: Browser Engine card, Backend Core card, Ops Tooling card, README, architecture/testing docs.
- local validation commands/checks: `make test-browser`; focused files `tests/unit/test_browser_tab_harvester.py`, `tests/unit/test_cdp_second_hop.py` (the latter runs a real stand-in debug server on loopback rather than mocking the HTTP client, so it exercises the redirect/proxy/body-cap behavior of the actual client).
