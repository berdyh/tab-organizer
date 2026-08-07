# tab-organizer — Storage, Scraper, and Agent-Portal Restructure

Branch: `agent-platform-subscription-routing-review`
Date: 2026-07-23 (last revised 2026-07-24)
Status: **REVIEWED AND LOCKED — this is the single source of truth.**

Reviewed by 7 advisor passes (Fable ×6; Codex gpt-5.1-codex; Codex gpt-5.6-sol @ xhigh).
41 decisions recorded; 2 non-blocking items open (see the end of this file). Every other
plan, spec, and review doc in this repo is superseded — see "SINGLE SOURCE OF TRUTH" below
for the archival inventory.

Work item zero is a real-tab end-to-end run: the live database shows `tab_import_jobs = 0`
and `tab_search_fts = 0` rows despite 15 successful scrapes, so CDP attach has never been
exercised and the scrape→FTS write path is broken in production.

> This file is the repo copy of the single source of truth. The upstream original (edited
> live during review) lives at
> `~/.gstack/projects/berdyh-tab-organizer/agent-platform-subscription-routing-review-design-20260723.md`
> on the author's machine — not fetchable from this repo. If the two ever diverge, this
> repo copy is authoritative for engineering work; reconcile the upstream copy by hand.

## Problem statement

tab-organizer is an "oracle over your open tabs": it reads the *content* of your
tabs (not just link titles), makes it semantically searchable, clusters it by the
user's need, answers questions over it, and runs saved tasks over it. The loop
(attach → scrape content → chunk → index → hybrid search → chat) works end to end
today, but the pieces that make it an oracle rather than a search box are missing
or misbuilt.

Division of labor (decided this session):
- **Local owns**: the database, the authenticating scraper (with securely stored
  credentials), and embeddings (API *or* local, chosen in settings).
- **Coding agents on the user's own subscription own the thinking**: labeling,
  grouping, summarizing, and user-authored tasks. Multiple agents (Claude Code,
  Codex, Gemini, others) must work, and an agent session can be launched from the
  webpage itself.

## Premises (CONFIRMED 2026-07-23 via /autoplan premise gate + user revisions)

1. **Embeddings: cloud provider is the DEFAULT, local (Ollama) is an opt-in fallback,
   chosen in settings and on initial setup.** (Revised from local-first.) Tradeoff
   accepted: cloud embedding means page text leaves the machine for the embedding API.
2. Bulk mechanical generation (cluster labels, facet extraction, per-page summaries)
   runs on a **cheap completion tier** (cheap API default, local optional) — NOT the
   user's subscription. The user's coding-agent subscription is reserved for
   **interactive / user-authored tasks**. Agent execution sits behind a swappable
   `AgentRunPort` so the app can later defer OFF subscriptions to LangGraph/deep-agents
   or a standalone API-key agent — one adapter file per backend. `agent_cli.py` is NOT
   retired; it becomes the first `AgentRunPort` adapter (or is replaced by the Claude
   Agent SDK, which works with both subscription auth and an API key).
3. Agents act on the corpus through a shared callable contract with two skins: a
   CLI (`tabctl`) for reads/aggregations, and typed writes for structured write-back.
   The stdio MCP skin ships **in-phase** (the webpage-launched/sandboxed case is
   already in scope, and `tabs.py` was built for it).
4. Agents receive page *content*; agents never receive *credentials*. **Enforced at
   the process boundary, not just the API**: the credential holder runs in a different
   security context than agents (separate container/UID; passphrase never in
   agent-reachable env). API-level rules are defense-in-depth against a confused/injected
   agent — the real threat model, since the agent is the user's own subscription on the
   user's own machine.
5. Agents write *structure* (facets, groupings, labels); agents never mutate
   `pages.text`. Corpus text is immutable; agent claims are grounded in **cited chunk
   ids**, not merely a content hash.

**Corollary to premise 4 (added 2026-07-24, wk0 security review).** Premise 4 above
reads as though the API-level defense-in-depth layer already exists. It did not. The wk0
review found the credential-proxy endpoints carrying no auth dependency at all, every
service accepting cross-origin requests from any website with credentials enabled, and
`scripts/cli.py` minting one token value under three names — so the agent credential was
byte-identical to the one guarding the credential and CDP control plane. All three are
fixed in wk0 (`0a22621`, and the CORS/exfil commit that follows it), but the lesson is
structural: **the API layer is a required deliverable, not an assumed property.** Until
every credential-adjacent and corpus-read endpoint carries an auth dependency and
cross-origin reach is closed, premise 4 rests on the process boundary alone and the
"defense-in-depth" clause is aspirational. The TS port inherits this obligation — the
frozen suite's route-auth and CORS probes exist to keep it honest.

A second, subtler failure the same review exposed: the frozen suite's harness synthesizes
four distinct tokens, a configuration `cli.py` could not produce. Its scope-isolation
invariants therefore passed while the shipped stack did the opposite. **A test that
constructs its own environment certifies that environment, not the product.** Deployment-
shape assertions belong in tooling unit tests, not in the portable black-box suite.

**Capture method (User Challenge D9):** KEEP the credential-store path — scraper stores
passwords, authenticates, re-fetches. Rejected the extension/tab-capture pivot because
scheduled headless re-fetch (tab closed / page changed) genuinely needs stored creds.
The keyring rewrite (OS keyring, passphrase fallback, FAIL CLOSED) proceeds.

## SINGLE SOURCE OF TRUTH (user directive 2026-07-24)

**This document is the only plan.** Every other plan/spec/review doc is superseded and must
be archived or deleted. No competing ledgers. Inventory of what that covers (35 docs):

| Doc | Disposition |
|---|---|
| `docs/REPO_IMPROVEMENTS_REVIEW.md` (PR #9, just merged) | Content folded in as items 27-35 → **archive**; it is now a competing ledger |
| `docs/MODULE_INDEX.md` + 22 `MODULE.md` cards | Describe the 4-service Python topology being replaced. **Archive**; regenerate cards per TS module as each lands |
| `docs/ARCHITECTURE.md` | **Archive**, supersede with a TS architecture doc at cutover |
| `docs/AI_CONFIG.md` | Rewrite for AI SDK provider config; archive the provider-matrix content |
| `docs/TESTING.md` | Rewrite around the security-invariant suite + TS tests |
| `docs/DEVELOPMENT.md`, `docs/MANUAL_SETUP.md` | **Archive** — Docker-compose workflow disappears |
| `docs/REQUIREMENTS.md`, `docs/SCRAPER_AGENTIC_ANALYSIS.md` | **Archive** |
| `README.md` | Rewrite (npx quickstart), do not accrete |
| `CLAUDE.md` / `AGENTS.md` | Rewrite as the agent-facing contract (the DX review's executable quickstart) |
| `issue-resolving-notes.html` (45KB, repo root) | **Delete** — scratch notes in the repo root |

Archive convention: move to `docs/archive/<date>-<name>.md` with a one-line header stating
what superseded it and when. Never leave a stale doc live; a stale doc is worse than none.

## Codebase cleanup (user directive 2026-07-24)

Cleanup runs **during and after** implementation, not as a deferred phase. Standing rule:
every phase deletes what it replaces in the same commit. Known targets, with their
status as of 2026-08-07: dead `services/web-ui/.env.example` (React vars Streamlit never
reads) — **done**, and its sibling `services/web-ui/.gitignore` with it; unused
`BackgroundTasks` param + stale docstring — **done** (the line numbers recorded here,
`routes.py:572/604`, had themselves gone stale; the real sites were `:631` and `:686`);
hardcoded inter-service URLs — **done** (finding 32); legacy timestamp deprecations and
the B2B platform subsystem (pending Q6) — still open, the latter deliberately so under
decision 41; and every Python module retired by the cutover.

## New workstreams (user directives 2026-07-23)

- **Cleanup**: remove junk/stale/dead code as an explicit pass. First targets: the
  B2B `platform` subsystem (993-line store + 733-line UI with no local-first user),
  the unused `BackgroundTasks` param + stale docstring (routes.py:572/604), legacy
  timestamp deprecations, obsolete comments/wrappers.
- **Modular monorepo (Humora-v2-mono style)**: reorganize into clear module/submodule
  package boundaries with per-module MODULE.md cards (extends the existing card system).
  Reconcile with reviewer pushback: modular *packages*, but consider FEWER runtime
  services (ai-engine embeddings are a library call; LanceDB is already in-process) —
  see Phasing.

## Current-state defects (evidence)

- Page text stored 3x with no shared key: `url_records.metadata["content"]` (SQLite),
  `tab_search_fts.content` (FTS5), and LanceDB chunk rows. `content_hash` /
  `embedding_id` columns exist (`sessions/manager.py:110-111`) but nothing joins on
  them.
- Clustering re-embeds the *whole page* as one vector (`clustering/pipeline.py:270-273`),
  silently truncated to the embedding model's token window. Root cause of
  "unimportantly labeled tabs" — a truncation problem, not a labeling problem.
- "Cluster by user need" is structurally impossible: one fixed vector per page →
  exactly one canonical partition, no axis parameter.
- Cluster identity is ephemeral: HDBSCAN int labels regenerate per run
  (`pipeline.py:191`); any user relabel is lost on re-cluster.
- Recursion bug: `pipeline.py:296-299` recurses into any cluster >10 tabs with no
  depth guard; a cluster HDBSCAN won't subdivide recurses forever → RecursionError.
- Credential store: `auth/queue.py:44-53` generates a throwaway Fernet key when
  `CREDENTIAL_ENCRYPTION_KEY` is unset, over a process-memory dict. Encryption theater;
  no fail-closed.
- Scrape task state is in-process memory (`browser-engine/app/main.py:53`); restart
  mid-batch loses all status.
- `trigger_scraping` is now awaited inline (`routes.py:590`) but still takes an unused
  `BackgroundTasks` param and a stale "Background task" docstring — undocumented
  behavior change from commit 3f1dcd8, ledger/card not updated.
- Module-card drift: scrape-dispatch stub rows in `MODULE_INDEX.md` and two cards
  describe pre-3f1dcd8 behavior.

## Proposed storage design

Content-addressed, single source of truth, model-keyed vectors, agent write-back.

```
pages   (immutable): content_hash PK | url_norm | fetched_at | title | text | lang
                     | http_status | auth_used            ← the ONLY copy of text
chunks  (1:N):       chunk_id=hash(content_hash,idx) | idx | char_start/end | text | heading_path
vectors (LanceDB):   chunk_id | model_id | dim | embedding | created_at   ← keyed by (chunk_id, model_id)
tabs    (mutable):   tab_id | session_id | url_norm | content_hash FK | status
facets  (writeback): content_hash | key | value | source(agent:<id>|user) | prompt_id | model_id | run_at
groupings:           grouping_id | name | axis | run_at | agent_id
  groups:            group_id | label | rationale
  members:           group_id | content_hash | confidence
  overrides:         content_hash | group_id | source='user'   ← user wins on every re-run
```

Rules:
- `content_hash` is the join key everywhere; FTS5 becomes a derived index over
  `pages.text`, not a second copy; `url_records.metadata` holds the hash, not the blob.
- Vectors keyed by `(chunk_id, model_id)` → model switch is an incremental backfill,
  not a wipe-and-reindex. Fixes the `/providers/switch` dead-end (refuses switch while
  indexed, with no clear-all/reindex endpoint).
- Document vector = mean of chunk vectors, computed on read. Never re-embed a truncated page.
- `facets` = the API surface agents write through; "cluster by language" becomes
  `GROUP BY facet_value` — deterministic, explainable, re-runnable with no re-embed.
  Geometry (UMAP+HDBSCAN) only handles the freeform case.
- `groupings` + `overrides` give clusters stable identity; a user relabel is a row
  that wins on every subsequent run.
- Every agent-produced row stamped with agent id, model, prompt version, run timestamp
  (replayability; matches prior learning `prompt-versioning-immutable-files`).

## Proposed service restructure

```
backend-core/  owns SQLite (pages, chunks, tabs, facets, groupings)
  app/api/routes.py            ← stays, aggregator only
    sessions.py urls.py scrape.py tabs.py search.py export.py platform.py
    tasks.py     ← NEW user-authored agent task defs
    groupings.py ← NEW agent/user write-back
  app/content/                 ← NEW pages+chunks, content-hash identity
  app/platform/store.py → auth.py tokens.py companies.py dashboard.py issues.py

ai-engine/     shrinks toward embeddings + retrieval
  app/main.py → routes/{health,providers,embed,index,search}.py
  app/embeddings/  ← API and local, selected in settings
  app/clustering/  ← geometry only; labels come from agents

browser-engine/  scraper + auth (the other load-bearing local piece)
  app/main.py → routes/{scrape,tabs,auth}.py
  app/scraper/engine.py → http.py browser.py batch.py callbacks.py
  app/extraction/  ← finally real (today re-exports from scraper)
  app/auth/store.py ← NEW OS keyring, passphrase fallback, FAIL CLOSED

agent-gateway/   NEW — the portal
  mcp_server.py  ← promote scripts/mcp/tabs.py: search, chunks, write_facet, create_grouping, label_group
  runner.py      ← spawn CC/codex/gemini as PTY, stream to UI
  tasks.py       ← run a saved task, persist transcript

web-ui/  dashboard over all of it
  src/pages/dashboard.py tabs_table.py clusters.py tasks.py agent_chat.py
  platform.py → split by panel
```

## Agent contract: CLI-first, shared core, MCP later

- Ship `tabctl` with `--json` on every read verb, `--json -` (stdin) on every write verb.
- Document in CLAUDE.md / AGENTS.md → every current and future agent works, zero
  per-agent code, zero context tax.
- Keep `scripts/mcp/tabs.py` as THE contract; CLI and (future) MCP server are thin skins.
- Reads/aggregations → CLI + jq (composition wins). Structured writes → typed stdin JSON.
- Add MCP skin when a real case needs it (sandboxed/remote agent, or webpage-launched
  session where handing curated tools beats teaching a CLI).
- Multi-agent support comes from the standard (MCP / documented CLI), NOT from N adapters.
  Worktree isolation (vibe-kanban et al) does not apply: agents mutate a tab corpus,
  not a git repo.

## Security boundaries (write into module cards day one)

- Agents get content, never credentials. Scraper authenticates and returns text;
  keyring behind browser-engine; agent-gateway cannot reach it.
- Agents write structure, never raw content. No tool mutates `pages.text`.
- Credential store: OS keyring first (Secret Service), file+passphrase-derived key
  (argon2id→Fernet) fallback, FAIL CLOSED — never invent a key.

## Out of scope (deferred)

- Full BERTopic topic modeling (clustering stays UMAP+HDBSCAN+facet GROUP BY).
- Textual TUI / rofi-fzf quick-pick.
- Platform B2B/maintainer expansion beyond current local-demo behavior.
- A hosted multi-tenant deployment (this is local-first).

## Schema must-fixes (forced by review, settle BEFORE code)

- **E1 — split content identity from capture provenance.** `pages` = pure content
  (`content_hash, text, title, lang`). New `fetches` (or `captures`) table holds
  `url_norm, fetched_at, http_status, auth_used`. Two URLs with identical text collide
  cleanly; re-fetch is a new capture row, not a mutation.
- **E2 — stable group identity (CRITICAL).** Overrides keyed on `(content_hash, group_id)`
  cannot "win on re-run" because each run mints new group_ids — the same ephemeral-identity
  bug the plan calls out for HDBSCAN labels, one level up. Fix: persistent group entities
  that successive runs UPDATE (matched by member-overlap / label), OR key overrides on a
  stable `(axis, canonical_label)` with a defined resolution rule. Define the override-verb
  vocabulary explicitly: `relabel(group→text)`, `merge(a,b)`, `pin(page→group)`,
  `exclude(page)` — each a typed, provenance-stamped row that IS the typed-write contract
  for `tabctl`.
- **E3 — content_hash = H(extractor_id ‖ extractor_version ‖ chunker_version ‖
  canonical(text))** with canonical pinning NFC + line endings + whitespace. An extractor
  upgrade is a declared re-derivation event, not a silent orphaning.
- **E4 — one LanceDB table per model_id** (fixed-size list can't mix dims). Drain =
  drop table A. Migration from current layout is a full re-embed (no old→new chunk_id
  map); surface cost/consent, since cloud embedding is now the default.
- **E5 — materialize the document mean-vector at index time** (`doc_vectors` cache),
  never compute on read.
- **Read rule during backfill**: query the TARGET model's table only; hybrid keyword
  leg covers not-yet-backfilled docs; expose progress to the UI.
- **Single ingest writer**: backend-core owns ingest; browser-engine calls back only to
  backend; backend forwards chunks to ai-engine. Kills the current dual best-effort writer.

## Agent execution abstraction (two ports)

- **CompletionPort** (exists as `BaseLLMProvider`): add `generate_json(prompt, schema)`
  with one repair retry; native structured-output adapters override. Bulk tier lives here.
- **AgentRunPort** (new, ~80 lines, stdlib): `run(AgentTask) -> AsyncIterator[AgentEvent]`,
  `cancel(task_id)`, `is_available()`, `capabilities`. `AgentTask` carries `context_refs`
  (IDs, not inlined content), `tools` (names from a small app-owned ToolRegistry),
  `output_schema`, `budget(max_steps, max_seconds)`. Stream EVENTS (status/tool_call/
  result), not tokens. Tools inverted: app owns the registry; adapters bridge (MCP for
  CLI adapters, LangChain tools for a deep-agent adapter, none for plain completion).
- **v1**: one `ClaudeAgentAdapter` (Agent SDK preferred over shelling `claude -p` —
  typed events, works with subscription OR API key). Skip router/checkpointing/LangChain
  dep. Rides the existing `providers/MODULE.md` "split agent_cli.py by base class" mandate.
- **Do NOT let LangGraph/LangChain types leak into the port signature** — that kills the swap.

## STACK DECISION (2026-07-24): full TypeScript, strangler-fig migration

Supersedes the Python service-restructure section above. Rationale: every genuinely
hard requirement is TypeScript-first, with Python as the port.

| Need | TS package | Replaces |
|---|---|---|
| Providers, streaming, tool calls, structured output | Vercel AI SDK (`ai`) | `llm_client.py` + all `providers/*` + hand-rolled CompletionPort |
| Agent execution + MCP client | `@anthropic-ai/claude-agent-sdk` | `agent_cli.py`, host-AI mode, the PTY runner, hand-rolled AgentRunPort |
| Vector store | `@lancedb/lancedb` | ai-engine LanceDB layer |
| Browser automation / CDP attach | Playwright Node (`connectOverCDP`) | browser-engine scraper (Node is upstream; Python is the port) |
| Extraction | `@mozilla/readability` | `ContentExtractor` (again, the reference impl) |
| MCP server | `@modelcontextprotocol/sdk` | `scripts/mcp/tabs.py` stdio adapter |
| SQLite + FTS5 | `better-sqlite3` / `libsql` | `sessions/manager.py` persistence |
| CLI + distribution | npm bin / `npx` / Tauri | `scripts/cli.py`, Docker Compose, `tabctl` packaging problem |

**Known gap:** UMAP + HDBSCAN have no mature TS equivalent (projects needing them shell
out to Python). Dissolved by decisions already made: facet-based grouping (`GROUP BY
facet_value`) is the primary axis, geometry handles only the freeform case, and at
hundreds-to-low-thousands of pages, agglomerative/k-means over cosine similarity is
adequate. UMAP is mostly preprocessing for HDBSCAN; dropping one drops the need for both.

**Keep a thin agent port anyway** (premise 2): the Agent SDK is the first adapter, not
the interface. A `LangGraphAdapter` / plain-API adapter must remain a one-file addition.
Do not let SDK types leak into the port signature.

**Target shape** (one process, pnpm/turbo workspace = the modular-monorepo directive):

```
app/            Next.js or TanStack Start — dashboard, tabs, clusters, tasks, chat, settings
server/
  capture/      Playwright CDP attach, readability extraction, credential store (keyring)
  content/      pages, chunks, content-hash identity (better-sqlite3 + FTS5)
  vectors/      LanceDB TS, one table per model_id
  grouping/     facets, groupings, overrides, override-verb vocabulary
  agents/       AI SDK (bulk tier) + Claude Agent SDK (interactive/tasks) behind a thin port
  mcp/          MCP server exposing corpus tools
cli/            tabctl (npm bin)
```

Resolves the three previously-unresolved taste decisions:
- **Service topology** → one process; the 5-service Docker mesh disappears.
- **Webpage-launched agent sessions** → natural via Agent SDK typed events + AI SDK
  streaming. No PTY, no ANSI parsing, no Streamlit hosting problem.
- **B2B platform** → not ported. Deleted by omission, no split tax paid.

**MODULE_INDEX.md boundary rule "preserve the four runtime services / this rework is not
a service split" is explicitly superseded by this decision** — update it in the same commit.

## Migration order (strangler fig — port behind existing HTTP contracts)

1. **TS app shell + agent layer + UI** — new code that doesn't exist in a form worth
   keeping. Calls the existing Python APIs. Delivers the interactive dashboard and agent
   sessions first. (Time-boxed dual-stack period starts here.)
2. **Vectors + embeddings** — LanceDB TS + AI SDK `embed`/`embedMany`. Contracts:
   `/embed`, `/index`, `/search`. Retire ai-engine.
3. **Content + storage** — better-sqlite3 with the content-addressed schema (E1-E5)
   **built once, in TS**. Do NOT implement E1-E5 in Python first; that is doing it twice.
   Retire backend-core.
4. **Capture** — Playwright Node + credential store (keyring, fail-closed). Ported LAST
   because it is the most security-sensitive and the Python version works today.
   Retire browser-engine.
5. Collapse to one process; delete Docker Compose; ship `npx` / Tauri binary.

**Phase-1 bug triage under migration** — fix in Python only what the migration won't fix:
- Recursion depth guard (`pipeline.py:296`) — live crash, ~2 lines. **Fix now in Python.**
- Fail-closed credentials (`auth/queue.py:44`) — capture ports last, so this Python code
  lives longest. **Fix now in Python.**
- Truncated whole-page re-embed + in-memory scrape state — **do not fix twice**; both are
  resolved by the TS storage/capture ports. Accept until then.

## Phasing (superseded above — retained for the storage/schema decisions it carries)

The repo's own `MODULE_INDEX.md` says "this rework is not a service split." Honor it by
sequencing, and update the boundary rule explicitly when the split begins.

- **Phase 1 (days, ships first)**: the 4 verified bugs — recursion depth guard
  (pipeline.py:296), fail-closed credentials (auth/queue.py:44), chunk-mean document
  vectors instead of truncated whole-page re-embed (pipeline.py:270), scrape-state
  durability (make browser-engine stateless; backend derives batch status from callbacks).
  Plus the read-side agent contract (`tabctl` search/get-page/get-chunks + stdio MCP skin)
  and card/ledger reconciliation. This is shippable value on the current architecture.
- **Phase 2**: content-addressed storage (E1-E5), single ingest writer, facet write-back,
  the CompletionPort bulk-labeling tier. Prove labels a human nods at BEFORE the UI rework.
- **Phase 3**: AgentRunPort + ClaudeAgentAdapter; user-authored tasks; interactive agent
  sessions (server-owned in agent-gateway, structured events, token-gated spawn endpoint,
  fixed corpus-tool allowlist — NOT PTY, NOT Streamlit-hosted).
- **Phase 4**: dashboard rework (state map per surface; interactive cluster review with
  the override-verb vocabulary), modular-monorepo reorg, cleanup pass.
- **Service topology (taste, see gate)**: modular packages regardless; consider folding
  ai-engine into backend-core as a library (LanceDB already in-process) to cut the
  five-service tax; keep browser-engine separate for the credential/sandbox boundary.

## Test intentions

- Content-hash canonicalization: tested pure function (idempotence breaks silently
  otherwise — prior learning `raw-ingest-content-hash-idempotence`).
- Clustering recursion depth guard: regression test with a non-subdividing cluster.
- Model-switch backfill: index under model A, switch to B, assert both coexist then A drains.
- Credential store: fail-closed when no key AND no keyring; round-trip through keyring.
- Scrape-state durability: restart browser-engine mid-batch, status survives.
- FTS rebuild equivalence (same keyword results pre/post migration).
- In-memory vs SQLite parity for the new content store (existing CLAUDE.md invariant).
- Concurrent same-hash ingest: idempotent upsert.
- Negative authz: agent token writes facets but is rejected mutating page text / reaching
  browser-engine auth endpoints (executable form of the security boundary).
- Migration mapping: old LanceDB rows dropped and re-embedded (no old→new chunk_id map).
- Override survival across re-runs (forces the E2 stable-identity decision).
- generate_json schema-validation + one-repair-retry on malformed output.
- Error contract: every tabctl/API error asserts `{code, cause, fix}`.

## Decision Audit Trail

| # | Phase | Decision | Class | Principle | Rationale |
|---|-------|----------|-------|-----------|-----------|
| 1 | CEO | Embeddings cloud-default, local optional | User-directed | — | user premise revision |
| 2 | CEO | Agent execution behind swappable AgentRunPort | User-directed | — | defer-off-subscriptions |
| 3 | CEO | Bulk labeling → cheap completion, not subscription | Taste | P1+P3 | quota/latency/ToS (both CEO voices) |
| 4 | CEO | Capture: keep credential store | User Challenge (kept original) | — | needs headless re-fetch |
| 5 | CEO | Sequence into 4 phases, bugfix slice first | Mechanical | P6+P2 | reviewers unanimous, no big-bang |
| 6 | Eng | Split content identity from capture provenance (E1) | Mechanical | P5 | both voices |
| 7 | Eng | Stable group identity before schema freeze (E2) | Mechanical | P5 | override "win" unimplementable otherwise |
| 8 | Eng | content_hash includes extractor+chunker version (E3) | Mechanical | P5 | silent-orphan prevention |
| 9 | Eng | One LanceDB table per model_id (E4) | Mechanical | P5 | fixed-size list can't mix dims |
| 10 | Eng | Materialize doc mean-vector at index time (E5) | Mechanical | P3 | avoid 160MB/run on-read |
| 11 | Eng | Single ingest writer (backend-core) | Mechanical | P5 | kills dual best-effort writer |
| 12 | Design | Agent sessions server-owned, structured events, NOT PTY/Streamlit | Mechanical | P5 | ANSI tar-pit; Streamlit can't host live |
| 13 | Eng | Security boundary = process-level; API = defense-in-depth | Mechanical | P5 | host-spawned agent reads .env/keyring |
| 14 | CEO | Cleanup + modular-monorepo workstreams | User-directed | — | user directive |

## GSTACK REVIEW REPORT

| Field | Value |
|---|---|
| Runs | CEO (Claude+Codex+Fable), Eng (Claude+Fable), Design (Fable), DX (Fable), Abstraction advisory (Fable+web) |
| Advisors | Fable ×5; Codex default/gpt-5.1-codex (gpt-5.6 unavailable on ChatGPT-account auth — substituted, flagged) |
| Status | REVIEWED — foundations confirmed, plan revised per user premises + convergent findings |
| Verified bugs | recursion pipeline.py:296 (High); credential fail-open auth/queue.py:44 (Med→High); truncated re-embed; in-memory scrape state — all Phase 1 |
| CEO consensus | 2/6 confirmed (sequence, examine-alternatives), 3 taste; subscription-fragility CONFIRMED |
| Eng consensus | 6 schema must-fixes (E1-E6); both claimed bugs REAL |
| Design/DX | 6 critical/high (PTY, Streamlit-hosting, CDP onboarding, TTHW, tabctl contract, cluster UI) |
| Degradation | Separate Codex Design/DX/Eng voices not run — folded into Codex CEO pass (eng/security depth). Single-model for those dimensions. |

VERDICT: PROCEED with the revised, phased plan. Phase 1 is unblocked and ships on current
architecture. Phases 2-4 gated on the E2 stable-group-identity design decision and the
service-topology taste call.

### Addendum 2026-07-24 — stack decision

| # | Decision | Class |
|---|---|---|
| 15 | Full-TypeScript single-process app, strangler-fig migration behind existing HTTP contracts | User-directed (D11) |
| 16 | Build content-addressed storage (E1-E5) once, in TS — not in Python first | Mechanical (P4 DRY) |
| 17 | AI SDK + Claude Agent SDK replace the hand-rolled two-port design; keep a thin port so LangGraph.js/plain-API stay one-file adapters | Mechanical (P4) |
| 18 | Service topology → one process (was unresolved) | Resolved by 15 |
| 19 | Webpage-launched agent sessions → in scope, via Agent SDK typed events (was unresolved) | Resolved by 15 |
| 20 | B2B platform → not ported, deleted by omission (was unresolved) | Resolved by 15 |
| 21 | Python-side fixes limited to recursion guard + fail-closed credentials; storage/capture bugs fixed by the port, not twice | Mechanical (P4) |
| 22 | `MODULE_INDEX.md` "preserve four runtime services / not a service split" rule explicitly superseded | Mechanical |

Evidence for 15: Vercel AI SDK (11.5M weekly downloads, unified providers + streaming +
Zod structured output), `@anthropic-ai/claude-agent-sdk` (Claude Code's agent loop, MCP
client first-class, TS primary target), LanceDB native TS SDK, Playwright Node (upstream;
Python is the port), `@mozilla/readability` (reference impl), MCP TS reference SDK.
Gap (UMAP/HDBSCAN, no mature TS equivalent) dissolved by the facet-first grouping decision.

### Addendum 2026-07-24 (b) — Codex gpt-5.6-sol @ xhigh review of the stack decision

VERDICT: **rejects the full-TS strangler as written.** Corrections it forces:

- Shared-DB strangling is UNSAFE: `manager.py:61` loads SQLite into process-local dicts;
  `manager.py:227` persists by DELETE-all + reinsert from memory. Live Python silently
  erases concurrent TS writes. Service-by-service porting behind shared contracts fails.
- Phase 2 (vectors) before Phase 3 (content) is internally inconsistent — vectors keyed
  by identities that don't exist yet. **Decision #16 "build once in TS" is refuted.**
- `/index` (`ai-engine/main.py:321`) + vector ids (`rag.py:324`) carry session/URL identity;
  an HTTP-compatible port cannot preserve both old and new identities.
- Scrape callback (`routes.py:1043`) is unaudited overwrite keyed on session+URL only —
  no capture id / attempt / replay protection. Late Python callback clobbers newer TS capture.
- **"One process" deletes the credential/sandbox boundary that premise 4 requires** and that
  the reviewed topology deliberately retained. Agents + credentials cannot be UID-separated
  inside one process.
- Phase 1 removes existing agent hardening (`agent_cli.py:53,150,288,357,370`) that Agent-SDK
  typed events do not replace. Prompt-injection defenses (`rag.py:41`, `pipeline.py:9`) matter
  MORE when moving from tool-free completion to an interactive agent with corpus tools.
- UMAP/HDBSCAN "dissolved by facets" is rationalization: facets need a KNOWN axis and cannot
  discover themes; k-means fallback caps at 10 clusters at any corpus size (`pipeline.py:135`).
  Retain Python geometry as a sidecar pending real evaluation.
- Dual-stack becomes de facto permanent in ~8-12 weeks; 4-6 months full-time, 9-18 as a
  side project. A time box without a calendar date, funded deletion phase, and a ban on new
  dual-stack features is not a time box.
- Corrections: branch is **29 commits** (not 30). Robots is **behavior, not compliance** —
  ignores Allow precedence/wildcards/crawl-delay and **fails open** (`engine.py:322`).
- Workable shape if TS is mandatory: **edge facade + ATOMIC data-plane cutover** (not
  service-by-service strangler), capture retained as a **separate security process**.

### Addendum 2026-07-24 (c) — RESHAPED MIGRATION (D12, supersedes D11's method)

Destination unchanged: **full TypeScript**. Method and topology changed.

| # | Decision | Supersedes |
|---|---|---|
| 23 | **Edge facade + ONE atomic data-plane cutover** — not service-by-service strangler | #15 method |
| 24 | **Capture is a separate process with its own UID** (written in TS). One-process target abandoned; premise 4 needs a process boundary, and a process boundary is not a language boundary | #18 |
| 25 | **Python UMAP/HDBSCAN retained as a stateless compute sidecar** (vectors in, labels out, no shared state) pending a real clustering evaluation | #15's "gap dissolved" |
| 26 | Python and TypeScript **never** write the same SQLite concurrently — hard invariant | — |

**Revised order:**

1. **Freeze executable security invariants first** — black-box tests for URL/redirect/DNS
   pinning, local-only CDP, credential isolation, agent env/tool restrictions, token scopes,
   prompt injection. The TS implementation must pass the same suite before it ships.
2. **Fix in Python now**: fail-closed credentials (`auth/queue.py:44`), recursion depth guard
   (`pipeline.py:296`), and introduce a **single versioned idempotent ingest endpoint**
   (capture ID + attempt + replay protection) — capture stops writing vectors directly.
   This unblocks the cutover; without it the callback at `routes.py:1043` clobbers newer writes.
3. **TS UI + agent facade** behind a server-side gateway (origin policy, CSRF/local-user
   policy, scoped capability tokens). Agent access initially **read-only and allowlisted**;
   port the `agent_cli.py` hardening properties, do not assume the SDK provides them.
4. **ONE atomic cutover**: content + chunks + FTS + vectors together, with the new
   content-addressed schema. Retire Python Backend Core + AI Engine at that moment.
   Rehearsed migration + rollback plan required; this is a high-risk event, not a phase.
5. **Port capture last**, into its own UID-separated TS process, only after it passes the
   step-1 security suite. Credential capture stays a separate process permanently.

**Time box is a date, not an adjective**: set a calendar deadline, a funded deletion phase,
rollback criteria, and a ban on new features that depend on the dual-stack state. Without
all four, expect the split to become permanent in ~8-12 weeks.

### Addendum 2026-07-24 (d) — reconciled with merged PR #9 (ultraplan)

`docs/REPO_IMPROVEMENTS_REVIEW.md` was rewritten on `main` (PR #9, branch
`claude/improve-plan-xe72x6`, merge 09ab07f). It is grounded in **main**, so parts are
stale relative to this branch: session/URL SQLite persistence, scrape-dispatch error
surfacing (3f1dcd8), and downstream-error accounting are already done here.

New findings it contributes that this plan did NOT have:

| # | Finding | Disposition |
|---|---|---|
| 27 | **`validate_config()` exists (`config/config_loader.py:233`) and is never called** | Wire into each FastAPI lifespan **now, in Python** — free fail-fast |
| 28 | **Silent status synthesis**: on browser-engine 404, both `web-ui/src/api/client.py` and `routes.py` fabricate status from local counts, so an outage reads as valid `not_started`/`completed` | **Correctness bug — fix now in Python.** Same family as fail-open credentials |
| 29 | **No runtime logging, request IDs, or metrics anywhere** (one `print` repo-wide) | **ELEVATED from Medium to migration PRECONDITION** — an atomic data-plane cutover with no logs is not debuggable. Add structured logs + `X-Request-ID` before step 4 |
| 30 | Config divergences: `.env.example` `AI_PROVIDER=ollama` vs code/yaml/compose `openrouter`; `ai_models.yaml` ollama at `localhost:11434` vs compose `ollama:11434` | Cleanup workstream; do NOT port the divergence to TS. Same container-vs-host bug class as CDP `localhost:9222` |
| 31 | Dead `services/web-ui/.env.example` (`REACT_APP_API_URL`, `NODE_ENV`) never read by Streamlit | Delete — cleanup workstream |
| 32 | Hardcoded `http://browser-engine:8083` / `http://ai-engine:8090` in `routes.py` | Route through env-configured base URLs |
| 33 | `check_health()` collapses 3 services to booleans; can't distinguish down/4xx/5xx | Feeds UI diagnostics; carry into the TS facade |
| 34 | `smoke` pytest marker defined, no `make smoke-test` target | Trivial, add |
| 35 | CI has bandit+safety but no secret scanning (gitleaks/trufflehog) | Add to the step-1 security invariant suite |

**CONFLICT flagged:** PR #9's "Next Sprint" proposes building a persistent state layer in
Python. This plan builds the content-addressed schema **once, in TS, at cutover**. Scope
Python persistence work to **scrape-task state and credentials only** — do not build the
content/session store twice.

### Addendum 2026-07-24 (e) — six open questions resolved (Q1-Q6)

Claude and Fable agreed on all six; user confirmed each.

**Q1 / decision 36 — REAL-TAB RUN IS WORK ITEM ZERO.** Verified against the live DB
(`tab-organizer_backend-data`): 158 sessions, 170 URL records, **15 scraped** (12 are QA
fixtures), **`tab_import_jobs` = 0**, **`tab_search_fts` = 0 rows**. Two facts follow:
CDP tab attach has *never been exercised*, and the scrape→FTS write path is **broken in
production** — 15 successful scrapes produced zero searchable rows. Six review passes read
code; none opened the database. Run ~50 real tabs through attach→scrape→index→search→chat
and let that reorder the bug list before any other work. CI green means mocks pass
(`tests/e2e/test_workflow.py` scrapes example.com in 2.2s), not that the product works.

**Q2 / decision 37 — HARD GATE: `auth_used=true` ⇒ local embeddings.** Not a setting, an
ingest-time refusal: the ingest writer will not route authenticated-capture chunks to any
non-local embedding provider. Per-domain explicit opt-in is the only escape hatch. Requires
propagating an auth flag from capture to index (does not exist today — browser-engine
tracks only `auth_required` status). Costs little: Ollama fallback exists, hybrid keyword
leg covers un-embedded docs.

**Q3 / decision 38 — MERGE PR #8 TO MAIN NOW**, then branch migration work off merged main.
Security work must be on main for the step-1 invariant freeze; capture ports last so that
Python is the production data plane for months; a 99-file branch would rot against a
already-moved main.

**Q4 / decision 39 — GROUP IDENTITY = persistent entities matched by member overlap.**
Resolves the schema blocker. Rule: a new run's group adopts an existing `group_id` at member
**Jaccard overlap ≥ 0.5, greedy one-to-one** by score. On a split, the higher-overlap child
keeps identity, the other is new. An unmatched old group → `status=stale`, its relabel
override dormant (revived if a later run re-matches). `pin`/`exclude` are **page-level** rows
that reattach independently of group survival. Facet-derived groups get identity free via
`(axis, facet_value)`; the overlap rule only covers freeform geometry.

**Q5 / decision 40 — 12-WEEK BOX, TWO-WAY KILL-SWITCH.**
wk0 security-invariant suite frozen + PR #8 merged · wk1-6 TS facade/UI/agent · wk8 full
cutover rehearsal on a copy of real data (which the Q1 run provides) · wk10 atomic cutover ·
wk11-12 **funded deletion phase**, scheduled as explicit work items, ending with
`docker-compose.yml` removed. **Rollback**: rehearsal fails twice, or a post-cutover P0
(data loss, search/scrape broken) unfixed in 72h → restore pre-cutover SQLite/LanceDB
snapshot + Python stack. **Reverse kill-switch**: no cutover by **week 16** → migration is
cancelled and the TS work is deleted. **Zero new features on the dual stack from day 0**,
including Python-side ones.

**Q6 / decision 41 — B2B PLATFORM: DELETE BY OMISSION AT CUTOVER.** Do not port, do not
archive, do not spend effort pre-deleting from Python — it dies wholesale with the Python
stack. Only action now: **exclude its endpoints from the step-1 security invariant suite**
so tests aren't frozen for code scheduled to die. Evidence that would reverse this: a real
second user/company on the platform routes, or a stated commercialization roadmap.

**UNRESOLVED DECISIONS:**
- Framework: Next.js vs TanStack Start; better-sqlite3 vs libsql (non-blocking; defaults
  Next.js + better-sqlite3 unless overridden)
- Clustering evaluation not yet run (100/500/2000-page corpora; purity, coherence, useful
  singleton rate, stability across reruns) — until then the Python geometry sidecar stays

## WI0 addendum (2026-07-24)

Work item zero (decision 36) ran ~50 real URLs through the live Docker stack end to end.
Every break below was verified empirically against the running system and live SQLite, not
by code reading. Full detail: WI0 findings notes (not checked into this repo).

- **B1 (CRITICAL)** — Semantic pipeline has never worked on this deployment: default
  provider `openrouter` has no API key configured, and the Ollama fallback container had
  zero models pulled. LanceDB has never received a vector.
- **B2 (CRITICAL)** — CDP tab attach is architecturally impossible in the Docker deployment:
  Chrome binds its debug port to 127.0.0.1 and rejects non-IP/`localhost` Host headers, so
  the shipped `host.docker.internal` default can never connect. `tab_import_jobs = 0` is not
  disuse — it could never work.
- **B3 (HIGH)** — `POST /scrape` with inline `urls` never registers them via
  `add_urls_to_session`; every callback then fails "URL not found in session" and content
  vanishes while batch status still reads `completed`. Root cause of "15 scraped / 0 FTS
  rows" in production.
- **B4 (HIGH)** — Hybrid search 500s whenever the semantic leg fails, even though the
  keyword leg has results; one dead leg kills the whole default search mode.
- **B5 (MEDIUM)** — Batch indexing counters under-report total failure: a 100% indexing
  outage reads as a 1-in-50 blip instead of "nothing was indexed."
- **B6 (MEDIUM)** — Auth detection false-positives on public pages (npmjs.com, Anthropic
  docs) misread bot-challenges/403s as login walls.
- **B7 (MEDIUM)** — web-ui calls ai-engine `/chat` directly, bypassing backend-core and
  violating the "Backend Core is the only orchestrator" rule.
- **B8 (precondition)** — Zero application logging made every other break invisible; no log
  line beyond uvicorn access logs recorded 50 failed callbacks or the 100% index outage.
- **B9 (HIGH)** — Clustering 500s on small corpora: UMAP spectral init requires k < N and
  the pipeline never adapts `n_neighbors`/`n_components` to corpus size, so every
  early-stage session crashes `/cluster`.

Also confirmed: scraping itself is healthy (46/50 real-world success); the full loop
(scrape → index → hybrid search → chat) works end to end once providers are configured
correctly — the deployment *default*, not the product, has never been a working
configuration. See "Reordering consequence for wk0 tasks" in the WI0 notes for how these
folded into the phase-1 task list.

### Addendum 2026-08-05 — T6 retriggered; migration status after wk0

| # | Decision | Class |
|---|---|---|
| 42 | `SEC_BOOT_*_CMD` (T6) lands before the first TS commit touching credentials, tokens, or agent subprocesses — NOT before facade work begins | Revises the wk0 triage's date-box |
| 43 | wk1 runs the frozen suite against the TS facade in **attached mode**; no new harness needed | Mechanical |
| 44 | `sec_managed` probe inputs and expected refusals get extracted to language-neutral JSON fixtures while Python behaviour is verified | Mechanical (hedge) |

**Why 42 revises the earlier date-box.** The wk0 triage set T6 "before TS facade
work begins" without checking what the `sec_managed` probes actually cover.
(That count was recorded as 19 here and in `tests/security/README.md`; collection
reports **21** as of SECSUITE 1.5.0 — SEC-46/47 updated the prose and not the
arithmetic. Corrected 2026-08-07.)
They break down as: agent subprocess hardening (SEC-28..33, whole file) →
ai-engine providers; credential isolation and URL-safety-under-config →
browser-engine; prompt envelope → ai-engine RAG/clustering; plus token scopes
and CORS. The first four groups validate components that port at the **wk10
cutover or later** (capture ports last, decision 24). Building boot mode at wk1
would mean validating a TypeScript implementation that does not exist yet, using
a Python stack scheduled for deletion as the proving ground.

**Why 43 covers the wk1 gap.** Only CORS and token scopes bite from the first
facade commit, and neither needs harness-controlled env — they are plain HTTP
assertions. Attached mode already runs 192 of 213 probes against any
implementation. Point `SEC_BACKEND_URL` at the TS facade and they run.

**Why 44 exists.** The argument for deferring T6 is sound; the risk in deferring
is not forgetting but arriving at wk8 under cutover pressure, where the cheapest
path is weakening a probe to pass against what was built — which inverts the
purpose of freezing them. Extracting the contracts to fixtures now makes that
softening a visible diff instead of a quiet edit. It is the pattern SEC-25/42
already use for the agent env allowlist, and it worked.

**Standing counter-argument, recorded so it is not relitigated from scratch.**
"We are rewriting in TypeScript, so a documented invariant will be implemented
correctly" has a measured track record in this repo, and it is 0 for 3: CLAUDE.md
documented the token fallback as intended while the setup script made it a
permanent scope collapse; MODULE_INDEX documented the credential store as
"keyring or env key" while keyring was in no requirements file; premise 4
documented an API-layer backstop that did not exist. All three were written by
people who believed them, and all three were caught only by execution. The wk0
security rounds add six more instances: every first-attempt fix was refuted,
three by executed exploits, each having implemented the *example* in the finding
rather than its *class*. Documentation states the class; only a probe tests it.

**Migration status at this addendum:** wk0 complete and merged. No TypeScript
exists — no `package.json`, no `tsconfig.json`, no `.ts` file. wk1-6 (facade,
UI, agent layer) not started. The wk16 reverse kill-switch clock started when
wk0 landed.

### Addendum 2026-08-07 — G4 clustering evaluation: protocol and decision rule

This section is written **before the evaluation runs**, deliberately. Decision 25 keeps
the Python UMAP/HDBSCAN sidecar "pending a real clustering evaluation", and the failure
mode of an un-preregistered eval is that mixed numbers get read through whichever prior
the reader brought. The rule below is committed first; the numbers land under it.

| # | Decision | Class |
|---|---|---|
| 45 | The G4 verdict is decided by a **pre-registered rule with `drop` as the null hypothesis**: a tie is a drop | Mechanical |
| 46 | G4 measures **decision-39 group survival under increments**, not plain rerun-ARI | Mechanical |
| 47 | The eval corpus is **tab-shaped**, not vanilla 20 Newsgroups | Mechanical |
| 48 | Clustering tests that do not prove which algorithm ran are not evidence (see below) | Mechanical |

**Decision rule (pre-registered).** Keep the Python geometry sidecar only if, on the
tab-shaped corpus at BOTH 500 and 2000 docs, the production pipeline beats the best
TypeScript-candidate arm by **≥ 0.10 ARI or ≥ 10 points purity**, AND is **no worse on
decision-39 group survival**. Anything else — including a tie, including a win at 2000
only — is a DROP.

**Why the null hypothesis is `drop`.** The sidecar carries a standing cost the metrics
do not see: a whole Python process inside what is otherwise an `npx`/Tauri-distributable
TypeScript app, plus the decision-26 invariant that Python and TypeScript never write the
same SQLite. The product's stated corpora are hundreds of tabs, so a sidecar that only
earns its keep at 2000 does not earn its keep. The burden of proof is on keeping it.

**Why group survival, not rerun-ARI.** `pipeline.py` pins `random_state=42`, so rerunning
UMAP on identical input returns identical output and a plain "stability across reruns"
number is vacuously 1.0 — it would pass while measuring nothing, the could-not-fail class
this repo has now hit five times. The metric that decides the product question is
decision 39's own rule applied across increments: add 10 tabs, re-run, match groups at
Jaccard ≥ 0.5 greedy one-to-one, and report what fraction of groups keep their identity.
If geometry re-partitions the corpus every time ten tabs arrive, user relabels and pins do
not survive, and a quality edge is worth nothing.

**Why not vanilla 20 Newsgroups.** Balanced, equal-sized, single-topic classes are exactly
where k-means and agglomerative do well and where HDBSCAN's real advantages (unknown k,
variable density, skewed sizes, genuine noise) never get exercised — a `drop` verdict from
that corpus would be pre-baked, and a `keep` verdict could not arise. The corpus is
therefore shaped like real tabs: power-law group sizes, injected near-duplicates from one
domain, injected low-content navigational stubs, injected one-off outliers, and
`remove=('headers','footers','quotes')` so purity cannot be scored on hostname matching.

**Useful singleton rate, defined.** A singleton is not automatically a failure — a
genuinely unrelated tab SHOULD be alone. Against the injected one-offs, report singleton
**precision** (produced singletons that are real one-offs), singleton **recall** (one-offs
that ended up isolated), and **orphan rate** (docs whose true group had ≥ min_cluster_size
members present but which still landed in "Uncategorized"). These are computed from
HDBSCAN's `-1` labels, not from the API payload: `pipeline.py:226-233` turns every noise
point into its own cluster row, so the response cannot distinguish "12 groups" from
"3 groups + 9 orphans".

**Decision 48 — a clustering test that does not prove which algorithm ran is not evidence.**
`tests/requirements.txt` installs neither `umap-learn` nor `hdbscan` (both are in
`services/ai-engine/requirements.txt` only), and `pipeline.py:127`/`:154` catch `ImportError`
and fall back to SVD + `_kmeans_cluster` **silently**. Every clustering test in CI has
therefore been characterizing the fallback while appearing to test UMAP+HDBSCAN, and an
eval run in that image would have filed k-means numbers under the sidecar's name. The
degradation must be logged, and any run claiming to measure the geometry pipeline must
assert the real imports succeeded.
