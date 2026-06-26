# Scraper Agentic Setup Analysis

## Overview
The scraper is implemented in `services/browser-engine` as an **orchestrated agentic workflow** with three collaborating components:

1. **Planner/Dispatcher** (`app/main.py`): receives scrape jobs, tracks per-session progress, and dispatches background execution.
2. **Executor** (`app/scraper/engine.py`): concurrently fetches pages, detects auth walls, and returns structured `ScrapeResult` objects.
3. **Human-in-the-loop auth agent** (`app/auth/queue.py` + `app/auth/detector.py`): pauses gated targets, requests credentials, and resumes when credentials are provided.

## End-to-End Flow

1. Backend sends `POST /scrape` to browser-engine.
2. Browser-engine initializes a progress state (`total/completed/success/failed/auth_required`) plus downstream callback/indexing counters.
3. `scrape_urls_background()` runs and calls `scraper.scrape_batch()`.
4. For each URL result, callback posts `callback/scrape-complete` to backend-core.
5. Backend updates URL status and metadata in session store.
6. Browser-engine batches successful pages to AI engine `/index` for RAG ingestion.

## Why this is “Agentic”

- **Autonomous task execution:** URLs are processed asynchronously and in parallel without user babysitting.
- **Stateful memory:** Session state and pending auth queue preserve intent across calls.
- **Conditional branching:** Auth detection changes behavior from direct scrape → credential request path.
- **Tool use:** The browser-engine uses HTTP endpoints as tools (backend callback, AI indexer).
- **Recovery behavior:** If auth is needed, work continues for other pages while waiting on user credentials.

## Auth Subsystem Design

### Detection (`AuthDetector`)
- Heuristic signal extraction from URL/page markers.
- Emits auth type (`basic`, `form`, `cookie`, `oauth`, etc.) and confidence.

### Queue + Store (`AuthQueue` + `CredentialStore`)
- Deduplicates requests per domain.
- Stores encrypted credentials with optional expiry.
- Supports callback registration to unblock waiting tasks.

This enables **parallel progress**: protected sites wait, public sites continue.

## Strengths

- Clean separation of concerns.
- Non-blocking background architecture.
- Explicit status telemetry (`/scrape/status/{session_id}`), including downstream backend callback and AI indexing failures.
- FastAPI-friendly integration with backend and AI services.

## Risks / Gaps

1. **In-memory state only**: progress/auth queue is lost on restart.
2. **No retry policy**: transient network failures are not retried with backoff.
3. **Limited observability**: no structured tracing/correlation IDs.
4. **Per-domain locking only**: no richer policy (e.g., captcha escalation, SSO handshake strategy).
5. **Single-pass indexing**: failed AI indexing is reported as a downstream error but is not retried automatically.

## Recommended Enhancements

1. Persist scrape task state + auth queue in Redis/SQLite.
2. Add retry budget with exponential backoff and jitter.
3. Add structured logs and request correlation IDs.
4. Add retry/dead-letter handling for failed callbacks/indexing after the visible downstream error path.
5. Add explicit scraping policies per domain (rate limits, browser-only domains, auth strategy).
6. Capture per-URL timeline events for UI debugging.

## Quick Verdict
The existing setup is a good **agentic MVP** (autonomous + conditional + HITL). Main improvements should focus on persistence, reliability, and observability.
