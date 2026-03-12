# Repository Improvement Review (Post-pass)

## High Priority

1. **Configuration consistency**
   - Standardize environment variable names across services.
   - Validate startup config and fail fast with actionable errors.

2. **Persistence gaps**
   - Session state and scraping task state are currently memory-backed.
   - Move session/task/auth states to persistent storage.

3. **Contract-driven APIs**
   - Add shared OpenAPI/JSON schema checks between UI, backend-core, browser-engine, and ai-engine.
   - Prevent drift in fields like scrape status payloads.

4. **Error propagation**
   - Several cross-service calls swallow exceptions.
   - Return structured error envelopes and surface them in UI.

## Medium Priority

1. **Test coverage expansion**
   - Add integration tests for: provider switching, scrape status proxy, auth credential flow, and RAG indexing/search lifecycle.

2. **Observability**
   - Add structured logging and request IDs for all internal HTTP calls.
   - Add latency metrics and scrape success/error counters.

3. **Security hardening**
   - Credential encryption key management should enforce non-default key in production.
   - Add secret scanning and `.env` guidance.

4. **UI reliability**
   - Add better user-facing diagnostics when backend services are unavailable.
   - Add retry hints and partial availability indicators.

## Low Priority

1. **Documentation freshness**
   - Several docs still mention Qdrant; ensure all architecture/setup docs reflect LanceDB default.

2. **Developer experience**
   - Add `make dev-up` and `make smoke-test` commands for a quick local validation loop.

3. **Code organization**
   - Consider consolidating API client logic and adding typed response models for Streamlit pages.

## Suggested Next Sprint Scope

- Persistent state layer (sessions + scrape tasks)
- Observability baseline (structured logs + metrics)
- Integration test suite for service-to-service contracts
- Startup health preflight checks for provider and vector DB config
