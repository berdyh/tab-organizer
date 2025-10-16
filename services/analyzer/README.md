# Analyzer Service

The analyzer service generates dense embeddings, rich summaries, and quality metrics for upstream content. It exposes a FastAPI application that coordinates dedicated components for hardware detection, adaptive model selection, text chunking, embedding generation, Qdrant persistence, and Ollama-based LLM analysis.

```
┌────────────┐     ┌──────────────┐     ┌───────────┐
│ Web UI     │ --> │ API Gateway  │ --> │ Analyzer  │
└────────────┘     └──────────────┘     │  Service │
                                         │    │     │
                                         ▼    ▼     ▼
                                      Qdrant  Ollama  Prometheus/Logs
```

* The **web UI** triggers clustering workflows through the API gateway.
* The **API gateway** orchestrates service calls, putting analyzer behind a single ingress.
* The analyzer service writes chunk-level vectors to **Qdrant**, queries LLMs via **Ollama**, and records metrics for observability.

## Package Layout

```
services/analyzer/analyzer/
├── app.py              # FastAPI wiring, lifecycle, and routes
├── cache.py            # Embedding cache with disk + memory tiers
├── embeddings.py       # SentenceTransformer orchestration
├── hardware.py         # Hardware probing (CPU/GPU awareness)
├── logging.py          # Structlog configuration
├── model_management.py # Model metadata + recommendations
├── ollama_client.py    # Resilient Ollama integration
├── performance.py      # Model/resource metrics aggregation
├── qdrant_manager.py   # Collection management and vector IO
├── schemas.py          # Pydantic request/response models
├── state.py            # Mutable runtime component registry
├── tasks.py            # Background analysis and embedding jobs
└── text_processing.py  # Token- and char-based chunking
```

`main.py` is now a thin compatibility shim that re-exports the new package. Tests (and any legacy tooling) can still `import main`, but new code should import from `analyzer`.

## Setup

1. **Install dependencies**

   ```bash
   pip install -r services/analyzer/requirements.txt
   ```

   The requirements include heavy ML libraries (torch, sentence-transformers). Use a virtual environment to avoid polluting global packages.

2. **Configure models**

   - Default configuration is read from `/app/config/models.json`; for local runs you can place `config/models.json` two directories above the service.
   - The analyzer caches embeddings under `/app/cache/embeddings`. Ensure the directory is writable.

3. **Service environment variables**

   - `OLLAMA_BASE_URL` (optional): override the default `http://ollama:11434`.
   - `QDRANT_HOST` / `QDRANT_PORT` (optional): point to a remote Qdrant instance.

## Running the Service

```bash
uvicorn analyzer.app:app --reload --factory
```

The `--factory` flag ensures the FastAPI application is created through `create_app()` and the lifecycle hooks initialise all components.

## Testing

```bash
pip install -r services/analyzer/requirements.txt
python -m pytest services/analyzer
```

Key testing notes:

- Heavy dependencies are mocked in the test suite; when adding new modules, prefer injecting dependencies via the shared `AnalyzerState`.
- The conftest registers `sys.modules["main"]` so existing monkeypatch helpers continue to work.
- To simulate HTTP interactions, reuse `create_mock_httpx_client` or fixture-provided AsyncClient patches.
- When patching network politeness code (e.g. `urllib.robotparser.RobotFileParser`) do so **before** importing the analyzer package to ensure the patched symbol is used.

## Development Tips

- Use `analyzer.get_state()` to access the live component registry instead of relying on module-level globals.
- Prefer updating or extending individual modules (e.g. `text_processing.py`) rather than expanding `app.py`.
- When adding background jobs, ensure they accept the shared `AnalyzerState`; see `tasks.py` for patterns.
- Metrics are aggregated in-memory—trim history with `state.performance_monitor.max_history_size` if experimenting with large loads.
- Tests create deterministic `HttpUrl` instances using `_http()` (see `conftest.py`) so you can safely rely on Pydantic casting in new code.

For a full workflow walkthrough (scraping → clustering → UI), follow `scripts/run-analyzer-tests.sh` or inspect `services/web-ui` to understand how analyzer output feeds the UI clustering views.
