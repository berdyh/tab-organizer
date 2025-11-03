# Web Scraping & Clustering Tool

A local-first microservice platform for collecting web content, analysing it with on-device AI models, clustering related information, and exporting results for downstream tooling.

## Highlights

- End-to-end scraping pipeline with authentication handling, PDF/HTML extraction, and deduplication.
- Local AI inference via Ollama for embeddings, semantic search, clustering labels, and chatbot answers.
- Real-time monitoring and visualization served directly from the monitoring service.
- Modular Docker architecture with per-service unit/integration/e2e test coverage.

## Service Topology

| Port | Service | Purpose |
|------|---------|---------|
| 8080 | API Gateway | Central entrypoint, routing, rate limiting, auth guard |
| 8081 | URL Input | URL ingestion, validation, enrichment |
| 8082 | Scraper | Scrapy workers, auth-aware fetching, content extraction |
| 8083 | Analyzer | Embedding generation, model orchestration |
| 8084 | Clustering | UMAP/HDBSCAN workflows, cluster insights |
| 8085 | Export | Markdown, Notion, Word, Obsidian exports |
| 8086 | Session | Persistent collections, incremental updates |
| 8087 | Auth | Credential management, interactive login helpers |
| 8088 | Chatbot | Conversational interface backed by embeddings |
| 8089 | Web UI | React front-end |
| 8091 | Monitoring | Metrics, alerting, visualization endpoints |
| 6333 | Qdrant | Vector database |
| 11434 | Ollama | Local LLM & embedding models |

## Quick Start

```bash
git clone <repository-url>
cd web-scraping-clustering-tool

# One-time environment bootstrap (copies .env, selects provider/models, pulls images)
./scripts/init.py --provider ollama

# Start the full stack (auto-detects local vs Docker Ollama, builds if needed)
./scripts/cli.py start --pull --build
```

### Provider & model selection

- **Ollama (default):** `./scripts/init.py --provider ollama --ollama-mode auto`
  - `auto` picks a host Ollama if reachable, otherwise the Dockerised service and GPU/CPU profile automatically.
  - Large model downloads (2–5 GB for LLMs, 90–700 MB for embeddings) occur on first start—allow time and disk space.
- **Anthropic Claude:** `./scripts/init.py --provider claude --anthropic-key <token>`
  - Prompts for Claude LLM/embedding models and disables Ollama containers.
- Inspect curated recommendations any time with `./scripts/cli.py models`.

Front-end lives at `http://localhost:8089`, API gateway at `http://localhost:8080`, and monitoring dashboards at `http://localhost:8091/visualization/dashboard`.

### Web UI tips

- The Search page now scopes results to the session selected in the new session picker; cluster filters refresh automatically when you switch sessions.
- Export jobs validate destination requirements up front (e.g., Notion requires both an integration token and database ID) and surface success/errors inline.

## Useful Commands

```bash
./scripts/cli.py start            # Launch services (auto compose profiles)
./scripts/cli.py stop             # Stop containers (add --volumes to prune data)
./scripts/cli.py status           # Docker compose ps shortcut
./scripts/cli.py logs [service]   # Tail combined or per-service logs
./scripts/cli.py restart          # Stop + start with the same settings

# Useful HTTP probes
curl http://localhost:8080/health
curl http://localhost:8091/visualization/health
```

### Compose Profiles

The single `docker-compose.yml` file powers every workflow via profiles:

| Profile | Purpose | Example |
|---------|---------|---------|
| (default) | Production-style runtime stack | `./scripts/cli.py start` |
| `dev` | Hot-reload development services | `docker compose --profile dev up -d qdrant-dev api-gateway-dev` |
| `test-unit` | Per-service unit test runners | `./scripts/cli.py test --type unit` |
| `test-integration` | Integration harness with ephemeral infra | `./scripts/cli.py test --type integration` |
| `test-e2e` | End-to-end API/UI flow validation | `./scripts/cli.py test --type e2e` |
| `test-performance` | Locust load testing setup | `./scripts/cli.py test --type performance` |
| `test-report` | Coverage aggregation (invoked automatically) | `docker compose --profile test-report up test-report-aggregator` |

## Testing

Test runners operate inside Docker to mirror production images.

```bash
./scripts/cli.py test --type unit         # Unit suite (per service)
./scripts/cli.py test --type integration  # Integration suite
./scripts/cli.py test --type e2e          # End-to-end browser/API flows
./scripts/cli.py test --type performance  # Locust load tests
```

Add `--skip-cleanup` to leave test services running or `--skip-artifacts` during quick iteration. Pytest reports land in `./test-results`, coverage in `./coverage`, and aggregated HTML in `./test-reports`.

## Monitoring & Visualization

The monitoring service (`services/monitoring`) now hosts:

- `/visualization/health` – module status
- `/visualization/architecture/diagram` – system graph JSON
- `/visualization/pipeline/status` – pipeline statistics
- `/visualization/dashboard` – browser dashboard with Mermaid diagrams

These supersede the legacy standalone visualization container that previously lived under `services/visualization`.

## Troubleshooting

- **GPU detection issues:** Analyzer mocks CUDA calls for unit tests; ensure `torch.cuda.is_available()` is correctly mocked to a `bool` when writing tests.
- **Async monitoring tests skipped:** The monitoring image bundles `pytest-asyncio`; if you see skips, rebuild the test image (`./scripts/cli.py test --type unit`).
- **PyPDF2 deprecation warning:** The scraper still relies on PyPDF2 for backwards compatibility. Migration to `pypdf` is tracked separately; warnings are safe to ignore for now.

Feel free to adapt individual services, but keep ingress via the API Gateway so the monitoring and visualization layers retain a consistent view of the system.
