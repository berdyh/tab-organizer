# Scraper Service

The scraper service orchestrates URL classification, authenticated crawling, content extraction, and duplicate detection for the Tab Organizer platform. It exposes a FastAPI application that coordinates Scrapy-based spiders, a parallel processing engine, and integrations with the authentication service and API gateway.

---

## Architecture Overview

```mermaid
flowchart LR
    subgraph Clients
        UI[Web UI] -->|HTTP / WebSocket| APIGateway
    end

    subgraph Services
        APIGateway[API Gateway] -->|POST /scrape| ScraperAPI
        ScraperAPI[FastAPI App] -->|Classify URLs| Classifier
        ScraperAPI -->|Prepare Sessions| AuthClient
        ScraperAPI -->|Dispatch Jobs| ParallelEngine
        ParallelEngine -->|Spawn Crawls| ScrapyRunner
        ScrapyRunner[Scrapy ContentSpider] -->|Fetch Pages| TargetSite[(Target Domains)]
        ScrapyRunner -->|Push Results| StateStore[(In-Memory State)]
        ScrapyRunner -->|Check| DuplicateDetector
        ScrapyRunner -->|Extract| ContentExtractor
    end

    AuthClient -.->|REST| AuthService[(Authentication Service)]
    ParallelEngine -->|Metrics| WebsocketHub[WebSocket Clients]
```

- **FastAPI Application (`services.scraper.app`)**: Handles HTTP/WebSocket APIs, job state, and scheduling.
- **Parallel Processing Engine**: Maintains queues for public, authenticated, and retry URLs while coordinating async scraping tasks.
- **Scrapy `ContentSpider`**: Executes the actual fetches, integrates auth cookies/headers, and streams results back.
- **Content Pipeline**: Trafilatura and BeautifulSoup extract clean text; duplicate detection fingerprints content; robots compliance ensures respectful crawling.
- **Auth Integration**: `AuthenticationServiceClient` negotiates sessions with the auth service when protected endpoints are detected.

---

## Repository Layout

```
services/scraper/
├── app/                 # Service source code (FastAPI app, engines, utilities)
├── tests/               # Unit and integration tests
├── main.py              # Entry point & re-export layer for backwards compatibility
├── requirements.txt     # Service-specific dependencies
└── README.md            # You are here
```

Key modules:

- `app/api/routes.py` – FastAPI routes for job creation, metadata, stats, WebSocket updates.
- `app/processing.py` – Parallel processing engine and queue management.
- `app/spider.py` – Scrapy spider with auth injection, retry logic, and duplicate detection.
- `app/extraction.py` – Multi-strategy HTML/PDF/text extraction and quality scoring.
- `app/auth_client.py` – Client for the authentication microservice.

---

## Getting Started

### Prerequisites

- Python 3.12 (a `.venv` virtual environment is recommended).
- Service dependencies: `pip install -r services/scraper/requirements.txt`.
- Optional: Running auth/API gateway services for end-to-end testing (not required for unit tests).

### Running the Service Locally

```bash
cd /media/shoh/Shared/Shox/Projects/tab_organizer
source .venv/bin/activate
uvicorn services.scraper.main:app --reload --host 0.0.0.0 --port 8000
```

Useful endpoints once the server is running:

- `GET /health` – Service health & feature list.
- `POST /scrape` – Start a scraping job (expects `ScrapeRequest` payload).
- `GET /jobs/{job_id}` – Poll job state.
- `GET /jobs/{job_id}/urls` – Inspect per-URL metadata and classifications.
- `GET /stats` – Aggregate job and content metrics.
- `WS /ws/{job_id}` – Real-time job status streaming.

Configuration is provided through environment variables (via `pydantic-settings`). Key settings:

| Environment Variable        | Default                      | Purpose                                  |
|-----------------------------|------------------------------|------------------------------------------|
| `SCRAPER_AUTH_SERVICE_URL`  | `http://auth-service:8082`   | Base URL for the authentication service  |
| `SCRAPER_API_GATEWAY_URL`   | `http://api-gateway:8080`    | Optional API gateway endpoint            |
| `SCRAPER_USER_AGENT`        | `WebScrapingTool/1.0 ...`    | Default user agent for crawlers          |
| `SCRAPER_MAX_PARALLEL_WORKERS` | `5`                      | Worker pool size for parallel engine     |

---

## Testing

Run the full suite (unit + integration):

```bash
cd /media/shoh/Shared/Shox/Projects/tab_organizer
source .venv/bin/activate
python -m pytest services/scraper/tests
```

Targeted test runs:

- Unit tests only: `python -m pytest services/scraper/tests/unit`
- Integration tests only: `python -m pytest services/scraper/tests/integration`
- Verbose output: append `-vv`
- Coverage example: `python -m pytest --cov=services.scraper.app services/scraper/tests`

---

## Development Tips

- **State Awareness**: In production, replace the in-memory state (`app/state.py`) with Redis or a database for persistence and horizontal scaling.
- **Auth Simulation**: Integration tests mock the auth service—when interacting with the real service ensure the authentication microservice is running.
- **Extensibility**: New content extraction strategies can be added to `ContentExtractor.extraction_methods`; duplicate detection thresholds are configurable via `DuplicateDetector`.
- **Backwards Compatibility**: `services.scraper.main` re-exports legacy names so older tests & scripts that import from `main` continue to function.

Feel free to expand this README with deployment specifics or service-level observability notes as the scraper evolves. Happy crawling!
