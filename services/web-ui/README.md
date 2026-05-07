# Web UI Service

A [Streamlit](https://streamlit.io/) front end for Tab Organizer. It talks to
`backend-core` (sessions, URLs, scraping orchestration, export), `ai-engine`
(clustering, chat, search), and `browser-engine` (auth flow) over HTTP.

> Parent docs live in the repo root. See [../../README.md](../../README.md) and
> [../../docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md) for the wider
> context.

## Layout

```
services/web-ui/
├── app.py              # Streamlit entry point + sidebar navigation
├── Dockerfile          # Production image (Python 3.12 + Streamlit)
├── requirements.txt
├── scripts/            # Local helper scripts
│   ├── run_single_test.sh
│   ├── run_tests_debug.sh
│   ├── test_quick.sh
│   └── test-docker.sh
└── src/
    ├── api/
    │   └── client.py   # Thin HTTP client used by every page
    └── pages/
        ├── url_input.py
        ├── scraping.py
        ├── clustering.py
        ├── chatbot.py
        └── settings.py
```

Pages are plain Python modules that Streamlit re-runs top-to-bottom on every
interaction. Cross-service calls go through `src/api/client.py`, which reads the
service URLs from environment variables.

## Running locally

In Docker (recommended — matches CI):

```bash
./scripts/cli.py start -d           # full stack
# UI: http://localhost:8089
```

Or directly with Streamlit (requires the other services running somewhere):

```bash
cd services/web-ui
uv pip install -r requirements.txt
streamlit run app.py --server.port=8089 --server.address=0.0.0.0
```

## Configuration

The UI reads these environment variables:

| Variable | Default in `docker-compose.yml` | Purpose |
|----------|---------------------------------|---------|
| `BACKEND_URL` | `http://backend-core:8080` | Sessions, URLs, scraping orchestration, export |
| `AI_ENGINE_URL` | `http://ai-engine:8090` | Clustering, chat, search |
| `BROWSER_ENGINE_URL` | `http://browser-engine:8083` | Auth flow |

When running outside Docker, point them at `http://localhost:<port>` instead.

## Testing

The repo's unified test pipeline covers the UI:

```bash
./scripts/cli.py test --type unit         # fast, isolated
./scripts/cli.py test --type integration  # against running services
./scripts/cli.py test --type e2e          # full stack
```

Helper scripts under `services/web-ui/scripts/` wrap common loops while
iterating locally:

```bash
./scripts/test_quick.sh       # quick smoke loop
./scripts/test-docker.sh      # build + run inside Docker
./scripts/run_single_test.sh  # run a single named test
./scripts/run_tests_debug.sh  # verbose, with PDB on failure
```

## Adding a new page

1. Create `src/pages/my_page.py` exposing a `render()` function.
2. Import and add it to the sidebar selector in `app.py`.
3. Use `src/api/client.py` for any backend call rather than `requests` directly,
   so error handling and base URLs stay consistent.
