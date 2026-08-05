# Tab Organizer

> Current direction: [docs/ARCHITECTURE_PLAN.md](docs/ARCHITECTURE_PLAN.md) — the reviewed, single-source-of-truth migration plan.

A **local-first web scraping and tab organization tool** that helps you analyze, cluster, and manage browser tabs using AI. The system scrapes tab URLs, generates embeddings, clusters related content, and provides chatbot-style discovery.

## Features

- **URL Deduplication**: Set-like storage with automatic normalization and tracking parameter removal
- **Live Browser Tab Import**: Attach to a local Chrome/Chromium CDP endpoint, import open tabs, extract readable content, and index it
- **Parallel Authentication**: Non-blocking scraping that continues for public sites while waiting for credentials
- **AI-Powered Clustering**: UMAP + HDBSCAN clustering with LLM-generated labels
- **Multi-Provider AI**: OpenRouter, Ollama, OpenAI, Anthropic Claude, Claude Code, Codex CLI/ACP, DeepSeek, and Google Gemini — always your explicit choice, never a default, and always announced
- **RAG Chatbot**: Query your scraped content using natural language (LanceDB native search)
- **Agent/CLI Access**: Protected CLI and MCP-oriented wrappers for importing, searching, clustering, opening, and exporting tabs
- **Export Options**: Markdown, JSON, HTML, Obsidian-compatible formats

## Architecture

```
┌─────────────────┐
│  Web UI         │ ← Streamlit (Python)
│  Port 8089      │
└────────┬────────┘
         │
┌────────▼────────┐
│  Backend Core   │
│  Port 8080      │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌───▼────┐
│  AI   │ │Browser │
│Engine │ │Engine  │
│ 8090  │ │ 8083   │
└───┬───┘ └────────┘
    │
┌───▼──────────────┐
│ LanceDB (embedded)│
│ volume: lancedb-data│
└──────────────────┘
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| **Web UI** | 8089 | Streamlit-based user interface |
| **Backend Core** | 8080 | Backend API, session management, URL storage |
| **AI Engine** | 8090 | Embeddings, clustering, chatbot (with embedded LanceDB) |
| **Browser Engine** | 8083 | Web scraping, auth detection |
| **Ollama** | 11434 | Local LLM inference (optional) |

The vector store is **LanceDB**, embedded inside the AI Engine container and persisted via the `lancedb-data` Docker volume — there is no separate vector-DB service.

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.12+ (for CLI)
- [uv](https://github.com/astral-sh/uv) (recommended for dependency management)

### Installation

1. **Clone and initialize**:
   ```bash
   git clone <repository>
   cd tab-organizer
   cp .env.example .env
   ```

2. **Configure AI provider** — there is no default, and nothing is chosen for
   you. Either run `./scripts/cli.py configure-provider`, or edit `.env`:
   ```bash
   # Subscription CLIs (preferred: spends a subscription you already pay for).
   # LLM-only, so pair with an embedding-capable provider.
   AI_PROVIDER=claude_code
   EMBEDDING_PROVIDER=ollama

   # OpenRouter (metered — one key for many models).
   # NOTE: OpenRouter serves NO embedding models, so EMBEDDING_PROVIDER must
   # point somewhere else.
   AI_PROVIDER=openrouter
   OPENROUTER_API_KEY=<openrouter-api-key>
   EMBEDDING_PROVIDER=ollama

   # Local-only via Ollama (opt-in; pull models first with `./scripts/cli.py init --models`)
   AI_PROVIDER=ollama
   EMBEDDING_PROVIDER=ollama

   # Mix and match
   AI_PROVIDER=anthropic
   ANTHROPIC_API_KEY=<anthropic-api-key>
   EMBEDDING_PROVIDER=openai
   OPENAI_API_KEY=sk-...

   # Local subscription LLMs from host CLI auth
   AI_PROVIDER=codex_acp
   EMBEDDING_PROVIDER=ollama
   CODEX_ACP_COMMAND=acpx
   ```

3. **Start services**:
   ```bash
   ./scripts/cli.py init --build --models
   ./scripts/cli.py start -d
   ```

   For local subscription CLI routing, run the AI Engine on the host and point
   the Docker services at it:
   ```bash
   ./scripts/cli.py host-ai --provider claude_code
   ./scripts/cli.py start -d --host-ai
   ```

4. **Open the UI**: http://localhost:8089

## Usage

### CLI Commands

```bash
# Start/Stop
./scripts/cli.py start -d          # Start in background
./scripts/cli.py start -d --host-ai # Route containers to host-run AI engine
./scripts/cli.py start --build     # Rebuild and start
./scripts/cli.py stop              # Stop all services
./scripts/cli.py stop -v           # Stop and remove volumes

# Local subscription LLM routing
./scripts/cli.py host-ai --provider claude_code
./scripts/cli.py host-ai --provider codex_cli
./scripts/cli.py host-ai --provider codex_acp
./scripts/cli.py check-provider --provider codex_acp --generate
./scripts/cli.py configure-provider   # probe real availability, write a verified provider choice to .env

# Management
./scripts/cli.py status            # Show service status
./scripts/cli.py logs -f web-ui    # Follow logs
./scripts/cli.py restart           # Restart services

# Testing
./scripts/cli.py test --type all          # Run all test suites
./scripts/cli.py test --type unit         # Run unit tests
./scripts/cli.py test --type integration  # Run integration tests
./scripts/cli.py test --type e2e          # Run end-to-end tests

# Browser tab management
./scripts/cli.py tabs import --cdp-url http://localhost:9222
./scripts/cli.py tabs status <job_id>
./scripts/cli.py tabs search "vector database notes" --mode hybrid --limit 10
./scripts/cli.py tabs cluster <session_id>
./scripts/cli.py tabs open --session-id <session_id>
./scripts/cli.py tabs export <session_id> --format markdown

# Ollama Models
./scripts/cli.py models --list            # List installed models
./scripts/cli.py models --pull llama3.2:3b   # Pull a model

# Cleanup
./scripts/cli.py clean             # Remove containers and volumes
./scripts/cli.py clean --images    # Also remove images
```

### Web UI Workflow

1. **Add URLs**: Paste URLs or upload a file on the URL Input page
2. **Scrape**: Start scraping on the Scraping page; handle auth requests as needed
3. **Cluster**: Generate AI-powered clusters on the Clusters page
4. **Chat**: Ask questions about your content on the Chatbot page
5. **Export**: Download organized tabs in your preferred format

### Live Browser Tab Workflow

Start Chrome or Chromium with a local debugging endpoint:

```bash
chromium --remote-debugging-port=9222
```

Then import and search the open tabs:

```bash
./scripts/cli.py start -d
./scripts/cli.py tabs import --cdp-url http://localhost:9222 --session-name "Live tabs"
./scripts/cli.py tabs status <job_id>
./scripts/cli.py tabs search "what was I reading about embeddings?"
```

The first implementation is attach-only: it connects to a user-started local
browser, never closes that browser/profile, and rejects non-local CDP endpoints.

## Configuration

### Environment Variables

| Variable | docker-compose default | `.env.example` default | Description |
|----------|-----------------------|------------------------|-------------|
| `AI_PROVIDER` | _(none)_ | _(none)_ | LLM provider (openrouter/ollama/openai/anthropic/claude_code/codex_cli/codex_acp/deepseek/gemini). No default anywhere — unset means the AI Engine reports `degraded` and names the fix rather than picking one |
| `EMBEDDING_PROVIDER` | _(none)_ | _(none)_ | Embedding provider (ollama/openai/gemini). No default and no fallback; an LLM-only provider here is an error, not a silent swap |
| `LLM_MODEL` | provider default | provider default | Model name for chat/analysis |
| `EMBEDDING_MODEL` | provider default | provider default | Model for embeddings |
| `EMBEDDING_DIMENSIONS` | model default | model default | Embedding vector size (must match the embedding model) |
| `AI_ENGINE_URL` | `http://ai-engine:8090` | — | URL backend/browser/web containers use for the AI Engine; set to `http://host.docker.internal:8090` for `--host-ai` |
| `AI_ENGINE_API_TOKEN` | — | — | Bearer token for protected AI Engine endpoints; generated locally by `start` and `host-ai` |
| `BROWSER_ENGINE_API_TOKEN` | — | — | Bearer token for Browser Engine control/auth endpoints; the ONLY token they accept (no cross-scope fallback); generated locally by `start` and `host-ai` |
| `BACKEND_CALLBACK_TOKEN` | — | — | Bearer token for browser-engine scrape callbacks into Backend Core; generated locally by `start` and `host-ai` |
| `BACKEND_AGENT_API_TOKEN` | — | — | Bearer token for local agent/CLI tab-management endpoints; generated locally by `start` |
| `AI_ENGINE_ALLOW_UNAUTHENTICATED` | `false` | `false` | Development escape hatch for direct AI Engine calls without a token |
| `BACKEND_DB_PATH` | `/data/backend/tab-organizer.sqlite3` | `./data/backend/tab-organizer.sqlite3` | SQLite database for sessions, URL records, callback metadata, clusters, and local platform data |
| `BACKEND_PUBLIC_URL` | `http://localhost:8080` | `http://localhost:8080` | Public Backend Core base URL used in generated B2B first-call examples |
| `PLATFORM_MAINTAINER_SIGNUP_CODE` | — | — | Local bootstrap code required when creating maintainer platform accounts |
| `VECTOR_DB_PATH` | `/data/lancedb` | — | LanceDB on-disk directory (mounted from `lancedb-data` volume) |
| `OLLAMA_HOST` | `http://ollama:11434` | `http://ollama:11434` | Ollama URL |
| `MAX_CONCURRENT_SCRAPES` | `10` | `10` | Parallel scraping limit |
| `SCRAPE_TIMEOUT` | `30` | `30` | Scrape timeout in seconds |
| `RESPECT_ROBOTS` | `true` | `true` | Honor robots.txt |
| `SCRAPE_ALLOW_PRIVATE_NETWORKS` | `false` | `false` | Opt-in escape hatch for scraping localhost/private-network targets |
| `CREDENTIAL_ENCRYPTION_KEY` | — | — | Fernet key for encrypted credential storage (browser-engine) |

`claude_code` and `codex_cli` are LLM-only providers that call the local `claude -p` or `codex exec` CLI using existing subscription login state. `codex_cli` is one-shot Codex CLI execution, not ACP mode, and is disabled by default for scraped-content prompts because `codex exec` is not a tool-free LLM-only mode. Use `codex_acp` when you want the app's LLM calls to go through an ACP Codex harness via `acpx`. ACP defaults to `deny-all` permissions for app-routed prompts; relax it only for trusted local experiments. Leave `LLM_MODEL` blank unless you need a provider-specific override, and keep `EMBEDDING_PROVIDER` on `ollama`, `openai`, or `gemini` — the only embedding-capable providers in the catalog. The stock Docker image does not install these CLIs, `acpx`, ACP adapters, or mount their auth state; use `./scripts/cli.py host-ai --provider claude_code` plus `./scripts/cli.py start -d --host-ai`, or build a custom image for Docker-based CLI/ACP routing.

AI Engine generation, embedding, indexing, chat, search, clustering, summarization, document deletion, and provider-switch endpoints fail closed unless `AI_ENGINE_API_TOKEN` is configured. Browser Engine scrape/auth control endpoints and Backend Core agent tab-management endpoints also require local service tokens. Use `./scripts/cli.py start`; for host-run AI, run `./scripts/cli.py host-ai` and `./scripts/cli.py start -d --host-ai` so the shared local token is generated and passed to the services.

If `EMBEDDING_PROVIDER` and `EMBEDDING_DIMENSIONS` are mismatched the AI Engine will refuse to write to the LanceDB table — keep them in sync.

### API Keys

For cloud providers, set the appropriate API key:

```bash
OPENROUTER_API_KEY=<openrouter-api-key>
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=<anthropic-api-key>
DEEPSEEK_API_KEY=...
GOOGLE_API_KEY=...
```

## Testing

```bash
# Run all tests
./scripts/cli.py test --type all

# Run specific test types
./scripts/cli.py test --type unit
./scripts/cli.py test --type integration
./scripts/cli.py test --type e2e

# Run tests locally (without Docker)
cd tests
uv pip install -r requirements.txt
pytest unit/ -v
```

## Development

### Dependency Management with uv

This project uses [uv](https://github.com/astral-sh/uv) for fast Python dependency management.

```bash
# Initialize
uv init

# Create venv
uv venv

# Install test dependencies
uv pip install -r tests/requirements.txt
```

## Project Structure

Development boundaries are tracked in [docs/MODULE_INDEX.md](docs/MODULE_INDEX.md)
and the local `MODULE.md` cards beside each service/submodule.

```
tab-organizer/
├── services/
│   ├── backend-core/          # Backend API & session management
│   │   └── app/
│   │       ├── api/           # FastAPI routes
│   │       ├── url_input/     # URL store & deduplication
│   │       ├── sessions/      # Session management
│   │       ├── platform/      # Local accounts, B2B tokens, companies
│   │       └── export/        # Export functionality
│   │
│   ├── ai-engine/             # AI Services
│   │   └── app/
│   │       ├── core/          # LLM client
│   │       ├── providers/     # Provider implementations
│   │       ├── clustering/    # Clustering pipeline
│   │       └── chatbot/       # RAG chatbot
│   │
│   ├── browser-engine/        # Web Scraping
│   │   └── app/
│   │       ├── auth/          # Auth detection & queue
│   │       ├── tabs/          # CDP tab import/open
│   │       ├── scraper/       # Scraping engine
│   │       └── extraction/    # Content extraction
│   │
│   └── web-ui/                # Streamlit UI
│       └── src/
│           ├── api/           # API client
│           └── pages/         # UI pages
│
├── scripts/
│   ├── cli.py                 # Management CLI
│   └── mcp/                   # Local MCP-oriented tab wrappers
│
├── tests/
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── e2e/                   # End-to-end tests
│
├── templates/                 # Export templates
├── docker-compose.yml
├── .env.example
└── README.md
```

## API Reference

### Backend Core (Port 8080)

The tab-management endpoints require bearer auth with `BACKEND_AGENT_API_TOKEN`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/sessions` | POST | Create session |
| `/api/v1/sessions` | GET | List sessions |
| `/api/v1/sessions/{session_id}` | GET | Get session stats |
| `/api/v1/sessions/{session_id}` | DELETE | Delete session |
| `/api/v1/urls` | POST | Add URLs |
| `/api/v1/urls/{session_id}` | GET | Get URLs |
| `/api/v1/scrape` | POST | Start scraping |
| `/api/v1/scrape/status/{session_id}` | GET | Scrape status (proxied from browser-engine) |
| `/api/v1/tabs/import` | POST | Start agent-protected browser tab import |
| `/api/v1/tabs/import/{job_id}` | GET | Get tab import job status |
| `/api/v1/tabs/open` | POST | Open URLs or a session in an attached browser |
| `/api/v1/search` | POST | Hybrid semantic/keyword search across indexed tabs |
| `/api/v1/cluster` | POST | Start clustering |
| `/api/v1/clusters/{session_id}` | GET | Get cluster results |
| `/api/v1/export` | POST | Export session |
| `/api/v1/auth/pending` | GET | List domains awaiting credentials |
| `/api/v1/auth/credentials` | POST | Submit credentials for a pending domain |
| `/api/v1/callback/scrape-complete` | POST | Internal callback used by browser-engine |
| `/api/v1/platform/auth/signup` | POST | Create local platform account |
| `/api/v1/platform/auth/login` | POST | Create platform session |
| `/api/v1/platform/me` | GET | Current platform user profile |
| `/api/v1/platform/companies/search` | GET | Authenticated company search |
| `/api/v1/platform/companies/{company_id}` | GET | Authenticated company detail |
| `/api/v1/platform/b2b/tokens` | GET/POST | List or create B2B API tokens |
| `/api/v1/platform/b2b/tokens/{token_id}` | DELETE | Revoke a B2B API token |
| `/api/v1/platform/b2b/first-call` | GET | B2B first API call guide |
| `/api/v1/platform/v1/companies/search` | GET | Token-authenticated public company API |
| `/api/v1/platform/dashboard` | GET | B2B dashboard counters and events |
| `/api/v1/platform/maintainer/issues` | GET | Maintainer issue visibility |
| `/api/v1/health` | GET | Health check |

### AI Engine (Port 8090)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/providers` | GET | Provider/model info |
| `/providers/switch` | POST | Hot-swap LLM/embedding provider |
| `/embed` | POST | Generate embeddings |
| `/generate` | POST | Generic LLM completion |
| `/cluster` | POST | Run UMAP + HDBSCAN + LLM labeling |
| `/index` | POST | Index documents into LanceDB with bounded chunks |
| `/chat` | POST | RAG chat over indexed content |
| `/search` | POST | Vector search over indexed content |
| `/summarize/{session_id}` | GET | Generate a session summary |
| `/documents/{session_id}` | DELETE | Drop indexed documents for a session |

### Browser Engine (Port 8083)

All Browser Engine endpoints below except `/health` require bearer auth using
`BROWSER_ENGINE_API_TOKEN`; no other service token is accepted, and an unset
value fails closed with 401 rather than opening the endpoints. Scrape
targets are limited to public `http`/`https` URLs unless
`SCRAPE_ALLOW_PRIVATE_NETWORKS=true` is explicitly set for local diagnostics.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/scrape` | POST | Start batch scraping |
| `/scrape/single` | POST | Scrape single URL |
| `/scrape/status/{session_id}` | GET | Get scrape status |
| `/tabs/import` | POST | Import tabs from a local CDP endpoint |
| `/tabs/open` | POST | Open URLs in a local CDP-attached browser |
| `/detect-auth` | POST | Probe a URL to detect auth requirements |
| `/auth/pending` | GET | List all domains awaiting credentials |
| `/auth/pending/{session_id}` | GET | Pending auth requests for one session |
| `/auth/pending/{domain}` | DELETE | Drop a pending auth request |
| `/auth/credentials` | POST | Submit credentials |
| `/auth/expire` | POST | Force-expire a stored credential |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `./scripts/cli.py test`
5. Submit a pull request
