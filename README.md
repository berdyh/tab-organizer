# Tab Organizer

A **local-first web scraping and tab organization tool** that helps you analyze, cluster, and manage browser tabs using AI. The system scrapes tab URLs, generates embeddings, clusters related content, and provides chatbot-style discovery.

## Features

- **URL Deduplication**: Set-like storage with automatic normalization and tracking parameter removal
- **Parallel Authentication**: Non-blocking scraping that continues for public sites while waiting for credentials
- **AI-Powered Clustering**: UMAP + HDBSCAN clustering with LLM-generated labels
- **Multi-Provider AI**: Support for Ollama, OpenAI, Anthropic Claude, DeepSeek, and Google Gemini
- **RAG Chatbot**: Query your scraped content using natural language
- **Export Options**: Markdown, JSON, HTML, Obsidian-compatible formats

## Architecture

```
┌─────────────────┐
│  Web UI         │ ← Streamlit (Python)
│  Port 8089      │
└────────┬────────┘
         │
┌────────▼────────┐     ┌──────────────┐
│  Backend Core   │────▶│ Qdrant       │
│  Port 8080      │     │ Port 6333    │
└────────┬────────┘     └──────────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌───▼────┐
│  AI   │ │Browser │
│Engine │ │Engine  │
│ 8090  │ │ 8083   │
└───────┘ └────────┘
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| **Web UI** | 8089 | Streamlit-based user interface |
| **Backend Core** | 8080 | API Gateway, session management, URL storage |
| **AI Engine** | 8090 | Embeddings, clustering, chatbot |
| **Browser Engine** | 8083 | Web scraping, auth detection |
| **Qdrant** | 6333 | Vector database for embeddings |
| **Ollama** | 11434 | Local LLM inference (optional) |

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for CLI)

### Installation

1. **Clone and initialize**:
   ```bash
   git clone <repository>
   cd tab-organizer
   cp .env.example .env
   ```

2. **Configure AI provider** (edit `.env`):
   ```bash
   # For local models (default)
   AI_PROVIDER=ollama
   EMBEDDING_PROVIDER=ollama

   # For cloud providers
   AI_PROVIDER=anthropic
   ANTHROPIC_API_KEY=sk-ant-...
   EMBEDDING_PROVIDER=openai
   OPENAI_API_KEY=sk-...
   ```

3. **Start services**:
   ```bash
   ./scripts/cli.py init --build --models
   ./scripts/cli.py start -d
   ```

4. **Open the UI**: http://localhost:8089

## Usage

### CLI Commands

```bash
# Start/Stop
./scripts/cli.py start -d          # Start in background
./scripts/cli.py start --build     # Rebuild and start
./scripts/cli.py stop              # Stop all services
./scripts/cli.py stop -v           # Stop and remove volumes

# Management
./scripts/cli.py status            # Show service status
./scripts/cli.py logs -f web-ui    # Follow logs
./scripts/cli.py restart           # Restart services

# Testing
./scripts/cli.py test --type unit         # Run unit tests
./scripts/cli.py test --type integration  # Run integration tests
./scripts/cli.py test --type e2e          # Run end-to-end tests

# Ollama Models
./scripts/cli.py models --list            # List installed models
./scripts/cli.py models --pull llama3.2   # Pull a model

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

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_PROVIDER` | `ollama` | LLM provider (ollama/openai/anthropic/deepseek/gemini) |
| `EMBEDDING_PROVIDER` | `ollama` | Embedding provider |
| `LLM_MODEL` | `llama3.2` | Model name for chat/analysis |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Model for embeddings |
| `EMBEDDING_DIMENSIONS` | `768` | Embedding vector size |
| `MAX_CONCURRENT_SCRAPES` | `10` | Parallel scraping limit |
| `SCRAPE_TIMEOUT` | `30` | Scrape timeout in seconds |
| `RESPECT_ROBOTS` | `true` | Honor robots.txt |

### API Keys

For cloud providers, set the appropriate API key:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEEPSEEK_API_KEY=...
GOOGLE_API_KEY=...
```

## Testing

```bash
# Run all tests
./scripts/cli.py test

# Run specific test types
./scripts/cli.py test --type unit
./scripts/cli.py test --type integration
./scripts/cli.py test --type e2e

# Run tests locally (without Docker)
cd tests
pip install -r requirements.txt
pytest unit/ -v
```

## Project Structure

```
tab-organizer/
├── services/
│   ├── backend-core/          # API Gateway & Session Management
│   │   └── app/
│   │       ├── api/           # FastAPI routes
│   │       ├── url_input/     # URL store & deduplication
│   │       ├── sessions/      # Session management
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
│   │       ├── scraper/       # Scraping engine
│   │       └── extraction/    # Content extraction
│   │
│   └── web-ui/                # Streamlit UI
│       └── src/
│           ├── api/           # API client
│           └── pages/         # UI pages
│
├── scripts/
│   └── cli.py                 # Management CLI
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

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/sessions` | POST | Create session |
| `/api/v1/sessions` | GET | List sessions |
| `/api/v1/sessions/{id}` | GET | Get session stats |
| `/api/v1/urls` | POST | Add URLs |
| `/api/v1/urls/{session_id}` | GET | Get URLs |
| `/api/v1/scrape` | POST | Start scraping |
| `/api/v1/cluster` | POST | Start clustering |
| `/api/v1/export` | POST | Export session |

### AI Engine (Port 8090)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/embed` | POST | Generate embeddings |
| `/cluster` | POST | Cluster URLs |
| `/chat` | POST | Chat with content |
| `/search` | POST | Search content |
| `/providers` | GET | Get provider info |
| `/providers/switch` | POST | Switch providers |

### Browser Engine (Port 8083)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/scrape` | POST | Start batch scraping |
| `/scrape/single` | POST | Scrape single URL |
| `/scrape/status/{session_id}` | GET | Get scrape status |
| `/auth/pending` | GET | Get pending auth |
| `/auth/credentials` | POST | Submit credentials |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `./scripts/cli.py test`
5. Submit a pull request
