# Manual Setup Guide

This guide explains how to run the Tab Organizer project without Docker, using Python directly on your system.

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

The web UI is a Streamlit (Python) app — no Node.js required. The default
storage layer is LanceDB on disk; PostgreSQL/Redis are not currently wired into
any service.

## Option 1: Using uv (Recommended)

### 1. Install uv

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### 2. Clone and Setup

```bash
git clone <repository-url>
cd tab-organizer
cp .env.example .env
```

### 3. Install Dependencies

```bash
# Create virtual environment (optional, uv manages one automatically)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install per-service runtime dependencies
uv pip install -r services/ai-engine/requirements.txt
uv pip install -r services/backend-core/requirements.txt
uv pip install -r services/browser-engine/requirements.txt
uv pip install -r services/web-ui/requirements.txt

# Install test dependencies
uv pip install -r tests/requirements.txt
```

### 4. Setup External Services

#### LanceDB (Vector Storage)

LanceDB is an embedded vector store — no separate service is needed. The AI Engine
opens a LanceDB directory on disk; pick any local path and point `VECTOR_DB_PATH`
at it (e.g. `./data/lancedb`). The directory is created on first use.

#### Ollama (Local LLM)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve &

# Pull models
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### 5. Start Services

Open multiple terminal windows:

```bash
# Terminal 1: Backend Core
cd services/backend-core
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080

# Terminal 2: AI Engine
cd services/ai-engine
uvicorn app.main:app --reload --host 0.0.0.0 --port 8090

# Terminal 3: Browser Engine
cd services/browser-engine
playwright install chromium
uvicorn app.main:app --reload --host 0.0.0.0 --port 8083

# Terminal 4: Web UI
cd services/web-ui
streamlit run app.py --server.port=8089 --server.address=0.0.0.0
```

### 6. Access the Application

- Web UI: http://localhost:8089
- Backend API: http://localhost:8080
- AI Engine: http://localhost:8090
- Browser Engine: http://localhost:8083
- LanceDB: embedded — inspect via the AI Engine's `/search` and `/providers` endpoints, or by opening the on-disk directory at `VECTOR_DB_PATH`

## Option 2: Using pip

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install per-service runtime + test dependencies
pip install -r services/ai-engine/requirements.txt
pip install -r services/backend-core/requirements.txt
pip install -r services/browser-engine/requirements.txt
pip install -r services/web-ui/requirements.txt
pip install -r tests/requirements.txt
```

### 3. Follow steps 4-6 from Option 1

## Configuration

### Environment Variables

Edit `.env` file:

```bash
# AI Provider Configuration
# Local: ollama. Cloud: openrouter, openai, anthropic, deepseek, gemini.
AI_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
LLM_MODEL=llama3.2:3b
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSIONS=768

# API keys (set whatever provider you picked above)
OPENROUTER_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
DEEPSEEK_API_KEY=
GOOGLE_API_KEY=

# Service URLs (for manual setup, when each service runs on localhost)
BACKEND_URL=http://localhost:8080
AI_ENGINE_URL=http://localhost:8090
BROWSER_ENGINE_URL=http://localhost:8083
OLLAMA_HOST=http://localhost:11434

# Embedded vector store (LanceDB on local disk)
VECTOR_DB_PATH=./data/lancedb
```

## Development Workflow

### Running Tests

```bash
# With uv
uv run pytest tests/unit/ -v
uv run pytest tests/integration/ -v
uv run pytest tests/e2e/ -v

# With pip
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/e2e/ -v
```

### Code Quality

```bash
# Format code
uv run black .
uv run isort .

# Lint code
uv run flake8 .
uv run pylint services/

# Type checking
uv run mypy services/

# Security check
uv run bandit -r services/
uv run safety check
```

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Find process using port
   lsof -i :8080
   
   # Kill process
   kill -9 <PID>
   ```

2. **Module not found**
   ```bash
   # Ensure you're in the correct directory
   cd services/backend-core
   
   # Install dependencies
   uv pip install -r requirements.txt
   ```

3. **Playwright browsers not installed**
   ```bash
   cd services/browser-engine
   playwright install chromium
   ```

4. **LanceDB errors at startup**
   - Ensure `VECTOR_DB_PATH` points to a writable directory
   - Check that the AI Engine process has read/write access to that path
   - Delete the directory and let the AI Engine recreate it if the index is corrupt

5. **Ollama models not found**
   ```bash
   ollama list
   ollama pull llama3.2:3b
   ```

6. **Backend sessions disappear after restart**
   - Set `BACKEND_DB_PATH` to a writable SQLite path, for example `./data/backend/tab-organizer.sqlite3`
   - Docker Compose sets this automatically on the `backend-data` volume
   - Set `PLATFORM_MAINTAINER_SIGNUP_CODE` before creating local maintainer accounts

### Performance Tips

1. **Configure Ollama for better performance**
   ```bash
   # Set Ollama environment variables
   export OLLAMA_MAX_LOADED_MODELS=2
   export OLLAMA_NUM_PARALLEL=2
   export OLLAMA_MAX_QUEUE=512
   ```

2. **Tune scraping concurrency** via `MAX_CONCURRENT_SCRAPES` and `SCRAPE_TIMEOUT` in `.env`.

## Production Deployment

For production deployment without Docker, consider:

1. **Use a process manager** (systemd, supervisor)
2. **Configure reverse proxy** (nginx, apache)
3. **Set up SSL certificates**
4. **Configure monitoring and logging**
5. **Back up `BACKEND_DB_PATH`** — the SQLite database holds sessions, URL records, callback metadata, clusters, and local platform data
6. **Back up `VECTOR_DB_PATH`** — the LanceDB directory holds all indexed content
7. **Set a strong `CREDENTIAL_ENCRYPTION_KEY`** — required by the browser-engine for encrypted credential storage

### Example systemd service for Backend Core

```ini
[Unit]
Description=Tab Organizer Backend Core
After=network.target

[Service]
Type=simple
User=tab-organizer
WorkingDirectory=/opt/tab-organizer/services/backend-core
Environment=PATH=/opt/tab-organizer/.venv/bin
ExecStart=/opt/tab-organizer/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8080
Restart=always

[Install]
WantedBy=multi-user.target
```

### Example nginx configuration

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8089;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /api {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Next Steps

- Read the [Development Guide](DEVELOPMENT.md) for contributing
- Check the [API Documentation](../README.md#api-reference) for integration
- Review the [Architecture Documentation](ARCHITECTURE.md) for understanding the system
