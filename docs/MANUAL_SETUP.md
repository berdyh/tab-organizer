# Manual Setup Guide

This guide explains how to run the Tab Organizer project without Docker, using Python directly on your system.

## Prerequisites

- Python 3.11+ 
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- PostgreSQL (optional, for persistent storage)
- Redis (optional, for caching)
- Node.js 18+ (only if you want to modify the web UI)

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

# Install development dependencies
uv pip install -r requirements-dev.txt
```

### 4. Setup External Services

#### Qdrant (Vector Database)

```bash
# Option A: Install locally
curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-linux-x86_64.tar.gz | tar xz
./qdrant/qdrant &

# Option B: Use Docker for just Qdrant
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest

# Option C: Use cloud service (e.g., Qdrant Cloud)
# Update .env with your cloud URL
```

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
- Qdrant Dashboard: http://localhost:6333/dashboard

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

# Install development dependencies
pip install -r requirements-dev.txt
```

### 3. Follow steps 4-6 from Option 1

## Configuration

### Environment Variables

Edit `.env` file:

```bash
# AI Provider Configuration
AI_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
LLM_MODEL=llama3.2:3b
EMBEDDING_MODEL=nomic-embed-text

# Service URLs (for manual setup)
BACKEND_URL=http://localhost:8080
AI_ENGINE_URL=http://localhost:8090
BROWSER_ENGINE_URL=http://localhost:8083
QDRANT_URL=http://localhost:6333
OLLAMA_URL=http://localhost:11434

# Database (optional, defaults to SQLite)
DATABASE_URL=sqlite:///./tab_organizer.db

# Redis (optional, for caching)
REDIS_URL=redis://localhost:6379
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

4. **Qdrant connection failed**
   - Ensure Qdrant is running on port 6333
   - Check firewall settings
   - Verify QDRANT_URL in .env

5. **Ollama models not found**
   ```bash
   ollama list
   ollama pull llama3.2:3b
   ```

### Performance Tips

1. **Use PostgreSQL for production**
   ```bash
   # Install PostgreSQL
   sudo apt-get install postgresql postgresql-contrib
   
   # Create database
   sudo -u postgres createdb tab_organizer
   
   # Update .env
   DATABASE_URL=postgresql://user:pass@localhost/tab_organizer
   ```

2. **Enable Redis for caching**
   ```bash
   # Install Redis
   sudo apt-get install redis-server
   
   # Start Redis
   sudo systemctl start redis
   
   # Update .env
   REDIS_URL=redis://localhost:6379
   ```

3. **Configure Ollama for better performance**
   ```bash
   # Set Ollama environment variables
   export OLLAMA_MAX_LOADED_MODELS=2
   export OLLAMA_NUM_PARALLEL=2
   export OLLAMA_MAX_QUEUE=512
   ```

## Production Deployment

For production deployment without Docker, consider:

1. **Use a process manager** (systemd, supervisor)
2. **Configure reverse proxy** (nginx, apache)
3. **Set up SSL certificates**
4. **Configure monitoring and logging**
5. **Use PostgreSQL instead of SQLite**
6. **Enable Redis for caching**

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
