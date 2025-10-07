# Web Scraping & Clustering Tool

A comprehensive microservice-based system for web scraping, content analysis, and intelligent clustering using local AI models. The system operates entirely offline with persistent memory and supports multiple export formats.

## 🚀 Features

- **🤖 AI Chatbot Interface**: Natural language queries to explore your scraped content
- **🌐 Web Content Scraping**: Scrapy-based scraping with authentication support
- **🧠 Local AI Processing**: Uses Ollama for LLM and embedding generation (no external APIs)
- **📊 Intelligent Clustering**: UMAP + HDBSCAN for meaningful content grouping
- **📤 Multi-format Export**: Notion, Obsidian, Word, and Markdown support
- **💾 Session Management**: Persistent storage with incremental processing
- **🏗️ Microservice Architecture**: Docker-based with health monitoring and API gateway
- **🔐 Authentication Handling**: Automatic detection and popup-based login
- **🎯 Smart Model Selection**: Hardware-aware AI model recommendations
- **🔍 Semantic Search**: Vector-based content search and similarity matching

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │────│  Qdrant Vector  │────│  Ollama LLM     │
│   (Port 8080)   │    │  DB (Port 6333) │    │  (Port 11434)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
    ┌────┴────┐
    │ Services │
    └─────────┘
         │
┌────────┼────────┐
│        │        │
▼        ▼        ▼
URL     Auth    Scraper
Input   Service  Service
│        │        │
▼        ▼        ▼
Analyzer Cluster Export
Service  Service Service
│        │        │
▼        ▼        ▼
Session Management
```

## 🤖 Supported AI Models (Latest 2024)

### 🚀 Latest LLM Models (Choose one)
- **qwen3:1.7b** - Ultra-efficient with thinking mode (~1.1GB) ⭐ **Low Resource**
- **phi4:3.8b** - Microsoft's latest SLM, 128K context (~2.3GB) ⭐ **GPU Optimized**
- **gemma3n:e4b** - Google multimodal: text/image/audio (~2.2GB) ⭐ **Balanced**
- **qwen3:4b** - Strong reasoning and coding (~2.5GB)
- **qwen3:8b** - Balanced performance, multilingual (~4.7GB)
- **qwen3:0.6b** - Ultra-lightweight for edge computing (~0.7GB)
- **gemma3n:e2b** - Edge-optimized multimodal (~1.2GB)

### 📚 Legacy Models (Still Good)
- **llama3.2:3b** - Fast, good quality (~2GB)
- **llama3.2:1b** - Fastest, basic quality (~1.3GB)
- **mistral:7b** - Good balance (~4.1GB)
- **codellama:7b** - Meta, code-focused (~3.8GB)

### 🔍 Embedding Models (Choose one)
- **nomic-embed-text** - Best general purpose (~274MB) ⭐ **Recommended**
- **all-minilm** - SentenceTransformers compatible (~90MB)
- **mxbai-embed-large** - Highest quality (~669MB)

### 🎯 Model Categories
- **Speed Optimized**: qwen3:0.6b, qwen3:1.7b, gemma3n:e2b
- **Quality Optimized**: qwen3:8b, phi4:3.8b, mistral:7b
- **Multimodal**: gemma3n:e2b, gemma3n:e4b (text, image, audio)
- **Multilingual**: qwen3 series (100+ languages), gemma3n series (140+ languages)
- **Code Focused**: qwen3:4b, phi4:3.8b, codellama:7b
- **Agent Capable**: qwen3:4b, qwen3:8b (tool use, reasoning)

## 🛠️ Quick Start

### Prerequisites
- **Docker** with Docker Compose V2 (built-in `docker compose` command)
- **8GB+ RAM** (16GB recommended for larger models)
- **10GB+ free disk space**
- **Python 3.7+** (for model management scripts)

### 🚀 One-Command Setup (Recommended)

```bash
git clone <repository-url>
cd web-scraping-clustering-tool
chmod +x scripts/*.sh
./scripts/quick-start.sh
```

This will automatically:
- Run initial setup
- Start all services
- Configure optimal AI models for your hardware
- Provide direct access links

### 📋 Manual Setup (Step by Step)

1. **Clone and initial setup**:
```bash
git clone <repository-url>
cd web-scraping-clustering-tool
chmod +x scripts/*.sh
./scripts/setup.sh
```

2. **Configure frontend environment**:
```bash
# Create frontend environment file
cp services/web-ui/.env.example services/web-ui/.env

# Verify the configuration (should show REACT_APP_API_URL=http://localhost:8080)
cat services/web-ui/.env
```

3. **Start services**:
```bash
./scripts/start.sh
```

4. **Setup optimal AI models** (after services are running):
```bash
# Automatic hardware-optimized setup (recommended)
python3 scripts/model-manager.py auto

# Or interactive setup with recommendations
python3 scripts/model-manager.py interactive
```

5. **Verify services are running**:
```bash
# Check all containers are running
docker compose ps

# Verify health endpoints
curl http://localhost:8080/health

# Check individual service health
curl http://localhost:8080/services
```

6. **Access the system**:
- **🌐 Web UI with Chatbot**: http://localhost:8089
- **🔗 API Gateway**: http://localhost:8080
- **📊 Qdrant Dashboard**: http://localhost:6333/dashboard
- **❤️ Health Check**: http://localhost:8080/health

### 🌐 Service URLs and Ports

| Service | Port | URL | Purpose |
|---------|------|-----|---------|
| **Web UI** | 8089 | http://localhost:8089 | React frontend interface |
| **API Gateway** | 8080 | http://localhost:8080 | Central API routing |
| **URL Input Service** | 8081 | http://localhost:8081 | URL validation and processing |
| **Scraper Service** | 8082 | http://localhost:8082 | Web content extraction |
| **Analyzer Service** | 8083 | http://localhost:8083 | AI analysis and embeddings |
| **Clustering Service** | 8084 | http://localhost:8084 | Content clustering |
| **Export Service** | 8085 | http://localhost:8085 | Multi-format export |
| **Session Service** | 8086 | http://localhost:8086 | Session management |
| **Auth Service** | 8087 | http://localhost:8087 | Authentication handling |
| **Monitoring Service** | 8088 | http://localhost:8088 | System monitoring |
| **Qdrant Vector DB** | 6333 | http://localhost:6333 | Vector database |
| **Ollama LLM** | 11434 | http://localhost:11434 | Local AI models |

**Important**: The frontend communicates with backend services through the API Gateway at port 8080. Direct access to individual service ports (8081-8088) is primarily for debugging and health checks.

### ✅ Verification Steps

After starting the services, verify everything is working correctly:

1. **Check Docker containers**:
```bash
# All services should show "Up" status
docker compose ps

# Expected services:
# - api-gateway (port 8080)
# - web-ui (port 8089)
# - ollama (port 11434)
# - qdrant (port 6333)
# - url-input-service (port 8081)
# - scraper-service (port 8082)
# - analyzer-service (port 8083)
# - clustering-service (port 8084)
# - export-service (port 8085)
# - session-service (port 8086)
# - auth-service (port 8087)
# - monitoring-service (port 8088)
```

2. **Test API Gateway connectivity**:
```bash
# Health check should return status "healthy"
curl http://localhost:8080/health | python3 -m json.tool

# Service registry should list all backend services
curl http://localhost:8080/services | python3 -m json.tool
```

3. **Test frontend connectivity**:
```bash
# Frontend should be accessible
curl -I http://localhost:8089

# Check if frontend can reach API gateway
# Open browser developer tools and check Network tab when using the UI
```

4. **Test end-to-end functionality**:
- Open http://localhost:8089 in your browser
- Create a new session
- Try uploading a URL or file
- Check browser console for any errors
- Verify API calls in browser Network tab show successful responses (200 status)

### 🎯 Using the System

1. **Open the Web UI**: Navigate to http://localhost:8089
2. **Create a Session**: Start a new scraping session
3. **Add URLs**: Upload URLs via file or paste directly
4. **Start Scraping**: Let the system extract and analyze content
5. **Explore with Chatbot**: Ask natural language questions like:
   - "Show me articles about AI"
   - "What are the main topics in my data?"
   - "Give me a summary of my content"
6. **Export Results**: Export to Notion, Obsidian, Word, or Markdown

## 📋 Available Scripts & Commands

### Setup & Configuration
```bash
# Initial project setup
./scripts/setup.sh                          # Setup Docker environment and .env

# Security key generation (required)
python3 -c "import secrets; print('AUTH_MASTER_KEY=' + secrets.token_urlsafe(32))"
python3 -c "import secrets; print('AUTH_ENCRYPTION_KEY=' + secrets.token_urlsafe(32))"
```

### Model Management (Python - Recommended)
```bash
# Automatic setup (recommended for most users)
python3 scripts/model-manager.py auto       # Fully automatic based on hardware
python3 scripts/model-manager.py interactive # Interactive with recommendations

# Information and status
python3 scripts/model-manager.py hardware   # Show hardware info & recommendations
python3 scripts/model-manager.py list       # Show all available models
python3 scripts/model-manager.py status     # Show installed & running models
python3 scripts/model-manager.py running    # Show only currently running models

# Task-specific recommendations
python3 scripts/model-manager.py recommend --task reasoning
python3 scripts/model-manager.py recommend --task code
python3 scripts/model-manager.py recommend --task multilingual

# Model operations
python3 scripts/model-manager.py pull --model qwen3:1.7b
python3 scripts/model-manager.py uninstall --model qwen3:1.7b
```

### Model Management (Shell Scripts - Alternative)
```bash
# Quick setup options
./scripts/setup-models.sh auto              # Automatic setup
./scripts/setup-models.sh interactive       # Interactive selection
./scripts/setup-models.sh list              # Show available models

# Model operations
./scripts/setup-models.sh pull qwen3:1.7b   # Pull specific model
./scripts/setup-models.sh status            # Show installed models
```

### Service Management
```bash
# Service lifecycle
./scripts/start.sh                          # Start all services
docker compose down                         # Stop all services

# Monitoring and logs
docker compose ps                           # Check service status
docker compose logs -f                     # View all logs (follow)
docker compose logs -f api-gateway         # View specific service logs
docker compose restart ollama              # Restart specific service

# Health checks
curl http://localhost:8080/health           # API health check
python3 scripts/model-manager.py status    # Model status check
```

### Development & Testing
```bash
# Development environment
make dev-up                                 # Start development environment
make dev-down                              # Stop development environment
make dev-logs                              # View development logs

# Testing
make test-all                              # Run all tests
make test SERVICE=api-gateway              # Test specific service
./scripts/validate-test-setup.sh           # Validate test environment
```

### Direct Docker Commands
```bash
# Direct Ollama model management
docker compose exec ollama ollama ls       # List installed models
docker compose exec ollama ollama ps       # List running models
docker compose exec ollama ollama pull gemma3n:e4b
docker compose exec ollama ollama rm gemma3n:e4b

# Container management
docker compose up -d                       # Start in background
docker compose down -v                     # Stop and remove volumes
docker compose pull                        # Pull latest images
docker compose build                       # Rebuild services
```

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Environment
ENVIRONMENT=development
LOG_LEVEL=INFO

# Service URLs (automatically configured for Docker)
QDRANT_URL=http://qdrant:6333
OLLAMA_URL=http://ollama:11434

# AI Models (updated automatically by model setup scripts)
OLLAMA_MODEL=gemma3n:e4b           # LLM model for summaries/clustering
OLLAMA_EMBEDDING_MODEL=nomic-embed-text  # Embedding model

# Authentication (REQUIRED - generate these keys)
AUTH_MASTER_KEY=<generate-with-python>      # Master key for credential encryption
AUTH_ENCRYPTION_KEY=<generate-with-python>  # 32-byte encryption key

# Scraping Configuration
SCRAPING_DELAY=1.0                 # Delay between requests (seconds)
RESPECT_ROBOTS_TXT=true            # Follow robots.txt rules
MAX_PAGES_PER_DOMAIN=100           # Limit pages per domain
USER_AGENT=WebScrapingClusteringTool/1.0

# Clustering Configuration
CLUSTERING_MIN_SIZE=5              # Minimum cluster size
CLUSTERING_MIN_SAMPLES=3           # Minimum samples for core points
UMAP_N_NEIGHBORS=15                # UMAP neighbors parameter
UMAP_MIN_DIST=0.1                  # UMAP minimum distance

# Qdrant Configuration (leave empty for local setup)
QDRANT_API_KEY=                    # Empty for local Docker setup
QDRANT_COLLECTION_SIZE=384         # Vector dimension size

# Export Configuration (optional)
NOTION_API_KEY=                    # For Notion export
NOTION_DATABASE_ID=                # Target Notion database

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
LOG_FORMAT=json
LOG_FILE_ENABLED=true
```

### Required Manual Configuration

**🔑 Security Keys (Required)**:
```bash
# Generate these keys and add to .env file
python3 -c "import secrets; print('AUTH_MASTER_KEY=' + secrets.token_urlsafe(32))"
python3 -c "import secrets; print('AUTH_ENCRYPTION_KEY=' + secrets.token_urlsafe(32))"
```

**🤖 AI Models (Automatic)**:
The model setup scripts will automatically update `OLLAMA_MODEL` and `OLLAMA_EMBEDDING_MODEL` in your `.env` file based on your hardware and preferences.

### Model Selection Guide

The system automatically detects your hardware and recommends optimal models, but here are manual guidelines:

**🚀 Automatic Selection (Recommended)**:
```bash
python3 scripts/model-manager.py auto
python3 scripts/model-manager.py hardware  # See what will be selected
```

**📋 Task-Specific Selection**:
```bash
python3 scripts/model-manager.py recommend --task reasoning
python3 scripts/model-manager.py recommend --task code
python3 scripts/model-manager.py recommend --task multilingual
```

**⚡ Performance-Based Selection**:
- **Speed Priority**: `qwen3:1.7b`, `gemma3n:e2b`, `qwen3:0.6b`
- **Quality Priority**: `qwen3:8b`, `phi4:3.8b`, `mistral:7b`
- **Balanced**: `gemma3n:e4b`, `qwen3:4b`, `llama3.2:3b`

**💻 Hardware-Based Selection**:
- **Low Resource (< 4GB RAM)**: `qwen3:1.7b` + `all-minilm`
- **Medium Resource (4-8GB RAM)**: `gemma3n:e4b` + `nomic-embed-text`
- **High Resource (8GB+ RAM)**: `qwen3:8b` + `mxbai-embed-large`
- **GPU Optimized (4GB+ VRAM)**: `phi4:3.8b` + `mxbai-embed-large`

**🎯 Use Case Selection**:
- **Web Scraping & Summarization**: `qwen3:4b`, `gemma3n:e4b`
- **Code Analysis**: `qwen3:4b`, `phi4:3.8b`, `codellama:7b`
- **Multilingual Content**: `qwen3:8b`, `gemma3n:e4b`
- **Multimodal (text + image/audio)**: `gemma3n:e4b`, `gemma3n:e2b`
- **Edge/Mobile Deployment**: `qwen3:0.6b`, `gemma3n:e2b`

## 📊 API Endpoints

### Core Workflow
```bash
# Upload URLs
POST /api/url-input/upload/csv

# Start scraping
POST /api/scraper/scrape

# Analyze content
POST /api/analyzer/analyze

# Cluster content
POST /api/clustering/cluster

# Export results
POST /api/export/notion
```

### Monitoring
```bash
# System health
GET /health

# Service status
GET /services

# Available models
GET /models

# Metrics
GET /metrics
```

## 🔐 Authentication Support

The system automatically detects websites requiring authentication and provides:

- **Automatic detection** of login requirements
- **Popup-based authentication** for manual login
- **Secure credential storage** with AES-256 encryption
- **Session persistence** across restarts
- **OAuth 2.0 support** for major providers
- **Parallel processing** of authenticated and public URLs

## 📤 Export Formats

- **Notion**: Structured database pages with metadata
- **Obsidian**: Markdown files with internal linking
- **Word**: Formatted documents with visualizations
- **Markdown**: Standard format with frontmatter

## 🔍 Monitoring & Debugging

### Health Monitoring
```bash
# Check all services status
curl http://localhost:8080/health | python3 -m json.tool

# Check Docker services
docker compose ps

# Check specific service health
curl http://localhost:8080/services/scraper
```

### Logs and Debugging
```bash
# View all service logs
docker compose logs -f

# View specific service logs
docker compose logs -f api-gateway
docker compose logs -f ollama
docker compose logs -f qdrant

# View logs with timestamps
docker compose logs -f --timestamps api-gateway
```

### Model Management & Status
```bash
# Check model status with Python manager
python3 scripts/model-manager.py status
python3 scripts/model-manager.py running
python3 scripts/model-manager.py hardware

# Direct Ollama commands (inside container)
docker compose exec ollama ollama ls        # List installed models
docker compose exec ollama ollama ps        # List running models
docker compose exec ollama ollama pull qwen3:1.7b  # Pull specific model
docker compose exec ollama ollama rm qwen3:1.7b    # Remove model
docker compose exec ollama ollama run qwen3:1.7b   # Interactive model chat

# Check Ollama service health
curl http://localhost:11434/api/tags
```

### Performance Monitoring
```bash
# System resource usage
docker stats

# Qdrant dashboard (visual interface)
open http://localhost:6333/dashboard

# Prometheus metrics (if enabled)
curl http://localhost:8080/metrics
```

## 🛠️ Development

### Service Structure
```
services/
├── api-gateway/     # Central orchestration
├── url-input/       # URL parsing and validation
├── auth/           # Authentication handling
├── scraper/        # Web content extraction
├── analyzer/       # AI analysis and embeddings
├── clustering/     # Content clustering
├── export/         # Multi-format export
└── session/        # Session management
```

### Adding New Models

1. Add model to `scripts/setup-models.sh`
2. Update `.env.example` with model information
3. Test with `./scripts/setup-models.sh pull <model-name>`

## 🐛 Troubleshooting

### Frontend-Backend Connectivity Issues

**❌ Frontend cannot connect to backend services**:
```bash
# 1. Verify .env file exists in web-ui directory
ls -la services/web-ui/.env

# If missing, create it:
cp services/web-ui/.env.example services/web-ui/.env

# 2. Check API URL configuration
cat services/web-ui/.env
# Should show: REACT_APP_API_URL=http://localhost:8080

# 3. Verify API Gateway is running and accessible
curl http://localhost:8080/health

# 4. Check if all services are registered with API Gateway
curl http://localhost:8080/services | python3 -m json.tool

# 5. Test direct service connectivity
curl http://localhost:8081/health  # URL Input Service
curl http://localhost:8082/health  # Scraper Service
curl http://localhost:8083/health  # Analyzer Service
```

**❌ "Network Error" or "Failed to fetch" in browser**:
```bash
# 1. Check browser console for specific error messages
# Open Developer Tools > Console tab

# 2. Check browser Network tab for failed requests
# Look for 404, 500, or CORS errors

# 3. Verify CORS configuration
curl -H "Origin: http://localhost:8089" \
     -H "Access-Control-Request-Method: POST" \
     -H "Access-Control-Request-Headers: Content-Type" \
     -X OPTIONS \
     http://localhost:8080/api/url-input-service/input/urls

# 4. Check if API Gateway proxy is working
curl -X POST http://localhost:8080/api/url-input-service/input/urls \
     -H "Content-Type: application/json" \
     -d '["https://example.com"]'
```

**❌ File upload fails with "400 Bad Request"**:
```bash
# 1. Check if the upload endpoint is accessible
curl -X POST http://localhost:8080/api/url-input-service/input/upload/text \
     -F "file=@test.txt"

# 2. Verify file format is supported
# Supported formats: text, json, csv, excel

# 3. Check service logs for detailed error
docker compose logs url-input-service

# 4. Test with a simple text file
echo -e "https://example.com\nhttps://google.com" > test_urls.txt
curl -X POST http://localhost:8080/api/url-input-service/input/upload/text \
     -F "file=@test_urls.txt"
```

**❌ Services show as "unhealthy" in health check**:
```bash
# 1. Check individual service status
docker compose ps

# 2. Check service logs for errors
docker compose logs api-gateway
docker compose logs url-input-service
docker compose logs web-ui

# 3. Restart unhealthy services
docker compose restart api-gateway
docker compose restart url-input-service

# 4. Check if services can communicate internally
docker compose exec api-gateway curl http://url-input-service:8081/health
```

### Common Issues & Solutions

**❌ "docker-compose: command not found"**:
```bash
# You need Docker Compose V2 (built into Docker)
docker --version          # Should be 20.10+ 
docker compose version    # Should work without hyphen

# If using old Docker, update to latest version
# Or install Docker Desktop which includes Compose V2
```

**❌ Services not starting**:
```bash
# Check Docker daemon is running
sudo systemctl status docker

# Check service status
docker compose ps

# View detailed service logs
docker compose logs api-gateway
docker compose logs ollama

# Restart specific service
docker compose restart ollama
```

**❌ Models not downloading**:
```bash
# Check Ollama service is running
docker compose ps ollama

# Check Ollama logs
docker compose logs ollama

# Manually pull model
docker compose exec ollama ollama pull gemma3n:e4b

# Check available disk space
df -h

# Check if model exists in registry
docker compose exec ollama ollama list
```

**❌ "AUTH_MASTER_KEY not set" errors**:
```bash
# Generate required authentication keys
python3 -c "import secrets; print('AUTH_MASTER_KEY=' + secrets.token_urlsafe(32))"
python3 -c "import secrets; print('AUTH_ENCRYPTION_KEY=' + secrets.token_urlsafe(32))"

# Add both keys to your .env file
nano .env
```

**❌ Out of memory errors**:
```bash
# Check available memory
free -h

# Use smaller models
python3 scripts/model-manager.py recommend --hardware-limit 4GB

# Recommended models for low memory:
# - qwen3:1.7b (~1.1GB)
# - all-minilm embedding (~90MB)

# Increase Docker memory limits (Docker Desktop)
# Settings > Resources > Memory > 8GB+
```

**❌ Slow performance**:
```bash
# Check system resources
docker stats

# Use faster models
python3 scripts/model-manager.py recommend --prioritize-speed

# Enable GPU support (if available)
nvidia-smi  # Check GPU availability
# GPU support is automatically detected in start.sh

# Optimize for your hardware
python3 scripts/model-manager.py hardware
```

**❌ Port conflicts**:
```bash
# Check what's using ports
sudo netstat -tulpn | grep :8080
sudo netstat -tulpn | grep :6333
sudo netstat -tulpn | grep :8089

# Stop conflicting services or change ports in docker-compose.yml
# Key ports used:
# - 8080: API Gateway
# - 8089: Web UI
# - 6333: Qdrant Vector DB
# - 11434: Ollama LLM
# - 8081-8088: Backend services
```

**❌ Permission errors**:
```bash
# Fix Docker permissions (Linux)
sudo usermod -aG docker $USER
newgrp docker

# Fix file permissions
sudo chown -R $USER:$USER .
chmod +x scripts/*.sh
```

**❌ API endpoints returning 404 errors**:
```bash
# 1. Verify API Gateway routing configuration
docker compose logs api-gateway | grep -i "routing\|proxy\|404"

# 2. Check if services are properly registered
curl http://localhost:8080/services | python3 -m json.tool

# 3. Test direct service endpoints (bypass API Gateway)
curl http://localhost:8081/api/input/urls  # Direct to URL Input Service

# 4. Verify correct API paths in frontend
# Frontend should call: /api/url-input-service/input/urls
# NOT: /api/url-input-service/api/input/urls (double /api/ prefix)
```

**❌ Environment variables not loading in frontend**:
```bash
# 1. Verify .env file exists and has correct content
cat services/web-ui/.env
# Should contain: REACT_APP_API_URL=http://localhost:8080

# 2. Rebuild frontend container to pick up new environment variables
docker compose build web-ui
docker compose up -d web-ui

# 3. Check if environment variables are available in browser
# Open browser console and type: console.log(process.env.REACT_APP_API_URL)
```

### Getting Help

**For Frontend-Backend Connectivity Issues**:
1. **Verify frontend environment**: `cat services/web-ui/.env`
2. **Check API Gateway health**: `curl http://localhost:8080/health`
3. **Test service registration**: `curl http://localhost:8080/services`
4. **Check browser console**: Open Developer Tools > Console for errors
5. **Verify API calls**: Check Network tab in browser Developer Tools
6. **Test direct API calls**: Use curl to test endpoints directly

**For General Issues**:
1. **Check logs first**: `docker compose logs -f`
2. **Verify prerequisites**: Docker Compose V2, sufficient RAM/disk
3. **Check service health**: `curl http://localhost:8080/health`
4. **Review configuration**: Ensure `.env` has required keys
5. **Test models**: `python3 scripts/model-manager.py status`
6. **Verify all containers running**: `docker compose ps`

## 📚 Documentation

- **[Architecture](docs/ARCHITECTURE.md)** - System design and technical architecture
- **[Requirements](docs/REQUIREMENTS.md)** - Detailed functional and non-functional requirements  
- **[Development Guide](docs/DEVELOPMENT.md)** - Development workflow, testing, and contribution guidelines

## 📝 License

[Add your license information here]

## 🤝 Contributing

See [Development Guide](docs/DEVELOPMENT.md) for detailed contribution guidelines.

## 📞 Support

[Add support information here]