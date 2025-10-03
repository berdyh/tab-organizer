# Web Scraping & Clustering Tool

A comprehensive microservice-based system for web scraping, content analysis, and intelligent clustering using local AI models. The system operates entirely offline with persistent memory and supports multiple export formats.

## 🚀 Features

- **Web Content Scraping**: Scrapy-based scraping with authentication support
- **Local AI Processing**: Uses Ollama for LLM and embedding generation (no external APIs)
- **Intelligent Clustering**: UMAP + HDBSCAN for meaningful content grouping
- **Multi-format Export**: Notion, Obsidian, Word, and Markdown support
- **Session Management**: Persistent storage with incremental processing
- **Microservice Architecture**: Docker-based with health monitoring
- **Authentication Handling**: Automatic detection and popup-based login

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
- Docker and Docker Compose
- 8GB+ RAM (16GB recommended for larger models)
- 10GB+ free disk space

### Installation

1. **Clone and setup**:
```bash
git clone <repository-url>
cd web-scraping-clustering-tool
./scripts/setup.sh
```

2. **Configure environment**:
```bash
# Edit .env file with your settings
cp .env.example .env
nano .env
```

3. **Choose AI models** (Automatic or Manual):
```bash
# Fully automatic setup (recommended for most users)
./scripts/setup-models.sh auto
python3 scripts/model-manager.py auto

# Interactive setup with automatic/manual choice
./scripts/setup-models.sh
python3 scripts/model-manager.py interactive

# Quick hardware-optimized setup
./scripts/setup-models.sh recommended
```

4. **Start services**:
```bash
./scripts/start.sh
```

5. **Access the system**:
- API Gateway: http://localhost:8080
- Qdrant Dashboard: http://localhost:6333/dashboard
- Health Check: http://localhost:8080/health

## 📋 Available Scripts

```bash
# Setup and configuration
./scripts/setup.sh              # Initial setup
./scripts/setup-models.sh       # Interactive model selection
python3 scripts/model-manager.py interactive  # Advanced model setup

# Model information and management
./scripts/setup-models.sh list                # Show available models
./scripts/setup-models.sh auto                # Fully automatic setup
python3 scripts/model-manager.py hardware     # Show hardware info & recommendations
python3 scripts/model-manager.py auto         # Fully automatic setup
python3 scripts/model-manager.py categories   # Show model categories
python3 scripts/model-manager.py status       # Show installed & running models
python3 scripts/model-manager.py running      # Show only running models
python3 scripts/model-manager.py recommend --task reasoning  # Task-specific recommendations
python3 scripts/model-manager.py benchmark --model qwen3:1.7b  # Performance estimates
python3 scripts/model-manager.py list --category speed_optimized

# Service management
./scripts/start.sh               # Start all services
./scripts/stop.sh                # Stop all services
./scripts/logs.sh                # View all logs
./scripts/logs.sh <service>      # View specific service logs

# Model management
./scripts/setup-models.sh pull qwen3:1.7b   # Pull specific model
./scripts/setup-models.sh status            # Show installed & running models
./scripts/setup-models.sh all               # Pull all models (25GB+)
python3 scripts/model-manager.py uninstall --model qwen3:1.7b  # Uninstall model
```

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# AI Models
OLLAMA_MODEL=llama3.2:3b           # LLM model for summaries/clustering
OLLAMA_EMBEDDING_MODEL=nomic-embed-text  # Embedding model

# Scraping
SCRAPING_DELAY=1.0                 # Delay between requests
RESPECT_ROBOTS_TXT=true            # Follow robots.txt
MAX_PAGES_PER_DOMAIN=100           # Limit per domain

# Clustering
CLUSTERING_MIN_SIZE=5              # Minimum cluster size
UMAP_N_NEIGHBORS=15                # UMAP neighbors parameter

# Authentication
AUTH_MASTER_KEY=your-key-here      # Master key for credential encryption
```

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
# Check all services
curl http://localhost:8080/health

# Check specific service
curl http://localhost:8080/services/scraper
```

### Logs
```bash
# All services
./scripts/logs.sh

# Specific service
./scripts/logs.sh api-gateway
./scripts/logs.sh scraper
```

### Ollama Model Management
```bash
# Direct Ollama commands (inside container)
docker-compose exec ollama ollama ls        # List installed models
docker-compose exec ollama ollama ps        # List running models
docker-compose exec ollama ollama pull qwen3:1.7b  # Pull specific model
docker-compose exec ollama ollama rm qwen3:1.7b    # Remove model
docker-compose exec ollama ollama run qwen3:1.7b   # Run model interactively
```

### Metrics
Prometheus metrics available at `/metrics` endpoint.

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

### Common Issues

**Services not starting**:
```bash
# Check Docker status
docker-compose ps

# View service logs
./scripts/logs.sh <service-name>
```

**Models not downloading**:
```bash
# Check Ollama status
docker-compose exec ollama ollama list

# Manually pull model
docker-compose exec ollama ollama pull llama3.2:3b
```

**Out of memory**:
- Use smaller models (llama3.2:1b, gemma2:2b)
- Increase Docker memory limits
- Close other applications

**Slow performance**:
- Use faster models (llama3.2:3b)
- Enable GPU support if available
- Reduce batch sizes in configuration

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