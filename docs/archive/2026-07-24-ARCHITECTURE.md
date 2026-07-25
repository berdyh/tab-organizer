> ARCHIVED 2026-07-24 — superseded by the reviewed architecture plan (see docs/ARCHITECTURE_PLAN.md). Retained for reference.

# Architecture Documentation

For development boundaries, load [MODULE_INDEX.md](MODULE_INDEX.md) first, then
the local `MODULE.md` card for the affected service/submodule. This document
describes runtime architecture and cross-service contracts.

## Table of Contents
1. [System Overview](#system-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Core Services](#core-services)
4. [Data Flow](#data-flow)
5. [Technology Stack](#technology-stack)
6. [Container Architecture](#container-architecture)
7. [API Specification](#api-specification)
8. [Security Architecture](#security-architecture)
9. [Performance & Scalability](#performance--scalability)
10. [Deployment Architecture](#deployment-architecture)

## System Overview

The Tab Organizer is a microservice-based system that processes web content through a pipeline of specialized services. The architecture emphasizes local AI processing, privacy, and scalability while maintaining simplicity in deployment through Docker containerization. The system operates with optional offline capability and supports multiple export formats including Markdown, JSON, HTML, and Obsidian.

### Key Architectural Principles
- **Microservice Architecture**: 4 core services with clear separation of concerns
- **Multi-Provider AI**: Support for local (Ollama) and cloud AI providers (OpenRouter, OpenAI, Anthropic, DeepSeek, Gemini). OpenRouter is the docker-compose default; Ollama is the `.env.example` default for fully local runs.
- **Parallel Processing**: Non-blocking authentication and scraping workflows
- **Container-First Design**: All development, testing, and deployment in Docker
- **Event-Driven Communication**: Asynchronous processing with proper orchestration

## High-Level Architecture

```mermaid
graph TB
    Agent[Agent CLI / MCP wrappers] --> Backend
    Client[Client/Browser] --> UI[Web UI :8089]
    
    UI --> Backend[Backend Core :8080]
    
    Backend --> AI[AI Engine :8090]
    Backend --> Browser[Browser Engine :8083]
    Browser --> Chrome[Local Chrome / Chromium CDP :9222]
    
    AI --> LanceDB[(LanceDB - embedded vector store)]
    AI --> Ollama[Ollama LLM :11434]
    Browser --> Backend
    
    LanceDB --> Storage1[Vector Storage volume: lancedb-data]
    Ollama --> Storage2[Model Storage]
    
    subgraph "Input Sources"
        TextFile[Plain Text Files]
        JSONFile[JSON Files]
        CSVFile[CSV Files]
        WebForm[Web Form Input]
    end
    
    UI --> TextFile
    UI --> JSONFile
    UI --> CSVFile
    UI --> WebForm
```

### Data Pipeline Flow

```mermaid
flowchart TD
    A[URL Input] --> B{Authentication Required?}
    B -->|No| C[Direct Scraping Queue]
    B -->|Yes| D[Authentication Queue]
    
    C --> E[HTTP Scraping]
    D --> F[Auth Detection & Credential Request]
    F --> G[Authenticated Scraping]
    
    E --> H[Content Extraction]
    G --> H
    
    H --> I[Embedding Generation]
    I --> J[Vector Storage - LanceDB]
    
    J --> K[UMAP Dimensionality Reduction]
    K --> L[HDBSCAN Clustering]
    L --> M[LLM Cluster Labeling]
    
    M --> N[Export Processing]
    
    N --> O[Markdown Export]
    N --> P[JSON Export]
    N --> Q[HTML Export]
    N --> R[Obsidian Export]
```

### Agent Tab Management Flow

```mermaid
sequenceDiagram
    participant Agent as Agent CLI / MCP wrapper
    participant BE as Backend Core
    participant BR as Browser Engine
    participant CH as Local Chrome CDP
    participant AI as AI Engine
    participant Q as LanceDB / SQLite FTS

    Agent->>BE: POST /api/v1/tabs/import
    BE->>BE: Create tab_import_job
    BE->>BR: POST /tabs/import
    BR->>CH: Attach over local CDP
    BR->>BR: Extract readable tab content
    BE->>BE: Store URL records + FTS rows
    BE->>AI: POST /index
    AI->>Q: Store bounded content chunks
    Agent->>BE: POST /api/v1/search
    BE->>Q: Merge vector + SQLite FTS results
    Agent->>BE: POST /api/v1/tabs/open
    BE->>BR: POST /tabs/open
    BR->>CH: Open selected URLs
```

This path is the primary backend tool goal: agents can import thousands of open
tabs, index the content, search across semantic and keyword stores, cluster a
session, export it, and reopen selected tabs. Browser control is attach-only in
v1 and requires a user-started local Chrome/Chromium debugging endpoint.

### Parallel Processing Architecture

```mermaid
graph TB
    subgraph "URL Processing Pipeline"
        URLQueue[URL Queue] --> URLClassifier{Auth Detector}
        URLClassifier -->|Public URLs| PublicQueue[Public Scraping Queue]
        URLClassifier -->|Auth Required| AuthQueue[Authentication Queue]
        
        PublicQueue --> ScraperPool[Parallel Scraper Workers]
        AuthQueue --> AuthHandler[Auth Credential Handler]
        AuthHandler -->|Credentials Ready| AuthScraper[Authenticated Scraper]
        
        ScraperPool --> ContentProcessor[Content Processing]
        AuthScraper --> ContentProcessor
    end
    
    subgraph "AI Processing"
        ContentProcessor --> EmbeddingService[Embedding Generation]
        EmbeddingService --> ClusteringService[Clustering Pipeline]
        ClusteringService --> LLMService[LLM Labeling]
    end
```

### Service Communication Diagram

```mermaid
sequenceDiagram
    participant UI as Web UI
    participant BE as Backend Core
    participant BR as Browser Engine
    participant AI as AI Engine
    participant Q as LanceDB (embedded)
    participant O as Ollama LLM

    UI->>BE: Submit URLs for processing
    BE->>BE: Deduplicate and store URLs
    BE->>BR: Request scraping
    
    par Parallel Processing
        BR->>BR: Scrape public URLs
    and
        BR->>UI: Request credentials for auth URLs
        UI->>BR: Provide credentials
        BR->>BR: Scrape authenticated URLs
    end
    
    BR->>BE: Send extracted content
    BE->>AI: Request embeddings
    AI->>O: Generate embeddings
    O-->>AI: Return embeddings
    AI->>Q: Store embeddings
    
    BE->>AI: Request clustering
    AI->>Q: Retrieve embeddings
    AI->>AI: UMAP + HDBSCAN clustering
    AI->>O: Generate cluster labels
    O-->>AI: Return labels
    AI->>BE: Return clusters
    
    BE->>UI: Processing complete
```

**Single-writer ingest.** Browser Engine sends each scrape result to Backend
Core's `POST /api/v1/ingest/v1` (never to AI Engine): a client-generated
`capture_id` gives replay protection and `(fetched_at, attempt)` gives
newest-wins ordering, so a late or duplicated delivery is acknowledged and
ignored rather than clobbering a newer capture. Backend Core applies the result
(ledger row + URL record + FTS row in one transaction) and is the **only**
writer that forwards content to AI Engine `/index`; capture never writes vectors
directly. AI-index success/failure is recorded in the `ingest_captures` ledger
and surfaced through `GET /api/v1/scrape/status/{id}`.

## Core Services

### 1. Backend Core (Port 8080)
**Purpose**: Backend API orchestration, session management, and URL storage

**Responsibilities**:
- Session lifecycle management
- URL deduplication and storage
- Agent-protected tab import/open/search APIs
- Durable tab import job status
- SQLite FTS metadata used with AI vector search
- Export functionality (Markdown, JSON, HTML, Obsidian)
- Orchestration of scraping and clustering workflows
- Health monitoring

**Key Components**:
- `app/api/routes.py` - FastAPI routes
- `app/url_input/store.py` - URL storage with deduplication
- `app/url_input/dedup.py` - URL normalization and deduplication
- `app/sessions/manager.py` - Session management
- `app/export/exporter.py` - Multi-format export

**Technology**: FastAPI, Python 3.12

### 2. AI Engine (Port 8090)
**Purpose**: AI services for embeddings, clustering, and chatbot

**Responsibilities**:
- Multi-provider LLM support (OpenRouter, Ollama, OpenAI, Anthropic, DeepSeek, Gemini)
- Embedding generation with configurable models
- UMAP + HDBSCAN clustering pipeline
- RAG-based chatbot with bounded chunk indexing and LanceDB vector search
- Dynamic provider switching

**Key Components**:
- `app/core/llm_client.py` - Unified LLM client
- `app/providers/` - Provider implementations
- `app/clustering/pipeline.py` - Clustering pipeline
- `app/chatbot/rag.py` - RAG chatbot

**Technology**: FastAPI, UMAP, HDBSCAN, LanceDB (embedded), Python 3.12

### 3. Browser Engine (Port 8083)
**Purpose**: Web scraping and authentication handling

**Responsibilities**:
- HTTP-based content scraping
- Local CDP attach for live tab inventory, readable extraction, and tab opening
- Authentication detection and credential management
- Parallel processing of public and authenticated URLs
- Content extraction and cleaning
- Robots.txt compliance

**Key Components**:
- `app/tabs/cdp.py` - Local CDP tab harvester/open helper
- `app/scraper/engine.py` - Scraping engine
- `app/auth/detector.py` - Authentication detection
- `app/auth/queue.py` - Authentication queue management
- `app/extraction/` - Content extraction utilities

**Technology**: FastAPI, httpx, BeautifulSoup, trafilatura, Python 3.12

### 4. Web UI (Port 8089)
**Purpose**: Streamlit-based user interface

**Responsibilities**:
- URL input and management
- Scraping progress monitoring
- Clustering visualization
- RAG chatbot interface
- Export and settings management

**Key Components**:
- `src/pages/url_input.py` - URL input page
- `src/pages/scraping.py` - Scraping page
- `src/pages/clustering.py` - Clustering page
- `src/pages/chatbot.py` - Chatbot page
- `src/pages/settings.py` - Settings page
- `src/api/client.py` - Backend API client

**Technology**: Streamlit, Python 3.12

## Container Network Architecture

```mermaid
graph TB
    subgraph "Docker Host"
        subgraph "Application Services"
            UI[Web UI :8089]
            Backend[Backend Core :8080]
            AI[AI Engine :8090]
            Browser[Browser Engine :8083]
        end
        
        subgraph "Infrastructure Services"
            OLLAMA[Ollama :11434]
        end
        
        subgraph "Storage Volumes"
            V1[lancedb-data - mounted into ai-engine]
            V2[ollama-data]
        end
    end
    
    UI --> Backend
    Backend --> AI
    Backend --> Browser
    Browser --> Backend
    
    AI --> OLLAMA
    AI --> V1
    OLLAMA --> V2
```

## Data Flow

### Primary Workflow
1. **Input**: URLs added via web UI (paste, upload file)
2. **Deduplication**: URL normalization and duplicate detection
3. **Authentication**: Automatic detection and credential request
4. **Scraping**: Parallel content extraction (public + authenticated)
5. **Analysis**: AI-powered embedding generation
6. **Clustering**: UMAP + HDBSCAN with LLM-generated labels
7. **Export**: Multi-format output (Markdown, JSON, HTML, Obsidian)

### Session Management
1. **Creation**: New session with unique ID
2. **URL Storage**: Deduplicated URL collection
3. **Processing**: Scraping and clustering workflows
4. **Backend persistence**: Backend sessions, URL records, tab import jobs, SQLite FTS rows, scrape callback metadata, clusters, and local platform data are stored in SQLite when `BACKEND_DB_PATH` is set. Docker stores this at `/data/backend/tab-organizer.sqlite3` on the `backend-data` volume.
5. **Runtime state**: Browser-engine scrape task status and target-site auth queue state are process-local in-memory state in the current implementation.
6. **AI persistence**: AI Engine RAG documents are stored in LanceDB tables on disk (volume `lancedb-data`).
7. **Export**: Session data exported in various formats

## Technology Stack

### Core Infrastructure
- **Containerization**: Docker & Docker Compose
- **API Framework**: FastAPI (Python 3.12)
- **Vector Database**: LanceDB (embedded, file-backed)
- **AI Models**: Ollama (local) or cloud providers

### AI & Machine Learning
- **LLM Providers**: OpenRouter, Ollama, OpenAI, Anthropic Claude, Claude Code, Codex CLI, Codex ACP, DeepSeek, Google Gemini
- **Embedding Models**: `nvidia/llama-nemotron-embed-vl-1b-v2:free` (OpenRouter, 1024-dim, default), `nomic-embed-text` (Ollama, 768-dim), `text-embedding-3-small` (OpenAI, 1536-dim), `text-embedding-004` (Gemini, 768-dim)
- **Clustering**: UMAP + HDBSCAN
- **Content Processing**: BeautifulSoup, trafilatura

### Data Processing
- **Web Scraping**: httpx, BeautifulSoup
- **Authentication**: Credential detection and management
- **File Processing**: CSV, JSON, text file parsing

### Export & Integration
- **Template Engine**: Jinja2
- **Export Formats**: Markdown, JSON, HTML, Obsidian
- **Vector Search**: LanceDB native search/query APIs powering the RAG chatbot
- **Keyword Search**: SQLite FTS5 metadata rows powering hybrid tab search

## Security Considerations

### Credential Management
- Encrypted credential storage
- Domain-specific credential isolation
- Secure error handling without credential exposure

### Network Security
- Internal Docker network isolation
- Service-to-service communication within Docker network
- Health check endpoints for monitoring
- No external API calls for local mode
- Backend Core agent tab endpoints require `BACKEND_AGENT_API_TOKEN`
- Browser CDP tab endpoints only accept local debugging endpoints in v1

### Data Privacy
- Optional local processing (Ollama mode)
- Persistent local storage
- Session-based data isolation
- Configurable data retention

## Scalability & Performance

### Horizontal Scaling
- Microservice architecture enables independent scaling
- Docker Compose profiles for different deployment scenarios
- Stateless services except backend SQLite state, the AI Engine's LanceDB volume, and Ollama's model cache

### Resource Optimization
- Configurable AI provider selection
- Dynamic model switching
- Parallel scraping with configurable concurrency
- Efficient on-disk vector storage with LanceDB

### Performance Monitoring
- Health check endpoints on all services
- Service status monitoring
- Resource usage tracking

## API Specification

### Backend Core (Port 8080)

The tab-management endpoints require bearer auth with `BACKEND_AGENT_API_TOKEN`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/sessions` | POST | Create new session |
| `/api/v1/sessions` | GET | List all sessions |
| `/api/v1/sessions/{session_id}` | GET | Get session details |
| `/api/v1/sessions/{session_id}` | DELETE | Delete a session |
| `/api/v1/urls` | POST | Add URLs to session |
| `/api/v1/urls/{session_id}` | GET | Get session URLs |
| `/api/v1/scrape` | POST | Start scraping |
| `/api/v1/scrape/status/{session_id}` | GET | Get scrape status (proxy to browser-engine) |
| `/api/v1/tabs/import` | POST | Start agent-protected browser tab import |
| `/api/v1/tabs/import/{job_id}` | GET | Get tab import job status |
| `/api/v1/tabs/open` | POST | Open URLs or a session in an attached browser |
| `/api/v1/search` | POST | Hybrid semantic/keyword search across indexed tabs |
| `/api/v1/cluster` | POST | Start clustering |
| `/api/v1/clusters/{session_id}` | GET | Get clustering result |
| `/api/v1/export` | POST | Export session |
| `/api/v1/auth/pending` | GET | List domains awaiting credentials |
| `/api/v1/auth/credentials` | POST | Submit credentials for a pending domain |
| `/api/v1/ingest/v1` | POST | Single versioned idempotent ingest of a scrape result (capture_id + attempt + fetched_at replay/ordering protection); backend is the sole ai-engine `/index` writer |
| `/api/v1/callback/scrape-complete` | POST | Deprecated legacy callback shim over `/api/v1/ingest/v1` (attempt=0 receipt-time newest-wins); kept for rolling-deploy compatibility |
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
| `/health` | GET | Health check (also reports LanceDB readiness) |
| `/providers` | GET | Get provider/model info |
| `/providers/switch` | POST | Hot-swap LLM/embedding providers |
| `/embed` | POST | Generate embeddings |
| `/generate` | POST | Generic LLM completion |
| `/cluster` | POST | UMAP + HDBSCAN + LLM cluster labeling |
| `/index` | POST | Index documents into LanceDB with bounded chunks |
| `/chat` | POST | RAG chat over indexed content |
| `/search` | POST | Vector search over indexed content |
| `/summarize/{session_id}` | GET | Summarize a session's content |
| `/documents/{session_id}` | DELETE | Drop a session's indexed documents |

### Browser Engine (Port 8083)

All Browser Engine endpoints below except `/health` require bearer auth using
`BROWSER_ENGINE_API_TOKEN` or the local callback/AI token fallback. Scrape
targets are restricted to public `http`/`https` URLs unless
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
| `/auth/pending` | GET | All pending auth requests across sessions |
| `/auth/pending/{session_id}` | GET | Pending auth requests for a session |
| `/auth/pending/{domain}` | DELETE | Drop a pending auth request |
| `/auth/credentials` | POST | Submit credentials |
| `/auth/expire` | POST | Force-expire a stored credential |

## Deployment Architecture

### Development Environment
- Single-machine Docker Compose deployment
- Local model storage and vector database
- Hot reloading for all services
- Containerized testing environment

### Production Considerations
- Persistent volume management
- Backup and disaster recovery
- Monitoring and alerting
- CI/CD pipeline integration
- Model update procedures
- Horizontal scaling for high load

### Container Orchestration
```yaml
services:
  # Infrastructure
  ollama:
    image: ollama/ollama:latest
    ports: ["11434:11434"]
    volumes: [ollama-data:/root/.ollama]

  # Application Services
  ai-engine:
    build: ./services/ai-engine
    ports: ["8090:8090"]
    environment:
      VECTOR_DB_PATH: /data/lancedb
    depends_on: [ollama]
    volumes: [lancedb-data:/data/lancedb]

  backend-core:
    build: ./services/backend-core
    ports: ["8080:8080"]
    depends_on: [ai-engine, browser-engine]

  browser-engine:
    build: ./services/browser-engine
    ports: ["8083:8083"]

  web-ui:
    build: ./services/web-ui
    ports: ["8089:8089"]
    depends_on: [backend-core]

volumes:
  lancedb-data:
  ollama-data:
```

## Service Dependencies

```mermaid
graph TD
    UI[Web UI] --> Backend[Backend Core]
    Backend --> AI[AI Engine]
    Backend --> Browser[Browser Engine]
    AI --> LanceDB[LanceDB embedded]
    AI --> Ollama[Ollama]
    Browser --> Backend
```

### Startup Order
1. Ollama (LLM server)
2. AI Engine (mounts LanceDB volume; depends on Ollama)
3. Browser Engine (independent)
4. Backend Core (depends on AI Engine, Browser Engine)
5. Web UI (depends on Backend Core)

### Health Checks
All services expose `/health` endpoints for monitoring:
- Backend Core: `http://localhost:8080/health`
- AI Engine: `http://localhost:8090/health` (also reports LanceDB readiness)
- Browser Engine: `http://localhost:8083/health`
- Ollama: `http://localhost:11434/`
