# Web Scraping & Clustering Tool

A comprehensive system for web scraping, content analysis, and intelligent clustering using local and cloud AI models.

## Features

- ** AI Chatbot Interface**: Natural language queries to explore your scraped content (Supports Ollama, OpenAI, DeepSeek, Gemini).
- ** Web Content Scraping**: Scrapy-based scraping with authentication support.
- ** AI Processing**:
    - **Local**: Ollama for LLM and embeddings.
    - **Cloud**: OpenAI, DeepSeek, Gemini integration.
- ** Intelligent Clustering**: UMAP + HDBSCAN for meaningful content grouping.
- ** Multi-format Export**: Notion, Obsidian, Word, and Markdown support.
- ** Microservice Architecture**: Consolidated into three main engines for efficiency.

## Architecture

```
┌─────────────────┐
│    Web UI       │
│   (Streamlit)   │
│   (Port 8089)   │
└────────┬────────┘
         │
    ┌────▼────┐    ┌─────────────────┐
    │ Backend │────│  Qdrant Vector  │
    │  Core   │    │  DB (Port 6333) │
    │(Port 8080)   └─────────────────┘
    └────┬────┘
         │
    ┌────┴────┐
    │ Engines │
    └────┬────┘
         │
┌────────▼────────┐      ┌────────▼────────┐
│    AI Engine    │      │ Browser Engine  │
│   (Port 8090)   │      │   (Port 8083)   │
│ - Analyzer      │      │ - Scraper       │
│ - Chatbot       │      │ - Auth Browser  │
│ - Clustering    │      └─────────────────┘
└─────────────────┘
```

## Quick Start

### Prerequisites
- **Docker** with Docker Compose V2
- **Python 3.9+** (if running locally)

### Setup & Run

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd web-scraping-clustering-tool
    ```

2.  **Start Services**:
    ```bash
    docker compose up --build -d
    ```

3.  **Access the App**:
    - Web UI: http://localhost:8089
    - API Gateway (Backend Core): http://localhost:8080

## Configuration

Configure AI providers in the Web UI **Settings** page.
- **Local**: Uses Ollama (ensure Ollama is running).
- **Cloud**: Enter API keys for OpenAI, DeepSeek, or Gemini.

## Service Details

- **Backend Core**: Unifies API Gateway, Session Management, URL Input, Export, and Auth API.
- **AI Engine**: Handles Embedding generation, LLM analysis, Chat, and Clustering. Supports dynamic model switching.
- **Browser Engine**: Handles heavy browser automation tasks (Playwright/Selenium) for scraping and auth detection.

## Contributing

See individual service folders for specific development instructions.
