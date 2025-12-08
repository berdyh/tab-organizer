# Web Scraping & Clustering Tool

A comprehensive system for web scraping, content analysis, and intelligent clustering using local and cloud AI models. The system consolidates powerful features into a modular, efficient microservice architecture.

## 🚀 Features

- **🤖 AI Chatbot Interface**: Natural language queries to explore your scraped content (Supports Ollama, OpenAI, DeepSeek, Gemini).
- **🌐 Web Content Scraping**: Robust scraping with interactive authentication handling.
- **🔐 Parallel Authentication**: Detects and handles login requirements in parallel to other workflows via `browser-engine`.
- **🧠 AI Processing**:
    - **Local**: Ollama for LLM and embeddings (Privacy-focused).
    - **Cloud**: Integration with OpenAI, DeepSeek, and Google Gemini.
- **📊 Intelligent Clustering**: UMAP + HDBSCAN for meaningful content grouping.
- **📤 Multi-format Export**: Notion, Obsidian, Word, and Markdown support.
- **🏗️ Microservice Architecture**: Consolidated into three main engines for efficiency.

## 🏗️ Architecture

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

## 🛠️ Quick Start

### Prerequisites
- **Docker** with Docker Compose V2
- **Python 3.9+** (if running locally)

### 🚀 Setup & Run

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
    - **Web UI**: http://localhost:8089 (Configure AI providers here)
    - **API Gateway**: http://localhost:8080

## 🔧 Configuration & Workflows

### 1. AI Configuration
Go to the **Settings** page in the Web UI to select your AI Provider:
- **Local**: Uses Ollama (ensure Ollama container is running).
- **Cloud**: Enter API keys for OpenAI, DeepSeek, or Gemini. Keys are stored securely in your session.

### 2. Scraping & Authentication
- Input URLs in the **URL Input** page.
- Start scraping in the **Scraping Status** page.
- **Authentication**: If a site requires login, the `browser-engine` detects this. The system is designed to handle this interactively or via parallel browser sessions, ensuring scraping continues for public sites while authenticated ones await credentials.

### 3. Analysis & Clustering
- Trigger **Analysis** to generate embeddings and summaries (vector size is automatically handled for different models).
- Use **Clustering** to group related content.

### 4. Chat & Discovery
- Use the **Chatbot** to ask questions like "Find related information about [Topic]".
- The system uses RAG (Retrieval-Augmented Generation) to find relevant content from Qdrant and answer using the selected LLM.

## 📚 Service Details

- **Backend Core**: Unifies API Gateway, Session Management, URL Input, Export, and Auth API.
- **AI Engine**: Handles Embedding generation, LLM analysis, Chat, and Clustering. Supports dynamic model switching.
- **Browser Engine**: Handles heavy browser automation tasks (Playwright/Selenium) for scraping and auth detection.

## 🤝 Contributing

See individual service folders for specific development instructions.
