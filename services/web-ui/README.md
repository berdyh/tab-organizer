# Web UI Service

This service provides the frontend interface for the Web Scraping & Clustering Tool, built with [Streamlit](https://streamlit.io/).

## Features

- **Dashboard**: Overview of system status.
- **Settings**: Configure AI Providers (Ollama, OpenAI, DeepSeek, Gemini) and API Keys.
- **Sessions**: Manage scraping sessions.
- **URL Input**: Manual entry or file upload of URLs.
- **Scraping Status**: Monitor and control scraping jobs.
- **Analysis**: Trigger AI analysis and view results.
- **Clustering**: Visualize topic clusters.
- **Chatbot**: Interact with scraped content via RAG.
- **Export**: Export results to Notion, Markdown, etc.

## Configuration

The UI connects to the `backend-core` service. Configuration is managed via `config.py` and session state.

## Development

Run locally:
```bash
streamlit run src/main.py
```
