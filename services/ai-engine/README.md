# AI Engine Service

This service handles AI-intensive tasks:
- **Content Analysis**: Embedding generation, summarization, quality assessment.
- **Chatbot**: RAG-based chat interface.
- **Clustering**: Topic clustering using UMAP/HDBSCAN.

## Features
- **Multi-Provider Support**: Switch between Local (Ollama) and Cloud (OpenAI, DeepSeek, Gemini).
- **Dynamic Vector Sizing**: Automatically handles different embedding dimensions (e.g. 1536 for OpenAI, 768 for Nomic).
- **LLM Factory**: Modular client management for different AI providers.

## Architecture

Built with FastAPI, it utilizes `torch`, `transformers`, and `qdrant-client` to process data. It exposes endpoints consumed by `backend-core`.

## Structure

- `app/routers/`: API endpoints.
- `app/core/`: Shared logic (LLM Factory).
- `app/services/`: Core application logic (Analyzer, Chatbot, Clustering).
- `main.py`: Application entry point mounting sub-services.
