# Chatbot Service

The chatbot service provides a natural-language interface for exploring clustered web content captured by the pipeline. It runs as a FastAPI application and exposes endpoints that power the web UI as well as the API Gateway.

## Architecture

```
┌─────────────────────┐      ┌─────────────────────────────┐
│  Web UI / API GW    │─────▶│  FastAPI App (chatbot.api)  │
└─────────────────────┘      │  • HTTP layer & routing     │
                             │  • Service resolution       │
                             └────────────┬────────────────┘
                                          │
                                          ▼
                             ┌─────────────────────────────┐
                             │ ChatbotService              │
                             │  • Intent extraction        │
                             │  • Semantic search logic    │
                             │  • Conversation memory      │
                             └──┬───────────────┬──────────┘
                                │               │
                ┌───────────────┘               └───────────────┐
                ▼                                               ▼
      ┌────────────────────┐                         ┌──────────────────────┐
      │ Qdrant (vector DB) │                         │ Ollama (LLM / embeds)│
      │ chatbot.clients    │                         │ chatbot.clients      │
      └────────────────────┘                         └──────────────────────┘
```

Key modules:

- `chatbot.config`: Centralised settings with environment variable support.
- `chatbot.clients`: Factories for Qdrant and Ollama clients.
- `chatbot.models`: Pydantic models (validated URLs, cluster summaries, responses).
- `chatbot.store`: Conversation memory with mapping semantics for compatibility.
- `chatbot.service`: Core business logic and dependency orchestration.
- `chatbot.api`: FastAPI application factory with compatibility fallbacks.
- `services/chatbot/main.py`: Compatibility shim re-exporting the symbols expected by legacy code and tests.

## Workflow & Integrations

- **API Gateway** forwards `/chat/*` requests to this service via the configured service registry entry (`chatbot-service:8092`).
- **Web UI** uses the chatbot endpoints to present conversational responses, summaries, and cluster exploration flows. Cluster metadata comes from Qdrant via the chatbot service.
- **Clustering pipeline** populates Qdrant collections (`session_<id>`). The chatbot service reads these collections to answer semantic search and cluster queries. Ensure clustering runs before expecting rich responses.

## Setup

1. Install dependencies inside the repository virtual environment:
   ```bash
   pip install -r services/chatbot/requirements.txt
   ```
2. Optional: configure environment variables (defaults shown):
   - `QDRANT_HOST=qdrant`
   - `QDRANT_PORT=6333`
   - `OLLAMA_BASE_URL=http://ollama:11434`
   - `DEFAULT_LLM_MODEL=phi4:3.8b`
   - `DEFAULT_EMBEDDING_MODEL=mxbai-embed-large`
   - `QDRANT_COLLECTION_PREFIX=session_`
3. Run the service locally:
   ```bash
   uvicorn services.chatbot.main:app --host 0.0.0.0 --port 8092
   ```

## Testing

Execute the chatbot test suite (after installing requirements):
```bash
python -m pytest services/chatbot/tests
```

The tests register `sys.modules["main"]` to maintain compatibility with legacy import paths and mock patch locations. They also assert that URLs are validated as `AnyHttpUrl` and that patched `RobotFileParser` instances propagate correctly.

Container-based test utilities now live in `docker/` (for example `docker/docker-compose.test.yml` and `docker/Dockerfile.test`). Coverage and test log artefacts produced by those flows are stored beneath `reports/`.

## Development Tips

- Import from `chatbot` (the package) for new code; legacy imports from `main` continue to work via the compatibility layer.
- When writing tests or tooling that patch dependencies, prefer patching `main.qdrant_client` or `main.chatbot_service`—the service resolves those symbols at runtime.
- Use `app.state.resolve_robot_parser_cls()` to fetch the latest `RobotFileParser` class. This honours runtime patches (handy for crawler integration tests).
- Conversation history is backed by `ConversationStore`, which behaves like a mapping and also offers helper methods (`append`, `get`, `clear`).
- To extend integrations with other services, favour dependency injection via the package factories (`chatbot.clients`) so behaviour remains easy to mock.
- Local tooling such as `tools/validate_implementation.py` assumes paths relative to the service root—invoke it from the service directory (`python tools/validate_implementation.py`) for accurate results.
