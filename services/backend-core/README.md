# Backend Core Service

This service consolidates the core backend functionalities including:
- **API Gateway Logic**: Routing and proxying requests.
- **Session Management**: Managing user sessions.
- **URL Input**: Handling URL submissions and file uploads.
- **Export**: Handling data export requests.
- **Authentication**: Basic token management.

## Architecture

Built with FastAPI, it acts as the entry point for the frontend and routes heavy tasks to `ai-engine` and `browser-engine`.

## Structure

- `app/routers/`: API endpoints for each domain.
- `app/core/`: Shared configuration and utilities.
- `main.py`: Application entry point.
