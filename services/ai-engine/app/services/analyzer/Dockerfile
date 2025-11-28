FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install test dependencies
RUN pip install --no-cache-dir pytest pytest-asyncio pytest-mock

COPY . .
RUN mkdir -p /app/logs /app/cache/embeddings /app/config

# Copy models config if it exists (optional)
# Note: This will be handled by docker-compose volume mount instead

EXPOSE 8084

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8084/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8084"]