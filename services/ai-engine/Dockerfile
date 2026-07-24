FROM python:3.12-slim-bookworm

# Install uv
RUN pip install uv

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY services/ai-engine/requirements.txt .
RUN uv pip install --system --no-cache-dir -r requirements.txt

COPY services/ai-engine/app/ ./app/
COPY services/__init__.py ./services/__init__.py
COPY services/observability.py ./services/observability.py
COPY config/ ./config/

EXPOSE 8090

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090"]
