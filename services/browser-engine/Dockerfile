FROM python:3.11-slim

# Install uv
RUN pip install uv

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN uv pip install --no-cache-dir -r requirements.txt

RUN playwright install chromium --with-deps

COPY app/ ./app/

EXPOSE 8083

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8083"]
