#!/bin/bash

# Setup script for Web Scraping & Clustering Tool
set -e

echo "🚀 Setting up Web Scraping & Clustering Tool..."

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not available. Please install Docker Compose V2 or update Docker."
    echo "💡 Try: 'docker --version' and 'docker compose version'"
    exit 1
fi

# Helper to upsert values in .env
update_env_var() {
    local key="$1"
    local value="$2"

    if [ ! -f .env ]; then
        return
    fi

    python3 - <<PY
from pathlib import Path

env_path = Path(".env")
key = "${key}"
value = "${value}"

if not env_path.exists():
    raise SystemExit(0)

lines = env_path.read_text().splitlines()
updated = False

for idx, line in enumerate(lines):
    if line.startswith(f"{key}="):
        lines[idx] = f"{key}={value}"
        updated = True
        break

if not updated:
    lines.append(f"{key}={value}")

env_path.write_text("\n".join(lines) + "\n")
PY
}

# Utility to determine which Docker profile to use
select_docker_profile() {
    if command -v nvidia-smi &> /dev/null && nvidia-smi >/dev/null 2>&1; then
        echo "🎮 GPU detected – enabling GPU profile for Ollama"
        PROFILE="--profile gpu"
    else
        echo "💻 No GPU detected – using CPU profile for Ollama"
        PROFILE="--profile cpu"
    fi
}

# Verify that Docker containers can reach the host-side Ollama instance
check_local_ollama_from_docker() {
    docker run --rm --add-host host.docker.internal:host-gateway \
        curlimages/curl:8.5.0 -s --max-time 5 http://host.docker.internal:11434/api/tags >/dev/null 2>&1
}

# Create logs directory
echo "📁 Creating logs directory..."
mkdir -p logs

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your configuration before starting services"
fi

# Decide how to provide Ollama
echo "🔍 Detecting Ollama availability..."
USE_LOCAL_OLLAMA=false
PROFILE=""

if command -v curl &> /dev/null && curl -s --connect-timeout 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "✅ Local Ollama detected on port 11434"
    echo "🔗 Setup will use the existing local installation"
    USE_LOCAL_OLLAMA=true
    update_env_var "OLLAMA_URL" "http://localhost:11434"

    echo "🧪 Verifying Docker network access to local Ollama..."
    if ! check_local_ollama_from_docker; then
        echo "⚠️  Docker containers cannot reach the local Ollama instance."
        echo "   Falling back to the Dockerized Ollama service."
        USE_LOCAL_OLLAMA=false
        select_docker_profile
        update_env_var "OLLAMA_URL" "http://ollama:11434"
    fi
else
    echo "🐳 No local Ollama found, preparing Docker deployment"
    select_docker_profile
    update_env_var "OLLAMA_URL" "http://ollama:11434"
fi

# Pull required Docker images
if [ "$USE_LOCAL_OLLAMA" = true ]; then
    echo "📦 Pulling Docker images (skipping Ollama – using local install)..."
    docker compose pull qdrant
else
    echo "📦 Pulling Docker images (including Ollama container)..."
    docker compose $PROFILE pull qdrant ollama
fi

# Build services
echo "🔨 Building services..."
docker compose build

echo "✅ Setup complete!"
echo ""
echo "🚀 Quick Start:"
echo "1. Run: ./scripts/start.sh"
echo "2. Access the Web UI at http://localhost:8089"
echo "3. Access the API Gateway at http://localhost:8080"
echo ""
echo "📝 Optional Configuration:"
echo "1. Edit .env file with your configuration (already has good defaults)"
echo "2. For advanced model selection (after starting services):"
echo "   - 'python3 scripts/model-manager.py auto' - Hardware-optimized model selection"
echo "   - './scripts/setup-models.sh' - Interactive model setup"
echo ""
echo "💡 The .env file already contains recommended models for your system:"
echo "  - LLM: phi4:3.8b (Microsoft's latest, GPU optimized)"
echo "  - Embedding: mxbai-embed-large (highest quality)"
echo ""
echo "⚠️  Note: Model setup tools require services to be running first!"
echo "   Start with './scripts/start.sh' then use model management tools."
