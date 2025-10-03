#!/bin/bash

# Stop script for Web Scraping & Clustering Tool
set -e

echo "🛑 Stopping Web Scraping & Clustering Tool..."

# Stop all services
docker-compose down

echo "✅ All services stopped successfully!"
echo ""
echo "💡 To remove all data (including Qdrant and Ollama models):"
echo "   docker-compose down -v"