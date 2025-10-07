# Visualization Service

Comprehensive architecture diagrams, data pipeline visualization, and real-time monitoring dashboard for the web scraping and clustering system.

## Features

### 1. Architecture Diagrams
- **System Architecture**: High-level view of all microservices and their connections
- **Data Pipeline Flow**: Detailed flow from URL input to export
- **Service Interactions**: Sequence diagrams showing API calls between services
- **Data Flow**: Visual representation of data transformation stages
- **Deployment Architecture**: Docker container and network topology

### 2. Real-Time Monitoring
- **Service Health Checks**: Monitor status of all microservices
- **Pipeline Metrics**: Track processing statistics for each stage
- **Bottleneck Detection**: Automatic identification of performance issues
- **Capacity Planning**: Resource allocation recommendations

### 3. Interactive Dashboard
- **Live Updates**: Auto-refresh every 30 seconds
- **Multiple Views**: Tabbed interface for different diagram types
- **Mermaid Diagrams**: Beautiful, interactive diagrams
- **Metrics Display**: Real-time processing statistics

### 4. Auto-Generated Documentation
- **Service Matrix**: Automatically generated service inventory
- **API Endpoints**: Documentation of all service endpoints
- **Architecture Overview**: System-wide documentation

## API Endpoints

### Core Endpoints
- `GET /` - Service information and available endpoints
- `GET /health` - Health check endpoint
- `GET /dashboard` - Interactive HTML dashboard

### Architecture & Diagrams
- `GET /architecture/diagram` - System architecture data (JSON)
- `GET /diagrams/mermaid/{type}` - Mermaid diagram code
  - Types: `architecture`, `pipeline`, `services`, `data-flow`, `deployment`

### Monitoring
- `GET /services/health` - Health status of all services
- `GET /pipeline/status` - Real-time pipeline metrics with bottleneck detection
- `GET /capacity/planning` - Capacity planning and scaling recommendations

## Usage

### Start the Service

```bash
# Using Docker Compose
docker compose up visualization-service

# Direct (requires Python 3.11+)
cd services/visualization
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8090
```

### Access the Dashboard

Open your browser to: `http://localhost:8090/dashboard`

### Get Architecture Diagram

```bash
curl http://localhost:8090/architecture/diagram
```

### Get Mermaid Diagram

```bash
curl http://localhost:8090/diagrams/mermaid/pipeline
```

### Check Service Health

```bash
curl http://localhost:8090/services/health
```

### Monitor Pipeline

```bash
curl http://localhost:8090/pipeline/status
```

## Generate Documentation

The service includes an auto-documentation generator:

```bash
python doc_generator.py
```

This creates:
- `docs/ARCHITECTURE_GENERATED.md` - System architecture overview
- `docs/API_ENDPOINTS.md` - API endpoint reference

## Testing

Run all tests:

```bash
# Using Docker
./scripts/run-visualization-tests.sh

# Direct
python run_tests.py
```

Test coverage includes:
- ✅ All API endpoints
- ✅ Mermaid diagram generation
- ✅ Bottleneck detection algorithms
- ✅ Service health checking
- ✅ Documentation generation
- ✅ Dashboard HTML rendering

## Architecture

### Service Registry
The service maintains a registry of all microservices with their URLs and ports:
- API Gateway (8080)
- URL Input (8081)
- Authentication (8082)
- Scraper (8083)
- Analyzer (8084)
- Clustering (8085)
- Export (8086)
- Session Manager (8087)
- Model Manager (8088)
- Web UI (8089)

### Pipeline Stages
Monitors 7 key pipeline stages:
1. Input - URL validation and classification
2. Validation - URL format and accessibility checks
3. Authentication - Login and credential management
4. Scraping - Content extraction
5. Analysis - Embedding generation
6. Clustering - HDBSCAN clustering and labeling
7. Export - Format conversion and delivery

### Bottleneck Detection
Automatically identifies issues:
- **High Queue Size**: Queue > 10 items
- **High Failure Rate**: Failures > 5%
- **Slow Processing**: Avg time > 4000ms

## Configuration

Environment variables:
- `LOG_LEVEL` - Logging level (default: INFO)
- `ENVIRONMENT` - Environment name (default: development)

## Dependencies

- FastAPI - Web framework
- Uvicorn - ASGI server
- httpx - Async HTTP client
- Mermaid.js - Diagram rendering (client-side)
- Jinja2 - Template engine
- PyYAML - Configuration parsing

## Integration

The visualization service integrates with:
- All microservices via health check endpoints
- Docker Compose for service discovery
- Qdrant and Ollama for data store monitoring

## Troubleshooting

### Service Shows as "Down"
- Check if the service is running: `docker ps`
- Verify network connectivity: `docker network inspect scraping_network`
- Check service logs: `docker logs <service-name>`

### Dashboard Not Loading
- Ensure port 8090 is not blocked
- Check browser console for JavaScript errors
- Verify API Gateway is healthy

### Diagrams Not Rendering
- Check browser supports Mermaid.js
- Verify CDN access to cdn.jsdelivr.net
- Try refreshing the page

## Future Enhancements

Potential improvements:
- Historical metrics storage and trending
- Alert notifications for critical issues
- Custom dashboard configurations
- Export diagrams as PNG/SVG
- Integration with monitoring tools (Prometheus, Grafana)
- Real-time WebSocket updates
- Performance profiling tools

## License

Part of the Web Scraping and Clustering Tool system.
