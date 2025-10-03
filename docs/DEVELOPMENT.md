# Development Guide

## Development Workflow

This project follows a spec-driven development methodology with incremental implementation and comprehensive testing at each stage.

## Implementation Status

### ✅ Completed Tasks

#### 1. Core Infrastructure Setup
- [x] Docker Compose configuration for all services
- [x] Qdrant vector database with persistent storage
- [x] Ollama LLM service with model management
- [x] Basic API Gateway with health checks
- [x] Inter-service communication and networking
- [x] Logging and monitoring infrastructure

#### 2. URL Input Service Foundation
- [x] FastAPI service for URL input processing
- [x] File upload endpoints (text, JSON, CSV, Excel)
- [x] URL validation and format detection utilities
- [x] Unit tests for URL parsing and validation
- [x] Multi-format parsing with metadata extraction

#### 8. AI Model Management (Advanced Implementation)
- [x] Hardware detection and model recommendation system
- [x] Dynamic model downloading and installation
- [x] Intelligent fallback chains for resource constraints
- [x] Hot model switching without service restart
- [x] Performance monitoring and resource tracking
- [x] Task-specific model optimization

### 🚧 In Progress

#### 2. URL Input Service (Remaining)
- [ ] URL metadata extraction (domain, path, parameters)
- [ ] URL deduplication and categorization logic
- [ ] Preview functionality for parsed URL lists
- [ ] Batch processing for large URL lists
- [ ] Integration tests for URL processing pipeline

### 📋 Upcoming Tasks

#### 3. Authentication Service
- [ ] Authentication requirement detection
- [ ] Secure credential storage with AES-256 encryption
- [ ] Popup-based authentication using Selenium/Playwright
- [ ] OAuth 2.0 flow handlers
- [ ] Session persistence and renewal

#### 4. Web Scraper Service
- [ ] Scrapy-based scraping framework with rate limiting
- [ ] Content extraction using Beautiful Soup and trafilatura
- [ ] Authentication integration
- [ ] Parallel processing of public/authenticated URLs
- [ ] Smart retry mechanisms

#### 5. Content Analyzer Service
- [ ] Configurable embedding model selection
- [ ] Text chunking with overlap preservation
- [ ] Qdrant integration for vector storage
- [ ] Keyword extraction and quality assessment
- [ ] Model performance monitoring

#### 6. Clustering Service
- [ ] UMAP dimensionality reduction
- [ ] HDBSCAN clustering implementation
- [ ] Cluster quality metrics and visualization
- [ ] LLM-powered cluster labeling
- [ ] Automatic parameter tuning

#### 7. Export Service
- [ ] Template-based export system
- [ ] Notion API integration
- [ ] Obsidian markdown export
- [ ] Word document generation
- [ ] Batch export processing

#### 9. Session Management Service
- [ ] Session-based Qdrant collection management
- [ ] Incremental clustering for new content
- [ ] Session comparison and evolution tracking
- [ ] Backup/restore functionality

#### 10. API Gateway Enhancement
- [ ] Workflow orchestration system
- [ ] Job queue for long-running operations
- [ ] Circuit breaker patterns
- [ ] Comprehensive monitoring

## Development Setup

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- Git

### Local Development Environment

1. **Clone the repository**:
```bash
git clone <repository-url>
cd web-scraping-clustering-tool
```

2. **Set up environment**:
```bash
cp .env.example .env
# Edit .env with your configurations
```

3. **Start development services**:
```bash
./scripts/start.sh
```

4. **Run tests**:
```bash
# Run specific service tests
docker run --rm url-input-service python run_tests.py

# Run integration tests (when available)
./scripts/test.sh
```

## Service Development Guidelines

### Creating a New Service

1. **Create service directory**:
```bash
mkdir services/new-service
cd services/new-service
```

2. **Create basic structure**:
```
services/new-service/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── Dockerfile          # Container configuration
├── test_*.py           # Unit tests
└── README.md           # Service documentation
```

3. **Implement FastAPI service**:
```python
from fastapi import FastAPI
import structlog

logger = structlog.get_logger()
app = FastAPI(title="New Service", version="1.0.0")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "new-service"}
```

4. **Add to Docker Compose**:
```yaml
new-service:
  build:
    context: ./services/new-service
  ports:
    - "808X:808X"
  networks:
    - scraping_network
```

### Testing Guidelines

#### Unit Tests
- Use pytest for Python services
- Test individual functions and classes
- Mock external dependencies
- Aim for >80% code coverage

#### Integration Tests
- Test service-to-service communication
- Use Docker containers for realistic testing
- Test complete workflows end-to-end
- Include error scenarios and edge cases

#### Example Test Structure:
```python
import pytest
from main import URLValidator, URLParser

class TestURLValidator:
    def test_valid_urls(self):
        valid_urls = ["https://example.com", "http://test.org"]
        for url in valid_urls:
            assert URLValidator.is_valid_url(url)
    
    def test_invalid_urls(self):
        invalid_urls = ["not-a-url", "ftp://example.com"]
        for url in invalid_urls:
            assert not URLValidator.is_valid_url(url)
```

### Code Quality Standards

#### Python Code Style
- Follow PEP 8 style guidelines
- Use type hints for function parameters and returns
- Include docstrings for all public functions
- Use meaningful variable and function names

#### Error Handling
- Use structured logging with correlation IDs
- Implement graceful degradation
- Provide clear error messages
- Don't expose sensitive information in errors

#### Example Error Handling:
```python
import structlog

logger = structlog.get_logger()

try:
    result = process_urls(urls)
    logger.info("URLs processed successfully", count=len(result))
    return result
except ValidationError as e:
    logger.error("URL validation failed", error=str(e))
    raise HTTPException(status_code=400, detail="Invalid URL format")
except Exception as e:
    logger.error("Unexpected error", error=str(e))
    raise HTTPException(status_code=500, detail="Internal server error")
```

## Git Workflow

### Commit Message Format
Follow conventional commit format:
```
type(scope): description

feat(url-input): add csv file upload support
fix(scraper): handle authentication timeout
docs(readme): update installation instructions
test(analyzer): add unit tests for embedding generation
```

### Branch Strategy
- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/task-name`: Individual task implementation
- `hotfix/issue-name`: Critical bug fixes

### Commit Guidelines
- Make atomic commits (one logical change per commit)
- Include tests with feature commits
- Update documentation when needed
- Reference issue numbers when applicable

## Debugging and Troubleshooting

### Service Logs
```bash
# View all service logs
./scripts/logs.sh

# View specific service logs
./scripts/logs.sh url-input-service

# Follow logs in real-time
docker-compose logs -f url-input-service
```

### Health Checks
```bash
# Check all services
curl http://localhost:8080/health

# Check specific service
curl http://localhost:8081/health
```

### Common Issues

#### Service Won't Start
1. Check Docker Compose configuration
2. Verify port conflicts
3. Check service dependencies
4. Review service logs for errors

#### Tests Failing
1. Ensure test environment is clean
2. Check test data and fixtures
3. Verify mock configurations
4. Run tests in isolation

#### Performance Issues
1. Monitor resource usage
2. Check model configurations
3. Review caching strategies
4. Optimize database queries

## Contributing Guidelines

### Before Starting Development
1. Check existing issues and tasks
2. Discuss major changes in issues
3. Follow the established architecture
4. Write tests for new functionality

### Pull Request Process
1. Create feature branch from develop
2. Implement changes with tests
3. Update documentation if needed
4. Ensure all tests pass
5. Submit pull request with clear description

### Code Review Checklist
- [ ] Code follows style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] Error handling is appropriate
- [ ] Performance considerations addressed
- [ ] Security implications reviewed

## Deployment

### Development Deployment
```bash
# Start all services
./scripts/start.sh

# Stop all services
./scripts/stop.sh

# Restart specific service
docker-compose restart url-input-service
```

### Production Considerations
- Use production Docker Compose configuration
- Implement proper backup strategies
- Set up monitoring and alerting
- Configure log aggregation
- Plan for model updates and migrations

## Resources

### Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Ollama Documentation](https://ollama.ai/docs/)

### Tools and Libraries
- **Web Framework**: FastAPI
- **Testing**: pytest
- **Logging**: structlog
- **Containerization**: Docker
- **Vector Database**: Qdrant
- **AI Models**: Ollama