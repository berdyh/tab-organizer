# Testing Suite

This directory contains the comprehensive testing suite for the Web Scraping, Analysis & Clustering Tool.

## Directory Structure

```
tests/
├── e2e/                    # End-to-end tests
│   ├── Dockerfile         # E2E test container
│   └── test_full_workflow.py
├── load/                   # Load and performance tests
│   └── locustfile.py      # Locust load testing configuration
├── fixtures/              # Shared test fixtures
├── mocks/                 # Mock services and data
└── README.md             # This file
```

## Quick Start

### Run All Tests

```bash
# Using make
make test-all

# Using script
./scripts/run-all-tests.sh all

# Using docker-compose directly
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

### Run Specific Test Types

```bash
# Unit tests only
make test-unit

# Integration tests only
make test-integration

# End-to-end tests only
make test-e2e

# Performance tests only
make test-performance
```

### Run Tests for Specific Service

```bash
# Using make
make test-service SERVICE=analyzer

# Using docker-compose
docker-compose -f docker-compose.test.yml up --build analyzer-unit-test
```

## Test Types

### Unit Tests

Located in each service directory (e.g., `services/analyzer/test_*.py`)

**Purpose**: Test individual components in isolation
**Speed**: Fast (< 1 second per test)
**Dependencies**: Minimal, mocked external services

**Example**:
```python
def test_url_validation():
    """Test URL validation logic"""
    validator = URLValidator()
    assert validator.is_valid("https://example.com") is True
    assert validator.is_valid("invalid-url") is False
```

### Integration Tests

Located in each service directory with `_integration` suffix

**Purpose**: Test service interactions and external dependencies
**Speed**: Medium (1-5 seconds per test)
**Dependencies**: Real Qdrant and Ollama instances

**Example**:
```python
def test_embedding_storage():
    """Test storing embeddings in Qdrant"""
    client = QdrantClient("http://test-qdrant:6333")
    embedding = generate_embedding("test content")
    result = store_embedding(client, embedding)
    assert result.status == "success"
```

### End-to-End Tests

Located in `tests/e2e/`

**Purpose**: Test complete user workflows
**Speed**: Slow (10-30 seconds per test)
**Dependencies**: Full system deployment

**Example**:
```python
def test_complete_workflow():
    """Test URL submission to export workflow"""
    # Submit URLs
    response = api.post("/urls/batch", json={"urls": [...]})
    job_id = response.json()["job_id"]
    
    # Wait for completion
    wait_for_job(job_id)
    
    # Export results
    export = api.post("/export", json={"format": "markdown"})
    assert export.status_code == 200
```

### Performance Tests

Located in `tests/load/`

**Purpose**: Test system performance under load
**Speed**: Variable (minutes)
**Dependencies**: Full system deployment

**Configuration**:
```python
class WebScrapingUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(5)
    def submit_urls(self):
        self.client.post("/api/urls/batch", json={...})
```

## Test Infrastructure

### Test Containers

All tests run in isolated Docker containers:

- **Unit Test Containers**: Minimal dependencies, fast startup
- **Integration Test Containers**: Include test-qdrant and test-ollama
- **E2E Test Containers**: Full system deployment
- **Load Test Containers**: Locust with custom scenarios

### Test Databases

Temporary databases with tmpfs storage for fast cleanup:

```yaml
test-qdrant:
  image: qdrant/qdrant:v1.7.4
  tmpfs:
    - /qdrant/storage  # In-memory storage
```

### Test Networks

Isolated Docker networks for test isolation:

```yaml
networks:
  test_network:
    driver: bridge
    name: test_network
```

## Writing Tests

### Test Structure

Follow the AAA (Arrange-Act-Assert) pattern:

```python
def test_example():
    # Arrange: Set up test data
    data = {"key": "value"}
    
    # Act: Execute the function
    result = process_data(data)
    
    # Assert: Verify the result
    assert result["status"] == "success"
```

### Using Fixtures

Create reusable test fixtures:

```python
import pytest

@pytest.fixture
def sample_urls():
    """Provide sample URLs for testing"""
    return [
        "https://example.com",
        "https://example.org"
    ]

def test_with_fixture(sample_urls):
    """Test using fixture"""
    assert len(sample_urls) == 2
```

### Mocking External Services

Use mocks for external dependencies:

```python
from unittest.mock import Mock, patch

@patch('module.external_api')
def test_with_mock(mock_api):
    """Test with mocked external API"""
    mock_api.return_value = {"status": "success"}
    result = call_external_api()
    assert result["status"] == "success"
```

### Async Tests

Use pytest-asyncio for async tests:

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    """Test async function"""
    result = await async_operation()
    assert result is not None
```

## Test Coverage

### Coverage Requirements

- **Critical Services**: 95% coverage (API Gateway, Auth, Model Manager)
- **Core Services**: 90% coverage (Analyzer, Scraper, URL Input)
- **Support Services**: 85% coverage (Clustering, Export, Visualization)
- **UI Components**: 80% coverage (Web UI)

### Viewing Coverage

```bash
# Generate coverage reports
make coverage

# Open HTML report
make coverage-report

# View in terminal
coverage report
```

### Coverage Reports

Reports are generated in multiple formats:

- **HTML**: `coverage/service/index.html` - Interactive browsing
- **XML**: `coverage/service.xml` - CI/CD integration
- **Terminal**: Summary in console output

## Debugging Tests

### Interactive Debugging

```bash
# Run tests with debugger
docker-compose -f docker-compose.test.yml run --rm analyzer-unit-test pytest --pdb

# Run specific test with debugger
docker-compose -f docker-compose.test.yml run --rm analyzer-unit-test pytest test_file.py::test_name --pdb
```

### View Test Logs

```bash
# View logs during test execution
docker-compose -f docker-compose.test.yml logs -f analyzer-unit-test

# View logs after test completion
docker logs analyzer-unit-test
```

### Inspect Test Containers

```bash
# Open shell in test container
docker-compose -f docker-compose.test.yml run --rm analyzer-unit-test /bin/bash

# Run tests manually inside container
pytest test_core_components.py -v
```

## CI/CD Integration

### GitHub Actions

Tests run automatically on:
- Push to main/develop branches
- Pull requests
- Manual workflow dispatch

### Local CI Simulation

```bash
# Run full CI pipeline locally
make ci-local

# Run only CI tests
make ci-test
```

## Best Practices

### Test Naming

- Use descriptive names: `test_url_validation_with_invalid_input`
- Group related tests in classes: `TestURLValidation`
- Use prefixes for test types: `test_integration_*`, `test_e2e_*`

### Test Independence

- Each test should run independently
- No shared state between tests
- Use fixtures for setup/teardown

### Test Speed

- Keep unit tests fast (< 1 second)
- Use mocks to avoid slow operations
- Run slow tests separately

### Test Maintenance

- Update tests when code changes
- Remove obsolete tests
- Refactor duplicated test code
- Keep test data realistic

## Troubleshooting

### Common Issues

**Tests timing out**:
```bash
# Increase timeout
pytest --timeout=60

# Check service health
docker-compose ps
```

**Port conflicts**:
```bash
# Clean up old containers
docker-compose down -v
make clean
```

**Out of memory**:
```bash
# Increase Docker memory limit
# Docker Desktop: Settings > Resources > Memory

# Run fewer tests in parallel
pytest -n 2  # Instead of -n auto
```

**Flaky tests**:
```bash
# Run test multiple times
pytest --count=10 test_file.py::test_name

# Add proper waits
await asyncio.sleep(1)
```

### Getting Help

1. Check test logs: `docker-compose logs service-name`
2. Review test results: `cat test-results/*.xml`
3. Check coverage: `coverage report`
4. Read documentation: `docs/TESTING.md`

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Docker Testing Guide](https://docs.docker.com/develop/dev-best-practices/)
- [Locust Documentation](https://docs.locust.io/)
- [Testing Best Practices](../docs/TESTING.md)

## Contributing

When adding new tests:

1. Follow existing patterns
2. Add appropriate fixtures
3. Update coverage requirements
4. Document complex test scenarios
5. Ensure tests pass in CI/CD

## License

Same as main project license.
