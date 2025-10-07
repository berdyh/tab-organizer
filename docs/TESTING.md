# Testing Documentation

## Overview

This document describes the comprehensive containerized testing strategy for the Web Scraping, Analysis & Clustering Tool. All tests run in Docker containers to ensure consistency across different environments.

## Table of Contents

1. [Test Architecture](#test-architecture)
2. [Test Types](#test-types)
3. [Running Tests](#running-tests)
4. [CI/CD Pipeline](#cicd-pipeline)
5. [Test Coverage](#test-coverage)
6. [Debugging Tests](#debugging-tests)
7. [Best Practices](#best-practices)

## Test Architecture

### Testing Pyramid

```
        /\
       /  \
      / E2E \
     /--------\
    /          \
   / Integration\
  /--------------\
 /                \
/   Unit Tests     \
--------------------
```

- **Unit Tests (70%)**: Fast, isolated tests for individual components
- **Integration Tests (20%)**: Tests for service interactions
- **End-to-End Tests (10%)**: Complete workflow tests

### Test Environments

All tests run in isolated Docker containers with the following environments:

1. **Unit Test Environment**: Minimal dependencies, mocked external services
2. **Integration Test Environment**: Real Qdrant and Ollama instances
3. **E2E Test Environment**: Full system deployment
4. **Load Test Environment**: Performance and stress testing

## Test Types

### 1. Unit Tests

Unit tests verify individual components in isolation.

**Location**: Each service has its own test files (e.g., `test_*.py`)

**Running Unit Tests**:
```bash
# Run all unit tests
./scripts/run-all-tests.sh unit

# Run unit tests for specific service
docker-compose -f docker-compose.test.yml up --build analyzer-unit-test

# Run with coverage
docker-compose -f docker-compose.test.yml up --build analyzer-unit-test
```

**Example Services**:
- `url-input-unit-test`: URL validation and parsing
- `auth-unit-test`: Authentication detection
- `scraper-unit-test`: Content extraction
- `analyzer-unit-test`: Embedding generation
- `clustering-unit-test`: UMAP and HDBSCAN
- `export-unit-test`: Export formatting
- `session-unit-test`: Session management
- `visualization-unit-test`: Visualization generation
- `api-gateway-unit-test`: API routing and middleware
- `web-ui-unit-test`: React component tests

### 2. Integration Tests

Integration tests verify interactions between services and external dependencies.

**Running Integration Tests**:
```bash
# Run all integration tests
./scripts/run-all-tests.sh integration

# Run integration tests for specific service
docker-compose -f docker-compose.test.yml up -d test-qdrant test-ollama
docker-compose -f docker-compose.test.yml up --build analyzer-integration-test
```

**Test Infrastructure**:
- `test-qdrant`: Temporary Qdrant instance with tmpfs storage
- `test-ollama`: Temporary Ollama instance with tmpfs storage

**Example Tests**:
- API communication between services
- Database operations with Qdrant
- Model interactions with Ollama
- Authentication workflows
- Data pipeline integration

### 3. End-to-End Tests

E2E tests verify complete user workflows from start to finish.

**Running E2E Tests**:
```bash
# Run all E2E tests
./scripts/run-all-tests.sh e2e

# Run specific E2E test
docker-compose -f docker-compose.test.yml up -d test-qdrant test-ollama test-api-gateway test-web-ui
docker-compose -f docker-compose.test.yml up --build e2e-test-runner
```

**Test Scenarios**:
- Complete URL submission workflow
- Session creation and management
- Search functionality
- Export workflows
- Error handling and recovery

### 4. Performance Tests

Load and performance tests verify system behavior under stress.

**Running Performance Tests**:
```bash
# Run load tests
./scripts/run-all-tests.sh performance

# Run with custom parameters
docker-compose -f docker-compose.test.yml up -d test-api-gateway
docker run -v ./tests/load:/mnt/locust locustio/locust \
  -f /mnt/locust/locustfile.py \
  --host http://test-api-gateway:8080 \
  --users 100 \
  --spawn-rate 10 \
  --run-time 5m \
  --headless
```

**Metrics Collected**:
- Response times (avg, min, max, percentiles)
- Requests per second (RPS)
- Error rates
- Resource utilization

## Running Tests

### Quick Start

```bash
# Run all tests
./scripts/run-all-tests.sh all

# Run specific test type
./scripts/run-all-tests.sh unit
./scripts/run-all-tests.sh integration
./scripts/run-all-tests.sh e2e
./scripts/run-all-tests.sh performance

# Run without cleanup (keep containers running)
./scripts/run-all-tests.sh unit false false
```

### Manual Test Execution

```bash
# Start test infrastructure
docker-compose -f docker-compose.test.yml up -d test-qdrant test-ollama

# Run specific service tests
docker-compose -f docker-compose.test.yml up --build analyzer-unit-test

# View logs
docker-compose -f docker-compose.test.yml logs -f analyzer-unit-test

# Cleanup
docker-compose -f docker-compose.test.yml down -v
```

### Development Workflow

```bash
# Start development environment with hot-reload
docker-compose -f docker-compose.dev.yml up -d

# Run tests against development environment
docker-compose -f docker-compose.test.yml up analyzer-unit-test

# View service logs
docker-compose -f docker-compose.dev.yml logs -f analyzer-dev

# Stop development environment
docker-compose -f docker-compose.dev.yml down
```

## CI/CD Pipeline

### GitHub Actions Workflow

The CI/CD pipeline automatically runs on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Manual workflow dispatch

**Pipeline Stages**:

1. **Unit Tests** (Parallel)
   - All services tested in parallel
   - Coverage reports generated
   - Fast feedback (~5-10 minutes)

2. **Code Quality**
   - Linting (flake8, pylint)
   - Formatting (black, isort)
   - Type checking (mypy)
   - Security scanning (bandit)

3. **Integration Tests** (Parallel)
   - Services tested with real dependencies
   - Database and model interactions verified
   - ~10-15 minutes

4. **End-to-End Tests**
   - Complete workflow validation
   - UI and API testing
   - ~15-20 minutes

5. **Performance Tests** (Main branch only)
   - Load testing
   - Performance benchmarking
   - ~10 minutes

6. **Build & Push Images** (Main/Develop only)
   - Docker images built and pushed
   - Tagged with branch and SHA

7. **Deployment**
   - Staging: Automatic on develop
   - Production: Automatic on main (after all tests pass)

### Local CI Simulation

```bash
# Simulate CI pipeline locally
./scripts/run-all-tests.sh all true true

# Check code quality
flake8 services/
black --check services/
isort --check-only services/
mypy services/ --ignore-missing-imports
bandit -r services/
```

## Test Coverage

### Coverage Requirements

| Service | Unit Tests | Integration Tests | E2E Tests | Target Coverage |
|---------|------------|-------------------|-----------|-----------------|
| API Gateway | ✅ | ✅ | ✅ | 95% |
| URL Input | ✅ | ✅ | ✅ | 90% |
| Authentication | ✅ | ✅ | ✅ | 95% |
| Web Scraper | ✅ | ✅ | ✅ | 90% |
| Content Analyzer | ✅ | ✅ | ✅ | 90% |
| Clustering | ✅ | ✅ | ✅ | 85% |
| Export | ✅ | ✅ | ✅ | 85% |
| Session Manager | ✅ | ✅ | ✅ | 90% |
| Model Manager | ✅ | ✅ | ✅ | 95% |
| Web UI | ✅ | ✅ | ✅ | 80% |

### Viewing Coverage Reports

```bash
# Generate coverage reports
./scripts/run-all-tests.sh unit

# View HTML reports
open coverage/analyzer/index.html
open coverage/clustering/index.html

# View combined coverage
open test-reports/coverage/index.html
```

### Coverage Metrics

Coverage reports include:
- Line coverage
- Branch coverage
- Function coverage
- Missing lines highlighted
- Complexity metrics

## Debugging Tests

### Container Debugging

```bash
# Run tests with interactive shell
docker-compose -f docker-compose.test.yml run --rm analyzer-unit-test /bin/bash

# Inside container, run tests manually
pytest test_core_components.py -v --pdb

# View container logs
docker logs analyzer-unit-test -f

# Inspect container
docker exec -it analyzer-unit-test /bin/bash
```

### Remote Debugging

Development containers expose debugger ports:

```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d analyzer-dev

# Connect debugger to port 5682 (analyzer service)
# Use your IDE's remote debugging feature
```

**VS Code launch.json example**:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Remote Attach",
      "type": "python",
      "request": "attach",
      "connect": {
        "host": "localhost",
        "port": 5682
      },
      "pathMappings": [
        {
          "localRoot": "${workspaceFolder}/services/analyzer",
          "remoteRoot": "/app"
        }
      ]
    }
  ]
}
```

### Test Debugging Tips

1. **Use pytest markers**:
   ```python
   @pytest.mark.skip(reason="Debugging")
   def test_something():
       pass
   ```

2. **Run specific tests**:
   ```bash
   pytest test_file.py::TestClass::test_method -v
   ```

3. **Use pytest-pdb**:
   ```bash
   pytest --pdb  # Drop into debugger on failure
   pytest --pdbcls=IPython.terminal.debugger:Pdb  # Use IPython debugger
   ```

4. **Increase verbosity**:
   ```bash
   pytest -vv --tb=long
   ```

5. **Capture output**:
   ```bash
   pytest -s  # Don't capture stdout
   pytest --log-cli-level=DEBUG  # Show debug logs
   ```

## Best Practices

### Writing Tests

1. **Follow AAA Pattern**:
   ```python
   def test_something():
       # Arrange
       data = setup_test_data()
       
       # Act
       result = function_under_test(data)
       
       # Assert
       assert result == expected_value
   ```

2. **Use Fixtures**:
   ```python
   @pytest.fixture
   def sample_data():
       return {"key": "value"}
   
   def test_with_fixture(sample_data):
       assert sample_data["key"] == "value"
   ```

3. **Mock External Dependencies**:
   ```python
   from unittest.mock import Mock, patch
   
   @patch('module.external_api')
   def test_with_mock(mock_api):
       mock_api.return_value = "mocked"
       result = function_that_calls_api()
       assert result == "mocked"
   ```

4. **Test Edge Cases**:
   - Empty inputs
   - Null values
   - Large datasets
   - Invalid data
   - Error conditions

5. **Keep Tests Independent**:
   - Each test should run independently
   - No shared state between tests
   - Use fixtures for setup/teardown

### Container Best Practices

1. **Use tmpfs for test data**:
   ```yaml
   tmpfs:
     - /qdrant/storage
     - /root/.ollama
   ```

2. **Set proper timeouts**:
   ```yaml
   healthcheck:
     test: ["CMD", "curl", "-f", "http://localhost:6333/"]
     interval: 10s
     timeout: 5s
     retries: 5
   ```

3. **Clean up resources**:
   ```bash
   docker-compose down -v  # Remove volumes
   docker system prune -f  # Clean up unused resources
   ```

4. **Use build cache**:
   ```yaml
   build:
     context: .
     cache_from:
       - myapp:latest
   ```

### Performance Testing Best Practices

1. **Start with baseline**:
   - Establish performance baseline
   - Track metrics over time
   - Set realistic targets

2. **Gradual load increase**:
   - Start with low load
   - Gradually increase
   - Monitor for breaking points

3. **Monitor resources**:
   - CPU usage
   - Memory usage
   - Network I/O
   - Disk I/O

4. **Test realistic scenarios**:
   - Mix of operations
   - Realistic data sizes
   - Expected user behavior

## Troubleshooting

### Common Issues

1. **Tests timing out**:
   - Increase timeout values
   - Check service health
   - Verify network connectivity

2. **Port conflicts**:
   - Check for running containers
   - Use different port mappings
   - Clean up old containers

3. **Out of memory**:
   - Reduce parallel tests
   - Increase Docker memory limit
   - Use tmpfs for temporary data

4. **Flaky tests**:
   - Add proper waits
   - Use health checks
   - Increase retry logic

### Getting Help

- Check logs: `docker-compose logs -f service-name`
- Inspect containers: `docker inspect container-name`
- View test results: `cat test-results/*.xml`
- Review coverage: `open coverage/service/index.html`

## Continuous Improvement

### Metrics to Track

- Test execution time
- Test coverage percentage
- Flaky test rate
- Build success rate
- Deployment frequency

### Regular Maintenance

- Update dependencies monthly
- Review and refactor tests quarterly
- Update documentation as needed
- Monitor and optimize slow tests
- Remove obsolete tests

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Docker Testing Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Locust Documentation](https://docs.locust.io/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
