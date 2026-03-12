# Testing Documentation

This file is the single source-of-truth for running and debugging the project's containerized tests.

## Quick Start

Run the full pipeline:

```bash
./scripts/cli.py test --type all
```

Run specific suites:

```bash
./scripts/cli.py test --type unit
./scripts/cli.py test --type integration
./scripts/cli.py test --type e2e
```

Useful Make targets:

```bash
make test          # unit (default)
make test-all
make test-integration
make test-e2e
make coverage
```

## What's Included

- Unit tests (fast, isolated)
- Integration tests (with Qdrant & Ollama)
- E2E tests (full workflow, API + UI)
- JUnit + HTML reports (artifacts in CI)

## Running Tests with Docker Compose

### Unit Tests

```bash
docker compose --profile test-unit up --build --abort-on-container-exit test-unit
```

### Integration Tests

Start infrastructure and services:

```bash
docker compose --profile test-integration up -d qdrant ollama
docker compose --profile test-integration up -d --build backend-core ai-engine browser-engine
docker compose --profile test-integration up --build --abort-on-container-exit test-integration
```

### E2E Tests

Start full stack:

```bash
docker compose --profile test-e2e up -d qdrant ollama
docker compose --profile test-e2e up -d --build backend-core ai-engine browser-engine web-ui
docker compose --profile test-e2e up --build --abort-on-container-exit test-e2e
```

### Cleanup

Bring down test infrastructure (remove volumes for clean slate):

```bash
docker compose --profile test-unit --profile test-integration --profile test-e2e down -v
```

## CI Pipeline

Pipeline stages (GitHub Actions):
1. Unit tests + coverage
2. Code quality (lint, format, type, security)
3. Integration tests
4. End-to-end tests
5. Build & push images (main/develop)
6. Deploy (staging=develop, prod=main)

CI produces JUnit XML and coverage artifacts per run.

## Coverage Targets

Aim to meet these minimums before merging:

- Backend Core: 90%
- AI Engine: 85%
- Browser Engine: 85%
- Web UI: 80%

Reports live in `coverage/` and `test-results/`.

## Debugging

Interactive container shell and rerun tests:

```bash
docker compose --profile test-unit run --rm test-unit /bin/bash
pytest tests/unit -k 'some_test' -vv --pdb
```

Stream logs:

```bash
docker compose --profile test-unit logs -f test-unit
```

Check service logs during integration tests:

```bash
docker compose --profile test-integration logs -f backend-core
docker compose --profile test-integration logs -f ai-engine
docker compose --profile test-integration logs -f browser-engine
```

## Pre-Push Checklist

1. Lint & format:

```bash
make lint
make format
```

2. Run unit tests:

```bash
make test-unit
```

3. Run integration smoke-tests:

```bash
docker compose --profile test-integration up -d qdrant ollama
./scripts/cli.py test --type integration
```

4. Confirm coverage artifacts exist in `coverage/`.

## Troubleshooting

### Common Issues

**Ollama not reachable**: Ensure Ollama is running and accessible at the configured host.

**Tests hang**: Check `docker compose ps` and service logs.

**Out of disk**: Run `docker system df` and clean up unused resources.

**Port conflicts**: Stop other services using the same ports (8080, 8083, 8089, 8090, 6333, 11434).

### Service Health Checks

Check if services are healthy:

```bash
curl http://localhost:8080/health  # Backend Core
curl http://localhost:8090/health  # AI Engine
curl http://localhost:8083/health  # Browser Engine
curl http://localhost:6333/        # Qdrant
curl http://localhost:11434/       # Ollama
```

## Test Organization

Tests are organized by type, not by service:

```
tests/
├── unit/                  # Fast, isolated unit tests
│   ├── test_auth_detector.py
│   ├── test_clustering.py
│   ├── test_dedup.py
│   └── test_url_store.py
├── integration/           # Tests with services running
│   └── test_api.py
└── e2e/                   # Full workflow tests
    ├── test_workflow.py
    └── test_full_workflow.py
```

## Writing Tests

### Best Practices

1. **Keep tests independent**: Each test should run independently without shared state.

2. **Use fixtures**:
   ```python
   @pytest.fixture
   def sample_data():
       return {"key": "value"}
   
   def test_with_fixture(sample_data):
       assert sample_data["key"] == "value"
   ```

3. **Mock external dependencies**:
   ```python
   from unittest.mock import Mock, patch
   
   @patch('module.external_api')
   def test_with_mock(mock_api):
       mock_api.return_value = "mocked"
       result = function_that_calls_api()
       assert result == "mocked"
   ```

4. **Test edge cases**:
   - Empty inputs
   - Null values
   - Large datasets
   - Invalid data
   - Error conditions

### Container Best Practices

1. **Use tmpfs for test data** (faster, no disk I/O):
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
   docker compose down -v  # Remove volumes
   docker system prune -f  # Clean up unused resources
   ```

## Artifacts & Locations

- Logs: `logs/`
- JUnit XML: `test-results/`
- Coverage HTML: `coverage/`

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
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
