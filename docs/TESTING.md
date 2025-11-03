# Testing — concise reference

This file is the single source-of-truth for running and debugging the project's containerized tests. It merges the detailed setup summary into a concise, push-ready guide.

## Quick start

Run the full pipeline:

```bash
./scripts/cli.py test --type all
```

Run specific suites:

```bash
./scripts/cli.py test --type unit
./scripts/cli.py test --type integration
./scripts/cli.py test --type e2e
./scripts/cli.py test --type performance
```

Useful Make targets:

```bash
make test          # unit (default)
make test-all
make test-integration
make test-e2e
make test-service SERVICE=analyzer
make coverage
```

## What’s included

- Unit tests (per-service, fast)
- Integration tests (ephemeral Qdrant & Ollama)
- E2E tests (full workflow, API + UI)
- Performance tests (Locust)
- JUnit + HTML reports (artifacts in CI)

## Running tests (compose)

Start shared test infra and run a runner:

```bash
docker compose --profile test-e2e up -d --build test-qdrant test-ollama test-api-gateway test-web-ui
docker compose --profile test-e2e up --build --abort-on-container-exit e2e-test-runner
```

Run a single service test:

```bash
docker compose --profile test-unit up --build --abort-on-container-exit analyzer-unit-test
```

Bring down test infra (remove volumes for a clean slate):

```bash
docker compose --profile test-unit --profile test-integration --profile test-e2e --profile test-performance --profile test-report down -v
```

## CI snapshot

Pipeline stages (GitHub Actions):
1. Unit tests (parallel) + coverage
2. Code quality (lint, format, type, security)
3. Integration tests
4. End-to-end tests
5. Performance tests (main)
6. Build & push images (main/develop)
7. Deploy (staging=develop, prod=main)

CI produces JUnit XML and coverage artifacts per run.

## Coverage targets

Aim to meet these minimums before merging:

- API Gateway, Auth, Model Manager: 95%
- Analyzer, Scraper, URL Input, Session: 90%
- Clustering, Export: 85%
- Web UI: 80%

Reports live in `coverage/` and `test-reports/`.

## Debugging

Interactive container shell and rerun tests:

```bash
docker compose --profile test-unit run --rm analyzer-unit-test /bin/bash
pytest tests/ -k 'some_test' -vv --pdb
```

Stream logs:

```bash
docker compose --profile test-unit logs -f analyzer-unit-test
```

Remote debugging (dev compose exposes ports):

```bash
docker compose --profile dev up -d qdrant-dev ollama-dev api-gateway-dev
# Attach your IDE to the service debugger port (e.g. 5682)
```

## Pre-push checklist

1. Lint & format:

```bash
make lint
make format
```

2. Run unit tests:

```bash
make test-unit
```

3. Run quick integration smoke-tests:

```bash
docker compose --profile test-integration up -d test-qdrant test-ollama
./scripts/cli.py test --type integration
```

4. Confirm coverage artifacts exist in `coverage/`.

## Troubleshooting (common)

- Ollama not reachable: ensure local Ollama serves on `0.0.0.0` or launch with `./scripts/cli.py start --ollama-mode docker`.
- Tests hang: check `docker compose ps` and service logs.
- Out of disk: run `docker system df` and follow safe cleanup steps (avoid `--volumes` unless you intend to delete persistent data like `ollama_models`).

## Artifacts & locations

- Logs → `logs/`
- JUnit XML → `test-results/`
- Coverage HTML → `coverage/`
- Aggregated reports → `test-reports/`

## Notes

- Keep this file concise; long operational details can remain in `docs/TESTING_SETUP_SUMMARY.md` until you remove it.
- If you want, I will overwrite this file now and move the setup summary to `docs/ARCHIVE/`.

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
   docker compose down -v  # Remove volumes
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

- Check logs: `docker compose logs -f service-name`
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
