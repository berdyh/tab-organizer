# Test Harness

Cross-service test assets. The unified `docker-compose.yml` exposes one profile
per test type and `scripts/cli.py test` is a thin wrapper that runs the right
profile. The full pipeline (linting, formatting, unit, integration, e2e, build)
is mirrored by `.github/workflows/ci-cd.yml`.

> Authoritative testing guide: [`docs/TESTING.md`](../docs/TESTING.md).

## Layout

```
tests/
├── unit/           # Fast, isolated unit tests (pytest)
├── security/       # Frozen black-box security-invariant suite (SECSUITE_VERSION)
├── integration/    # Tests against running services
├── e2e/            # End-to-end workflow tests (full docker stack)
├── load/           # Locust scenarios (no CLI wrapper yet)
├── conftest.py     # Shared pytest fixtures
├── Dockerfile.test # Image used by every test-* compose service
├── requirements.txt
└── README.md
```

## Running tests

Prefer the CLI so the same containers run locally and in CI:

```bash
./scripts/cli.py test --type unit
./scripts/cli.py test --type integration   # boots the default stack first
./scripts/cli.py test --type e2e           # boots the default stack first
```

Use module-local Make targets for fast iteration inside a boundary:

```bash
make test-backend
make test-ai
make test-browser
make test-web
make test-ops
make test-security   # tests/security -- run as its own CI job (security-tests),
                      # NOT part of tests/unit or test-unit's default command
```

Or invoke `docker compose` directly when you need to iterate on a single
profile:

```bash
docker compose --profile test-unit up --build --abort-on-container-exit test-unit
AI_ENGINE_API_TOKEN=local-test-token BACKEND_CALLBACK_TOKEN=local-test-token BACKEND_AGENT_API_TOKEN=local-test-token PLATFORM_MAINTAINER_SIGNUP_CODE=local-maintainer \
  docker compose --profile default --profile test-integration run --rm test-integration
AI_ENGINE_API_TOKEN=local-test-token BACKEND_CALLBACK_TOKEN=local-test-token BACKEND_AGENT_API_TOKEN=local-test-token PLATFORM_MAINTAINER_SIGNUP_CODE=local-maintainer \
  docker compose --profile default --profile test-e2e run --rm test-e2e
```

`./scripts/cli.py test --type all` runs unit, integration, and e2e in sequence.
There is no `--type performance` in `cli.py` today. `make test-performance`
prints the manual load-test entrypoint; run the load suite from `tests/load/`
when you need it.

## Artefacts

- `coverage/` — HTML coverage report (mounted from `test-unit`)
- `test-results/` — JUnit XML produced by integration and e2e runs

Both are gitignored and recreated on every run.
