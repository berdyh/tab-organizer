"""Static regressions for auth-aware local runtime entry points."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_ci_cross_service_jobs_export_local_service_tokens():
    workflow = (ROOT / ".github" / "workflows" / "ci-cd.yml").read_text()

    assert "AI_ENGINE_API_TOKEN: local-test-token" in workflow
    assert "BACKEND_CALLBACK_TOKEN: local-test-token" in workflow
    assert "BACKEND_AGENT_API_TOKEN: local-test-token" in workflow
    assert "PLATFORM_MAINTAINER_SIGNUP_CODE: local-maintainer" in workflow


def test_make_start_targets_use_cli_token_generation():
    makefile = (ROOT / "Makefile").read_text()

    assert "./scripts/cli.py start -d --dev" in makefile
    assert "./scripts/cli.py start -d --build --dev" in makefile
    assert "./scripts/cli.py start -d" in makefile


def test_direct_compose_docs_show_required_local_service_tokens():
    testing_doc = (ROOT / "docs" / "TESTING.md").read_text()
    test_readme = (ROOT / "tests" / "README.md").read_text()

    for text in (testing_doc, test_readme):
        assert "AI_ENGINE_API_TOKEN=local-test-token" in text
        assert "BACKEND_CALLBACK_TOKEN=local-test-token" in text
        assert "BACKEND_AGENT_API_TOKEN=local-test-token" in text
        assert "PLATFORM_MAINTAINER_SIGNUP_CODE=local-maintainer" in text


def test_service_images_package_shared_url_safety_module():
    compose = (ROOT / "docker-compose.yml").read_text()
    backend_dockerfile = (ROOT / "services" / "backend-core" / "Dockerfile").read_text()
    browser_dockerfile = (
        ROOT / "services" / "browser-engine" / "Dockerfile"
    ).read_text()

    assert "dockerfile: services/backend-core/Dockerfile" in compose
    assert "dockerfile: services/browser-engine/Dockerfile" in compose
    assert "context: ./services/backend-core" not in compose
    assert "context: ./services/browser-engine" not in compose

    for dockerfile in (backend_dockerfile, browser_dockerfile):
        assert "COPY services/__init__.py ./services/__init__.py" in dockerfile
        assert "COPY services/url_safety.py ./services/url_safety.py" in dockerfile
