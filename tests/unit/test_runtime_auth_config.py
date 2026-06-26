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
