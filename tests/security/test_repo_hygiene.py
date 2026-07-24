"""SEC-37 / SEC-38: repo + CI secret-scanning hygiene (plan finding 35)."""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from tests.security.conftest import REPO_ROOT

pytestmark = [pytest.mark.security]

CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci-cd.yml"


def _step_uses_gitleaks(step: dict) -> bool:
    uses = str(step.get("uses", "")).lower()
    run = str(step.get("run", "")).lower()
    return "gitleaks" in uses or "gitleaks" in run


def test_secret_scanning_runs_in_ci():
    """A gitleaks job must exist and must not be conditionally skipped."""
    assert CI_WORKFLOW.exists(), "CI workflow file is missing"
    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    jobs = workflow.get("jobs", {})

    matched_job = None
    matched_step = None
    for job in jobs.values():
        for step in job.get("steps", []) or []:
            if _step_uses_gitleaks(step):
                matched_job = job
                matched_step = step
                break
        if matched_job is not None:
            break

    assert matched_job is not None, (
        "no CI job invokes gitleaks (gitleaks/gitleaks-action or a gitleaks binary)"
    )
    assert "if" not in matched_job, "gitleaks job is gated by a job-level condition"
    assert "if" not in matched_step, "gitleaks step is gated by a step-level condition"


def test_gitleaks_config_allowlists_fixtures():
    """The suite's fake tokens/fixtures must be allowlisted so scans stay clean."""
    config = REPO_ROOT / ".gitleaks.toml"
    assert config.exists(), ".gitleaks.toml is missing"
    text = config.read_text(encoding="utf-8")
    assert "tests/security/fixtures" in text


def test_working_tree_scans_clean():
    """If a gitleaks binary is available, the working tree must scan clean."""
    binary = shutil.which("gitleaks")
    if not binary:
        pytest.skip("gitleaks binary not on PATH")
    result = subprocess.run(
        [
            binary,
            "detect",
            "--source",
            str(REPO_ROOT),
            "--no-banner",
            "--exit-code",
            "1",
        ],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, (
        "gitleaks reported findings:\n"
        + result.stdout.decode("utf-8", "replace")
        + result.stderr.decode("utf-8", "replace")
    )
