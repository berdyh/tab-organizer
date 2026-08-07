"""SEC-48: every ``sec_managed`` probe has a language-neutral fixture entry.

Plan decision 44 extracted every ``sec_managed`` probe's inputs and expected
refusals to ``fixtures/*.json`` so the deferred ``SEC_BOOT_*_CMD`` boot mode
becomes a runner over data and softening a contract is a reviewable diff. That
hedge only holds while the extraction stays COMPLETE: without this guard, probe
N+1 gets added inline next year, boot mode's data set silently stops covering
the suite, and nobody finds out until cutover.

So this probe enumerates the ``sec_managed`` probes from the suite's own test
modules (module-level and function-level ``pytestmark``) and requires:

1. every one of them to be registered in ``fixtures/sec_managed_index.json``;
2. every registered entry to resolve to a real probe block in a real fixture
   file with a non-empty ``expect``/``cases``/``steps``/``checks`` body;
3. every registered entry to still name a live ``sec_managed`` probe, so the
   registry cannot rot in the other direction either.

Seam exception (``sec_seam``): it introspects Python test modules, which is the
only way to ask "which probes exist" without hardcoding the answer it is
checking. TS-porting rule: enumerate the TS suite's equivalent
managed-only-tagged cases and apply the same three-way reconciliation against
the same ``sec_managed_index.json``.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from tests.security import contracts

pytestmark = [pytest.mark.security, pytest.mark.sec_seam]

SUITE_DIR = Path(__file__).parent
INDEX_PATH = contracts.FIXTURES_DIR / "sec_managed_index.json"

# Probe bodies that carry their contract under a key other than `expect`.
_CONTRACT_KEYS = ("expect", "cases", "steps", "checks")


def _index_entries() -> list[dict]:
    data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    entries = data.get("entries")
    assert entries, f"{INDEX_PATH} carries no entries"
    return entries


def _discovered_sec_managed() -> set[tuple[str, str]]:
    """Return {(module_name, test_function_name)} for every sec_managed probe."""
    found: set[tuple[str, str]] = set()
    for path in sorted(SUITE_DIR.glob("test_*.py")):
        module_name = path.stem
        module = importlib.import_module(f"tests.security.{module_name}")
        module_marks = {
            getattr(mark, "name", "") for mark in getattr(module, "pytestmark", [])
        }
        for attr_name in dir(module):
            if not attr_name.startswith("test_"):
                continue
            func = getattr(module, attr_name)
            if not callable(func):
                continue
            func_marks = {
                getattr(mark, "name", "")
                for mark in getattr(func, "pytestmark", [])
            }
            if "sec_managed" in (module_marks | func_marks):
                found.add((module_name, attr_name))
    return found


def test_sec48_every_sec_managed_probe_has_a_fixture_entry():
    """No ``sec_managed`` probe may carry its contract only in Python."""
    entries = _index_entries()
    registered = {(entry["module"], entry["test"]) for entry in entries}
    discovered = _discovered_sec_managed()

    assert discovered, (
        "no sec_managed probes were discovered at all -- the marker moved or "
        "the discovery walk is broken, which would make this guard vacuous"
    )

    unregistered = discovered - registered
    assert not unregistered, (
        "sec_managed probes with no fixture entry (plan decision 44 requires "
        "their inputs and expected refusals to live in fixtures/*.json so boot "
        "mode can run them against any language): "
        f"{sorted(unregistered)}"
    )

    stale = registered - discovered
    assert not stale, (
        "fixtures/sec_managed_index.json registers probes that no longer exist "
        f"or are no longer sec_managed: {sorted(stale)}"
    )


@pytest.mark.parametrize(
    "entry", _index_entries(), ids=[e["test"] for e in _index_entries()]
)
def test_sec48_registered_fixture_entries_resolve(entry):
    """Each registered entry must resolve to a real, non-empty contract block."""
    spec = contracts.probe_spec(entry["fixture"], entry["probe"])
    present = [key for key in _CONTRACT_KEYS if spec.get(key)]
    assert present, (
        f"{entry['probe']} in fixtures/{entry['fixture']}.json carries none of "
        f"{list(_CONTRACT_KEYS)} -- an entry with no contract is not an "
        "extraction, it is a placeholder"
    )
