#!/bin/sh
# The repo's mypy gate. `make type-check` runs this inside a throwaway
# python:3.12-slim container; CI runs it directly on the runner. Both go
# through this one file so the local gate and the CI gate cannot drift -- when
# they were two copies of a command, CI's copy also carried `|| true`.
#
# WHY PER-SERVICE, NOT `mypy services/`
# ------------------------------------
# backend-core, ai-engine and browser-engine each ship a top-level package
# literally named `app`. A single pass over `services/` therefore dies with
#     Duplicate module named "app" ... (errors prevented further checking)
# and exits 2 before analysing one line. That is what this gate did from the
# day it was written until 2026-08-07: it was red for a reason that had nothing
# to do with anyone's change, so nobody could read a result out of it.
#
# The alternative (--explicit-package-bases + MYPYPATH) also resolves the
# collision, but by inventing module paths -- `services.ai-engine.app` -- that
# nothing at runtime ever uses. Running mypy from inside each service instead
# reproduces exactly what that service sees in its own container, where `app`
# IS the root package. Same reason each service has its own requirements.txt.
#
# WHAT THIS DOES AND DOES NOT CHECK
# ---------------------------------
# Third-party runtime dependencies (fastapi, playwright, lancedb, streamlit,
# httpx, ...) are NOT installed here; only their stub packages are, and
# --ignore-missing-imports turns the rest into `Any`. So this gate checks our
# own code and its internal consistency -- signatures, Optionals, annotations,
# overrides -- not our use of those libraries. Installing four services' worth
# of dependencies to type-check them is a different, much slower gate; if you
# want it, make it a separate target rather than quietly widening this one.
#
# Every service is checked even after one fails, so a single run reports the
# whole tree. Exit status is non-zero if any of them failed.
set -u

# Pinned: an unpinned type checker changes its mind between runs, and a gate
# that goes red on someone else's release is a gate people learn to ignore.
# --python-version below is pinned for the same reason: the runtime is 3.12
# regardless of which interpreter happens to run mypy.
MYPY_PINS="mypy==1.13.0 types-requests==2.32.0.20250328 types-PyYAML==6.0.12.20250516"

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
MYPY_ARGS="--ignore-missing-imports --python-version=3.12"

if [ "${TYPE_CHECK_SKIP_INSTALL:-0}" != "1" ]; then
    # shellcheck disable=SC2086 # $MYPY_PINS is a deliberate word list
    pip install --quiet --disable-pip-version-check --root-user-action=ignore \
        $MYPY_PINS || exit 2
fi

status=0

check() {
    # $1 = directory to run from (relative to repo root), rest = mypy targets
    dir="$1"
    shift
    echo "--- mypy $dir: $*"
    # shellcheck disable=SC2086 # $MYPY_ARGS is a deliberate word list
    (cd "$REPO_ROOT/$dir" && mypy "$@" $MYPY_ARGS) || status=1
}

# The three FastAPI services: `app` is the root package inside each.
check services/ai-engine app
check services/backend-core app
check services/browser-engine app

# web-ui is a Streamlit app: its code lives in `src`, entered through app.py.
check services/web-ui src app.py

# Modules shared by every service, plus the underscore compatibility shims
# (services/ai_engine/ etc.) that tests and scripts/cli.py import through.
check . \
    services/__init__.py \
    services/cors.py \
    services/observability.py \
    services/url_safety.py \
    services/ai_engine/__init__.py \
    services/backend_core/__init__.py \
    services/browser_engine/__init__.py \
    services/web_ui/__init__.py

if [ "$status" -eq 0 ]; then
    echo "--- mypy: all services clean"
fi
exit "$status"
