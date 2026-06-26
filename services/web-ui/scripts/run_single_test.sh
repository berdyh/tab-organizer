#!/bin/bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <pytest-args>"
    echo "Example: $0 tests/unit/test_web_ui_platform.py -q"
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$REPO_ROOT"
exec docker compose --profile test-unit run --rm test-unit pytest "$@"
