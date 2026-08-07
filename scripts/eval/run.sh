#!/usr/bin/env bash
# Regenerate the G4 clustering evaluation end to end.
#
#   scripts/eval/run.sh [work_dir]
#
# Stages 1, 3 and 4 run inside the test-unit image, which is the only place
# umap-learn and hdbscan are installed -- decision 48: a clustering run in an
# image without them silently measures SVD + k-means. Stage 2 runs on the host
# because ollama is published on 127.0.0.1 only, so containers in this compose
# project cannot reach it.
#
# The work dir holds the 20 Newsgroups download, the corpora, and the embedding
# cache. Keep it OUTSIDE the repo: vectors are not committed.
set -euo pipefail

WORK="${1:-/tmp/g4-clustering}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
mkdir -p "$WORK"

DC=(docker compose --profile test-unit run --rm --no-deps -v "$WORK:/g4" test-unit)

echo "== stage 1: corpora =="
"${DC[@]}" python scripts/eval/g4_corpus.py --work /g4

echo "== stage 2: embeddings (host -> local ollama) =="
python3 scripts/eval/g4_embed.py --work "$WORK"

echo "== stage 3: arms =="
"${DC[@]}" python scripts/eval/g4_arms.py --work /g4

echo "== stage 4: tables =="
"${DC[@]}" python scripts/eval/g4_report.py --work /g4
