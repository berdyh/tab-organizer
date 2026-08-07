#!/usr/bin/env python3
"""Diagnostic: how many usable documents each newsgroup has after cleaning.

Used once to pick the seven tab-shaped groups and their power-law ranks; kept so
the choice is reproducible rather than asserted.
"""

import argparse
import os

from sklearn.datasets import fetch_20newsgroups

MIN_DOC_CHARS = 300  # keep in sync with g4_corpus.MIN_DOC_CHARS

ap = argparse.ArgumentParser()
ap.add_argument("--work", default=os.environ.get("G4_WORK", "/tmp/g4-clustering"))
args = ap.parse_args()

bunch = fetch_20newsgroups(
    subset="all",
    remove=("headers", "footers", "quotes"),
    data_home=os.path.join(args.work, "sklearn_data"),
    shuffle=False,
)
counts: dict[str, int] = {}
for text, target in zip(bunch.data, bunch.target):
    if len(text.strip()) >= MIN_DOC_CHARS:
        name = bunch.target_names[target]
        counts[name] = counts.get(name, 0) + 1

for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"{n:5d}  {name}")
