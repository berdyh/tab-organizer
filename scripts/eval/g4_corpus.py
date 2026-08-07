#!/usr/bin/env python3
"""G4 stage 1 -- build the evaluation corpora.

Writes tab-shaped corpora (the verdict corpora) and one vanilla 20 Newsgroups
corpus (calibration only) as JSON under ``<work>/corpus/``.

Needs scikit-learn, so this stage runs inside the test-unit image. It does not
need umap/hdbscan and does not touch ollama.

Corpus shape is the one pre-registered in the 2026-08-07 addendum to
docs/ARCHITECTURE_PLAN.md: power-law group sizes, injected near-duplicates from
one "domain", injected title-only navigational stubs, ~10 one-off outliers drawn
from held-out groups, and ``remove=('headers','footers','quotes')`` so that no
arm can score purity on a hostname or a signature block.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path

from sklearn.datasets import fetch_20newsgroups

SEED = 20260807

# Seven topically separable groups. Rank order below is the power-law rank:
# group 0 is the "big domain" the user has 800 tabs from. Ordered by usable pool
# size descending (see g4_pool_sizes.py) so the heaviest rank never runs the
# 20 Newsgroups pool dry at n=2000.
TAB_GROUPS = [
    "sci.med",  # 729 usable
    "talk.politics.mideast",  # 725
    "sci.space",  # 681
    "rec.sport.hockey",  # 668
    "rec.autos",  # 602
    "comp.sys.mac.hardware",  # 600
    "comp.graphics",  # 586
]

# Never appear as a group -- only as one-off outliers, one document each.
HELDOUT_GROUPS = [
    "soc.religion.christian",
    "misc.forsale",
    "talk.religion.misc",
]

# Vanilla calibration corpus: balanced, equal-sized, no injections.
VANILLA_GROUPS = [
    "comp.sys.mac.hardware",
    "rec.sport.hockey",
    "sci.space",
    "talk.politics.mideast",
    "comp.graphics",
    "rec.autos",
]

POWER_LAW_EXPONENT = 1.1
NEAR_DUP_FRACTION = 0.25  # of the two largest groups
NEAR_DUP_GROUP_RANKS = (0, 1)
# Near-duplicates are boilerplate variants of a handful of seed pages rather
# than of one single page: "many tabs open on one docs site" is several pages
# each opened repeatedly, and 200 copies of a single vector would be a
# degenerate blob no clusterer could get wrong.
NEAR_DUP_SEEDS = 8
STUB_FRACTION = 0.08  # of all base documents
N_OUTLIERS = 10
N_INCREMENTS = 5
INCREMENT_SIZE = 10
MIN_DOC_CHARS = 300

# Boilerplate variants injected around near-duplicates. Deliberately generic:
# nothing here names a topic, so a near-duplicate stays near its source group
# rather than forming a "boilerplate" cluster of its own.
BOILERPLATE = [
    "Skip to main content. Sign in. Search this site.",
    "Home > Docs > Reference. Edit this page. Report an issue.",
    "Menu Toggle navigation Newsletter Subscribe Contact us",
    "On this page: Overview, Usage, Notes, See also.",
    "Cookie notice: we use cookies. Accept all. Manage preferences.",
    "Version 4.2 (latest). Other versions: 4.1, 4.0, 3.9.",
    "Last updated 3 days ago by the docs team. Was this helpful?",
    "Table of contents. Previous page. Next page. Back to top.",
]

FOOTERS = [
    "Copyright 2026. All rights reserved. Terms Privacy Status.",
    "Built with love. Community forum. Changelog. RSS.",
    "Questions? Open a discussion. Star us on the tracker.",
    "Page 1 of 4. Continue reading.",
    "Page 2 of 4. Continue reading.",
    "Related pages below. Suggested for you.",
]


def _slug(text: str, limit: int = 48) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s[:limit] or "page").strip("-")


def _title_of(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if len(line) >= 12:
            return re.sub(r"\s+", " ", line)[:90]
    return re.sub(r"\s+", " ", text)[:90] or "Untitled"


def _host_for(group: str) -> str:
    # Synthetic hostname. Never fed to any arm -- all arms see vectors only.
    return _slug(group.replace(".", "-"), 40) + ".example.net"


def _power_law_sizes(n_total: int, n_groups: int, exponent: float) -> list[int]:
    weights = [(i + 1) ** (-exponent) for i in range(n_groups)]
    total_w = sum(weights)
    raw = [n_total * w / total_w for w in weights]
    sizes = [max(1, int(round(x))) for x in raw]
    # Fix rounding drift against the largest group.
    drift = n_total - sum(sizes)
    sizes[0] += drift
    return sizes


class _Pool:
    """Deterministic, non-repeating draw of documents per newsgroup."""

    def __init__(self, data_home: str, groups: list[str], rng: random.Random):
        bunch = fetch_20newsgroups(
            subset="all",
            categories=groups,
            remove=("headers", "footers", "quotes"),
            data_home=data_home,
            shuffle=False,
        )
        self._by_group: dict[str, list[str]] = {g: [] for g in groups}
        for text, target in zip(bunch.data, bunch.target):
            name = bunch.target_names[target]
            text = text.strip()
            if len(text) < MIN_DOC_CHARS:
                continue
            self._by_group[name].append(text)
        for g in groups:
            rng.shuffle(self._by_group[g])
        self._cursor = {g: 0 for g in groups}

    def available(self, group: str) -> int:
        return len(self._by_group[group]) - self._cursor[group]

    def take(self, group: str) -> str:
        i = self._cursor[group]
        if i >= len(self._by_group[group]):
            raise RuntimeError(f"pool exhausted for {group}")
        self._cursor[group] += 1
        return self._by_group[group][i]

    def peek_first(self, group: str) -> str:
        return self._by_group[group][self._cursor[group]]


def _make_doc(
    doc_id: str,
    group_label: str,
    host: str,
    text: str,
    kind: str,
    inc: int,
) -> dict:
    title = _title_of(text)
    return {
        "id": doc_id,
        "true_group": group_label,
        "kind": kind,
        "inc": inc,
        "url": f"https://{host}/{_slug(title)}-{doc_id[-6:]}",
        "title": title,
        "text": text,
    }


def build_tab_corpus(pool_home: str, n_base: int) -> dict:
    rng = random.Random(SEED + n_base)
    pool = _Pool(pool_home, TAB_GROUPS + HELDOUT_GROUPS, rng)

    n_group_docs = n_base - N_OUTLIERS
    sizes = _power_law_sizes(n_group_docs, len(TAB_GROUPS), POWER_LAW_EXPONENT)

    docs: list[dict] = []
    seq = 0

    def next_id() -> str:
        nonlocal seq
        seq += 1
        return hashlib.sha256(f"{n_base}:{seq}".encode()).hexdigest()[:16]

    for rank, (group, size) in enumerate(zip(TAB_GROUPS, sizes)):
        host = _host_for(group)
        n_dup = int(round(size * NEAR_DUP_FRACTION)) if rank in NEAR_DUP_GROUP_RANKS else 0
        n_dup = min(n_dup, max(0, size - 1))
        n_orig = size - n_dup

        seed_texts = [pool.peek_first(group)] if n_dup else []
        originals = [pool.take(group) for _ in range(n_orig)]
        if n_dup:
            seed_texts = originals[:NEAR_DUP_SEEDS] or seed_texts
        for text in originals:
            docs.append(_make_doc(next_id(), group, host, text, "base", 0))
        for j in range(n_dup):
            head = BOILERPLATE[j % len(BOILERPLATE)]
            foot = FOOTERS[j % len(FOOTERS)]
            body = seed_texts[j % len(seed_texts)]
            # Rotate a small window of the source so the variants are near-,
            # not exact-, duplicates.
            cut = (j * 137) % max(1, len(body) // 4)
            text = f"{head}\n\n{body[cut:]}\n\n{foot}"
            docs.append(_make_doc(next_id(), group, host, text, "neardup", 0))

    # Navigational stubs: title-only pages, drawn uniformly from the base docs.
    n_stub = int(round(len(docs) * STUB_FRACTION))
    base_idx = [i for i, d in enumerate(docs) if d["kind"] == "base"]
    rng.shuffle(base_idx)
    for i in base_idx[:n_stub]:
        docs[i]["text"] = docs[i]["title"]
        docs[i]["kind"] = "stub"

    # One-off outliers: one document per unique held-out label.
    for k in range(N_OUTLIERS):
        group = HELDOUT_GROUPS[k % len(HELDOUT_GROUPS)]
        label = f"oneoff::{k:02d}"
        docs.append(
            _make_doc(next_id(), label, f"oneoff{k:02d}.example.org",
                      pool.take(group), "outlier", 0)
        )

    rng.shuffle(docs)

    # Increments: +10 documents, five times, drawn from the same distribution.
    inc_weights = [(i + 1) ** (-POWER_LAW_EXPONENT) for i in range(len(TAB_GROUPS))]
    for inc in range(1, N_INCREMENTS + 1):
        for _ in range(INCREMENT_SIZE):
            group = rng.choices(TAB_GROUPS, weights=inc_weights, k=1)[0]
            if pool.available(group) == 0:
                group = max(TAB_GROUPS, key=pool.available)
            docs.append(
                _make_doc(next_id(), group, _host_for(group),
                          pool.take(group), "base", inc)
            )

    return {
        "name": f"tab_{n_base}",
        "shape": "tab-shaped",
        "n_base": n_base,
        "recipe": {
            "seed": SEED + n_base,
            "groups": TAB_GROUPS,
            "heldout_groups": HELDOUT_GROUPS,
            "power_law_exponent": POWER_LAW_EXPONENT,
            "group_sizes": sizes,
            "near_dup_fraction": NEAR_DUP_FRACTION,
            "near_dup_group_ranks": list(NEAR_DUP_GROUP_RANKS),
            "near_dup_seeds": NEAR_DUP_SEEDS,
            "stub_fraction": STUB_FRACTION,
            "n_outliers": N_OUTLIERS,
            "n_increments": N_INCREMENTS,
            "increment_size": INCREMENT_SIZE,
            "min_doc_chars": MIN_DOC_CHARS,
            "remove": ["headers", "footers", "quotes"],
        },
        "docs": docs,
    }


def build_vanilla_corpus(pool_home: str, n_base: int) -> dict:
    rng = random.Random(SEED + 999)
    pool = _Pool(pool_home, VANILLA_GROUPS, rng)
    per = n_base // len(VANILLA_GROUPS)
    docs: list[dict] = []
    seq = 0

    def next_id() -> str:
        nonlocal seq
        seq += 1
        return hashlib.sha256(f"vanilla:{seq}".encode()).hexdigest()[:16]

    for group in VANILLA_GROUPS:
        host = _host_for(group)
        for _ in range(per):
            docs.append(_make_doc(next_id(), group, host, pool.take(group), "base", 0))
    rng.shuffle(docs)

    for inc in range(1, N_INCREMENTS + 1):
        for _ in range(INCREMENT_SIZE):
            group = rng.choice(VANILLA_GROUPS)
            docs.append(
                _make_doc(next_id(), group, _host_for(group),
                          pool.take(group), "base", inc)
            )

    return {
        "name": f"vanilla_{per * len(VANILLA_GROUPS)}",
        "shape": "vanilla-20ng (calibration only)",
        "n_base": per * len(VANILLA_GROUPS),
        "recipe": {
            "seed": SEED + 999,
            "groups": VANILLA_GROUPS,
            "per_group": per,
            "injections": "none",
            "n_increments": N_INCREMENTS,
            "increment_size": INCREMENT_SIZE,
            "min_doc_chars": MIN_DOC_CHARS,
            "remove": ["headers", "footers", "quotes"],
        },
        "docs": docs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.environ.get("G4_WORK", "/tmp/g4-clustering"))
    ap.add_argument("--sizes", default="100,500,2000")
    ap.add_argument("--vanilla-size", type=int, default=498)
    args = ap.parse_args()

    work = Path(args.work)
    out = work / "corpus"
    out.mkdir(parents=True, exist_ok=True)
    data_home = str(work / "sklearn_data")

    for size in [int(s) for s in args.sizes.split(",")]:
        corpus = build_tab_corpus(data_home, size)
        path = out / f"{corpus['name']}.json"
        path.write_text(json.dumps(corpus))
        kinds: dict[str, int] = {}
        for d in corpus["docs"]:
            kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
        print(f"{path}  docs={len(corpus['docs'])}  kinds={kinds} "
              f"sizes={corpus['recipe']['group_sizes']}")

    corpus = build_vanilla_corpus(data_home, args.vanilla_size)
    path = out / f"{corpus['name']}.json"
    path.write_text(json.dumps(corpus))
    print(f"{path}  docs={len(corpus['docs'])}  (calibration)")


if __name__ == "__main__":
    main()
