#!/usr/bin/env python3
"""G4 metric definitions. Kept separate so each one can be read on its own."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score

NOISE = -1


def encode(labels: np.ndarray) -> np.ndarray:
    uniq = {v: i for i, v in enumerate(sorted(set(labels.tolist())))}
    return np.array([uniq[v] for v in labels.tolist()], dtype=int)


def expand_noise(labels: np.ndarray) -> np.ndarray:
    """Turn every -1 into its own cluster.

    This is what the product actually shows: `pipeline.py::_group_by_labels`
    emits one "Uncategorized" cluster row per noise point. Scoring the raw -1s
    as a single giant cluster would score a partition no user ever sees.
    """
    out = labels.astype(int).copy()
    nxt = int(out.max()) + 1 if out.size and out.max() >= 0 else 0
    for i in np.where(out == NOISE)[0]:
        out[i] = nxt
        nxt += 1
    return out


def ari(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(adjusted_rand_score(encode(y_true), y_pred))


def purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of documents in the majority true class of their cluster.

    Read alongside the cluster count: a partition of N singletons has purity
    1.0 and is worthless, which is why this is never reported on its own.
    """
    total = 0
    for c in np.unique(y_pred):
        members = y_true[y_pred == c]
        vals, counts = np.unique(members, return_counts=True)
        total += int(counts.max())
    return total / len(y_true)


def coherence(X: np.ndarray, y_pred: np.ndarray) -> dict:
    """Mean intra- vs inter-cluster cosine similarity on the raw doc vectors.

    Computed on the same 768-d vectors for every arm, so a UMAP-reduced arm
    gets no advantage from being scored in its own reduced space.
    """
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    S = Xn @ Xn.T
    n = len(y_pred)
    same = y_pred[:, None] == y_pred[None, :]
    off = ~np.eye(n, dtype=bool)
    intra = S[same & off]
    inter = S[~same & off]
    a = float(intra.mean()) if intra.size else float("nan")
    b = float(inter.mean()) if inter.size else float("nan")
    return {"intra": a, "inter": b, "separation": a - b}


def singleton_set(labels_raw: np.ndarray, treat_noise_as_singleton: bool) -> set:
    """Documents the product would show alone.

    For the production arm that means HDBSCAN's -1 points (each becomes its own
    "Uncategorized" row). For the TypeScript-candidate arms, which have no noise
    label, it means clusters that ended up with exactly one member.
    """
    out: set = set()
    if treat_noise_as_singleton:
        out |= set(np.where(labels_raw == NOISE)[0].tolist())
    vals, counts = np.unique(labels_raw, return_counts=True)
    for v, c in zip(vals, counts):
        if v == NOISE:
            continue
        if c == 1:
            out |= set(np.where(labels_raw == v)[0].tolist())
    return out


def singleton_metrics(
    labels_raw: np.ndarray,
    y_true: np.ndarray,
    kinds: np.ndarray,
    min_cluster_size: int,
    treat_noise_as_singleton: bool,
) -> dict:
    singles = singleton_set(labels_raw, treat_noise_as_singleton)
    oneoffs = set(np.where(kinds == "outlier")[0].tolist())

    hit = len(singles & oneoffs)
    precision = hit / len(singles) if singles else None
    recall = hit / len(oneoffs) if oneoffs else None

    # Orphan rate: documents whose true group DOES have enough members present
    # to form a cluster, but which were shown alone anyway.
    vals, counts = np.unique(y_true, return_counts=True)
    big = {v for v, c in zip(vals, counts) if c >= min_cluster_size}
    eligible = [i for i in range(len(y_true)) if y_true[i] in big]
    orphans = [i for i in eligible if i in singles]
    return {
        "n_singletons": len(singles),
        "singleton_precision": precision,
        "singleton_recall": recall,
        "orphan_rate": (len(orphans) / len(eligible)) if eligible else None,
        "n_orphans": len(orphans),
        "n_eligible": len(eligible),
    }


def _members(ids: np.ndarray, labels: np.ndarray) -> dict:
    out: dict = {}
    for i, lab in enumerate(labels.tolist()):
        out.setdefault(lab, set()).add(ids[i])
    return out


def group_survival(
    ids_t: np.ndarray,
    labels_t: np.ndarray,
    ids_t1: np.ndarray,
    labels_t1: np.ndarray,
    jaccard_threshold: float = 0.5,
    min_group_size: int = 3,
) -> dict:
    """Decision 39's own rule: greedy one-to-one match at Jaccard >= 0.5.

    Reported twice. "all" counts every produced row including the one-document
    "Uncategorized" ones, which trivially match themselves and inflate the
    number. "ge{min_group_size}" counts only groups big enough to carry a user
    relabel or a pin, which is the thing decision 39 exists to protect.
    """
    A = _members(ids_t, labels_t)
    B = _members(ids_t1, labels_t1)

    pairs = []
    for ka, sa in A.items():
        for kb, sb in B.items():
            inter = len(sa & sb)
            if not inter:
                continue
            j = inter / len(sa | sb)
            if j >= jaccard_threshold:
                pairs.append((j, ka, kb))
    pairs.sort(key=lambda p: -p[0])

    used_a: set = set()
    used_b: set = set()
    match: dict = {}
    for j, ka, kb in pairs:
        if ka in used_a or kb in used_b:
            continue
        used_a.add(ka)
        used_b.add(kb)
        match[ka] = kb

    def frac(keys) -> float:
        keys = list(keys)
        if not keys:
            return float("nan")
        return sum(1 for k in keys if k in match) / len(keys)

    big_a = [k for k, s in A.items() if len(s) >= min_group_size]

    # Membership churn, over the documents present in BOTH runs.
    pos_t1 = {doc: labels_t1[i] for i, doc in enumerate(ids_t1.tolist())}
    common = [i for i, doc in enumerate(ids_t.tolist()) if doc in pos_t1]
    changed = 0
    for i in common:
        expected = match.get(labels_t[i])
        if expected is None or pos_t1[ids_t[i]] != expected:
            changed += 1

    return {
        "n_groups_t": len(A),
        "n_groups_t1": len(B),
        "n_groups_t_ge_min": len(big_a),
        "survival_all": frac(A.keys()),
        "survival_ge_min": frac(big_a),
        "membership_changed_frac": changed / len(common) if common else float("nan"),
        "n_common_docs": len(common),
    }
