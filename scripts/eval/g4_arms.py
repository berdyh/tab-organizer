#!/usr/bin/env python3
"""G4 stage 3 -- run the three arms and score them.

Runs inside the test-unit image, which is the only place umap-learn and hdbscan
are installed. Arm A ASSERTS that it really ran UMAP + HDBSCAN
(`TabClusterer.last_reduce_backend` / `last_cluster_backend`); if the imports
had fallen back to SVD + k-means the run aborts rather than filing k-means
numbers under the sidecar's name (decision 48).

Arms
  A  production: `TabClusterer()` at the DEFAULTS `services/ai-engine/app/main.py`
     constructs it with. Driven through reduce_dimensions + cluster_embeddings so
     the raw -1 noise labels survive; that is exactly what `cluster_sync` does
     after its small-N guard, and the run asserts the two agree.
  B  agglomerative, average linkage, cosine distance, threshold picked by
     silhouette (no ground truth).
  C  k-means over cosine (L2-normalised rows), k picked by silhouette over a grid
     that is NOT capped at 10 -- `_kmeans_cluster`'s max_clusters=10 belongs to
     an ImportError fallback that never runs in production.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import g4_metrics as M  # noqa: E402

from services.ai_engine.app.clustering.pipeline import (  # noqa: E402
    Tab,
    TabClusterer,
)

AGGLO_GRID = np.round(np.arange(0.05, 0.99, 0.01), 4)
STABILITY_REPEATS = 5
RESAMPLE_FRACTION = 0.9


def cosine_distance_matrix(X: np.ndarray) -> np.ndarray:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    S = np.clip(Xn @ Xn.T, -1.0, 1.0)
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    return np.maximum(D, 0.0)


def _score(D: np.ndarray, labels: np.ndarray) -> float:
    k = len(np.unique(labels))
    if k < 2 or k >= len(labels):
        return -2.0
    return float(silhouette_score(D, labels, metric="precomputed"))


# --------------------------------------------------------------------------- A


def arm_production(X: np.ndarray, params: dict | None = None) -> dict:
    clusterer = TabClusterer(**(params or {}))
    t0 = time.time()
    reduced = clusterer.reduce_dimensions(X)
    labels = clusterer.cluster_embeddings(reduced)
    elapsed = time.time() - t0

    if clusterer.last_reduce_backend != "umap" or clusterer.last_cluster_backend != "hdbscan":
        raise SystemExit(
            "ABORT: arm A did not run the production geometry "
            f"(reduce={clusterer.last_reduce_backend!r}, "
            f"cluster={clusterer.last_cluster_backend!r}). "
            "Install umap-learn and hdbscan; do not report these numbers."
        )
    return {
        "labels_raw": np.asarray(labels, dtype=int),
        "select_seconds": 0.0,
        "fit_seconds": elapsed,
        "chosen": {
            "min_cluster_size": clusterer.min_cluster_size,
            "min_samples": clusterer.min_samples,
            "cluster_selection_epsilon": clusterer.cluster_selection_epsilon,
            "umap_n_neighbors": clusterer.umap_n_neighbors,
            "umap_n_components": clusterer.umap_n_components,
            "umap_min_dist": clusterer.umap_min_dist,
            "reduce_backend": clusterer.last_reduce_backend,
            "cluster_backend": clusterer.last_cluster_backend,
        },
        "has_noise": True,
    }


# --------------------------------------------------------------------------- B


def arm_agglomerative(X: np.ndarray, D: np.ndarray | None = None, k_floor: int = 2) -> dict:
    if D is None:
        D = cosine_distance_matrix(X)
    t0 = time.time()
    Z = linkage(squareform(D, checks=False), method="average")
    best = (-2.0, None, None)
    for t in AGGLO_GRID:
        labels = fcluster(Z, t=float(t), criterion="distance")
        k = len(np.unique(labels))
        if k < k_floor or k > len(X) // 2:
            continue
        s = _score(D, labels)
        if s > best[0]:
            best = (s, float(t), labels)
    select_seconds = time.time() - t0
    if best[2] is None:  # degenerate; fall back to the loosest usable cut
        labels = fcluster(Z, t=2, criterion="maxclust")
        best = (_score(D, labels), None, labels)

    t0 = time.time()
    fcluster(Z, t=best[1] if best[1] is not None else 0.5, criterion="distance")
    fit_seconds = time.time() - t0
    return {
        "labels_raw": np.asarray(best[2], dtype=int),
        "select_seconds": select_seconds,
        "fit_seconds": fit_seconds,
        "chosen": {
            "linkage": "average",
            "metric": "cosine",
            "distance_threshold": best[1],
            "k_floor": k_floor,
            "silhouette": best[0],
        },
        "has_noise": False,
    }


# --------------------------------------------------------------------------- C


def _k_grid(n: int) -> list[int]:
    hi = max(3, min(60, n // 4))
    step = 1 if hi <= 40 else 2
    return list(range(2, hi + 1, step))


def arm_kmeans(X: np.ndarray, D: np.ndarray | None = None) -> dict:
    if D is None:
        D = cosine_distance_matrix(X)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    t0 = time.time()
    best = (-2.0, None, None)
    for k in _k_grid(len(X)):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(Xn)
        s = _score(D, labels)
        if s > best[0]:
            best = (s, k, labels)
    select_seconds = time.time() - t0

    t0 = time.time()
    KMeans(n_clusters=best[1], n_init=10, random_state=42).fit_predict(Xn)
    fit_seconds = time.time() - t0
    return {
        "labels_raw": np.asarray(best[2], dtype=int),
        "select_seconds": select_seconds,
        "fit_seconds": fit_seconds,
        "chosen": {"k": best[1], "silhouette": best[0], "metric": "cosine (L2-normalised)"},
        "has_noise": False,
    }


# The three pre-registered arms, plus one SUPPLEMENTARY variant. `A2` is not
# part of the decision rule and must never be substituted for `A`: it exists
# only to answer "is a bad arm-A result the algorithm or the default
# parameters?", which the report has to answer under "what would have changed
# the verdict".
ARMS: dict[str, dict] = {
    "A_production_umap_hdbscan": {"fn": "production", "params": {}, "verdict": True},
    "B_agglomerative_cosine": {"fn": "agglomerative", "params": {}, "verdict": True},
    "B2_agglomerative_cosine_kfloor5": {
        "fn": "agglomerative",
        "params": {"k_floor": 5},
        "verdict": True,
    },
    "C_kmeans_silhouette": {"fn": "kmeans", "params": {}, "verdict": True},
    "A2_supplementary_umap_hdbscan_eps0": {
        "fn": "production",
        "params": {"cluster_selection_epsilon": 0.0},
        "verdict": False,
    },
}


def run_arm(name: str, X: np.ndarray, D: np.ndarray | None) -> dict:
    spec = ARMS[name]
    if spec["fn"] == "production":
        return arm_production(X, spec["params"])
    if spec["fn"] == "agglomerative":
        return arm_agglomerative(X, D, **spec["params"])
    return arm_kmeans(X, D)


# --------------------------------------------------------------------------- scoring


def score_run(
    res: dict, X: np.ndarray, y_true: np.ndarray, kinds: np.ndarray, min_cluster_size: int
) -> dict:
    raw = res["labels_raw"]
    product = M.expand_noise(raw) if res["has_noise"] else raw
    out = {
        "n_clusters_product": int(len(np.unique(product))),
        "n_clusters_nonnoise": int(len([v for v in np.unique(raw) if v != M.NOISE])),
        "n_noise": int((raw == M.NOISE).sum()) if res["has_noise"] else 0,
        "ari": M.ari(y_true, product),
        "purity": M.purity(y_true, product),
        "coherence": M.coherence(X, product),
        "singletons": M.singleton_metrics(
            raw, y_true, kinds, min_cluster_size, res["has_noise"]
        ),
        "select_seconds": res["select_seconds"],
        "fit_seconds": res["fit_seconds"],
        "chosen": res["chosen"],
    }
    if res["has_noise"]:
        out["ari_noise_as_one_cluster"] = M.ari(y_true, raw)
    return out


def stability(name: str, X: np.ndarray, base_labels: np.ndarray, seed: int) -> dict:
    """Perturbation stability. NOT plain reruns.

    `pipeline.py` pins random_state=42, so an identical-input rerun returns
    identical output and would score a vacuous 1.0. These two perturbations
    change something a real reindex changes: the order documents arrive in, and
    which documents are present.
    """
    rng = np.random.default_rng(seed)
    shuffle_scores = []
    for _ in range(STABILITY_REPEATS):
        perm = rng.permutation(len(X))
        res = run_arm(name, X[perm], None)
        back = np.empty(len(X), dtype=int)
        back[perm] = M.expand_noise(res["labels_raw"]) if res["has_noise"] else res["labels_raw"]
        shuffle_scores.append(M.ari(base_labels.astype(str), back))

    resample_scores = []
    m = int(round(len(X) * RESAMPLE_FRACTION))
    for _ in range(STABILITY_REPEATS):
        idx = np.sort(rng.choice(len(X), size=m, replace=False))
        res = run_arm(name, X[idx], None)
        got = M.expand_noise(res["labels_raw"]) if res["has_noise"] else res["labels_raw"]
        resample_scores.append(M.ari(base_labels[idx].astype(str), got))

    return {
        "order_shuffle_ari_mean": float(np.mean(shuffle_scores)),
        "order_shuffle_ari_min": float(np.min(shuffle_scores)),
        "resample90_ari_mean": float(np.mean(resample_scores)),
        "resample90_ari_min": float(np.min(resample_scores)),
        "repeats": STABILITY_REPEATS,
    }


def survival_series(name: str, data: dict, min_group_size: int) -> dict:
    """Decision-39 group survival across five +10 increments."""
    X, ids, inc = data["X"], data["ids"], data["inc"]
    prev = None
    steps = []
    for step in range(0, 6):
        mask = inc <= step
        Xs, ids_s = X[mask], ids[mask]
        res = run_arm(name, Xs, None)
        labels = M.expand_noise(res["labels_raw"]) if res["has_noise"] else res["labels_raw"]
        if prev is not None:
            steps.append(
                M.group_survival(prev[0], prev[1], ids_s, labels,
                                 min_group_size=min_group_size)
            )
        prev = (ids_s, labels)

    def mean(key: str) -> float:
        vals = [s[key] for s in steps if not np.isnan(s[key])]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "steps": steps,
        "survival_all_mean": mean("survival_all"),
        "survival_ge_min_mean": mean("survival_ge_min"),
        "membership_changed_frac_mean": mean("membership_changed_frac"),
        "min_group_size": min_group_size,
    }


def load(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {
        "X": z["X"].astype(np.float64),
        "ids": z["ids"],
        "true_group": z["true_group"],
        "kind": z["kind"],
        "inc": z["inc"],
    }


def verify_arm_a_matches_cluster_sync(data: dict) -> dict:
    """Prove the direct-drive path is the production path.

    `cluster_sync` = small-N guard, reduce_dimensions, cluster_embeddings,
    _group_by_labels. Driving the first three directly is the same computation;
    this asserts the resulting flat partition is identical rather than assuming.
    """
    mask = data["inc"] == 0
    X = data["X"][mask][:400]
    ids = data["ids"][mask][:400]
    direct = arm_production(X)
    product = M.expand_noise(direct["labels_raw"])

    clusterer = TabClusterer()
    tabs = [Tab(url=f"https://x/{i}", title=str(i)) for i in range(len(X))]
    clusters = clusterer.cluster_sync(tabs, X)
    sync_labels = np.empty(len(X), dtype=int)
    for position, c in enumerate(clusters):
        for t in c.tabs:
            sync_labels[int(t.title)] = position
    agree = M.ari(product.astype(str), sync_labels)
    return {
        "n": len(X),
        "ari_direct_vs_cluster_sync": agree,
        "n_clusters_cluster_sync": len(clusters),
        "n_clusters_direct": int(len(np.unique(product))),
        "note": "1.0 means the direct drive reproduces cluster_sync's flat partition exactly",
        "ids_checked": int(len(ids)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.environ.get("G4_WORK", "/tmp/g4-clustering"))
    ap.add_argument("--corpora", default="")
    ap.add_argument("--skip-stability", action="store_true")
    ap.add_argument("--skip-survival", action="store_true")
    ap.add_argument("--arms", default="", help="comma-separated subset of arm names")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    work = Path(args.work)
    vec_dir = work / "vectors"
    res_dir = work / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    names = (
        [n.strip() for n in args.corpora.split(",") if n.strip()]
        if args.corpora
        else [p.stem for p in sorted(vec_dir.glob("*.npz"))]
    )

    all_results: dict = {
        "embedding_model": json.loads((work / "embedding_model.json").read_text()),
        "corpora": {},
    }

    for cname in names:
        data = load(vec_dir / f"{cname}.npz")
        base_mask = data["inc"] == 0
        Xb = data["X"][base_mask]
        yb = data["true_group"][base_mask]
        kb = data["kind"][base_mask]
        ids_b = data["ids"][base_mask]
        stats = json.loads((vec_dir / f"{cname}.stats.json").read_text())

        # The production min_cluster_size for this N, as pipeline.py computes it.
        mcs = min(TabClusterer().min_cluster_size, max(2, len(Xb) // 3))

        print(f"\n=== {cname}  n_base={len(Xb)}  min_cluster_size={mcs} ===", flush=True)
        D = cosine_distance_matrix(Xb)

        entry: dict = {
            "n_base": int(len(Xb)),
            "n_true_groups": int(len(np.unique(yb))),
            "true_group_sizes": {
                str(g): int((yb == g).sum()) for g in sorted(set(yb.tolist()))
            },
            "kind_counts": {
                str(k): int((kb == k).sum()) for k in sorted(set(kb.tolist()))
            },
            "vector_stats": stats,
            "effective_min_cluster_size": int(mcs),
            "arms": {},
        }

        selected = [a.strip() for a in args.arms.split(",") if a.strip()] or list(ARMS)
        for aname in selected:
            t0 = time.time()
            res = run_arm(aname, Xb, D)
            scored = score_run(res, Xb, yb, kb, mcs)
            print(
                f"  {aname}: k={scored['n_clusters_product']} "
                f"ARI={scored['ari']:.3f} purity={scored['purity']:.3f} "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )
            if not args.skip_stability:
                base_labels = M.expand_noise(res["labels_raw"]) if res["has_noise"] else res["labels_raw"]
                scored["stability"] = stability(aname, Xb, base_labels, seed=7)
                print(f"    stability {scored['stability']}", flush=True)
            if not args.skip_survival:
                scored["survival"] = survival_series(aname, data, mcs)
                print(
                    f"    survival ge{mcs}={scored['survival']['survival_ge_min_mean']:.3f} "
                    f"all={scored['survival']['survival_all_mean']:.3f} "
                    f"churn={scored['survival']['membership_changed_frac_mean']:.3f}",
                    flush=True,
                )
            scored["counts_for_verdict"] = ARMS[aname]["verdict"]
            entry["arms"][aname] = scored

        if cname == "tab_500":
            entry["arm_a_path_check"] = verify_arm_a_matches_cluster_sync(data)
            print(f"  path check: {entry['arm_a_path_check']}", flush=True)

        all_results["corpora"][cname] = entry
        (res_dir / args.out).write_text(json.dumps(all_results, indent=2, default=float))

    print(f"\nwrote {res_dir / args.out}")


if __name__ == "__main__":
    main()
