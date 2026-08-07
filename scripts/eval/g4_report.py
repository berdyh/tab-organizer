#!/usr/bin/env python3
"""G4 stage 4 -- render the metric tables from results.json.

Emits markdown only. The verdict prose in docs/EVAL-clustering.md is written by
hand; this exists so the numbers in that document can be regenerated and
diffed rather than retyped.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ARM_LABEL = {
    "A_production_umap_hdbscan": "A production (UMAP+HDBSCAN)",
    "B_agglomerative_cosine": "B agglomerative (cosine, avg)",
    "B2_agglomerative_cosine_kfloor5": "B2 agglomerative (cosine, avg, k>=5)",
    "C_kmeans_silhouette": "C k-means (silhouette k)",
    "A2_supplementary_umap_hdbscan_eps0": "A2 supplementary (eps=0, NOT in verdict)",
}


def f(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float) and x != x:  # NaN
        return "n/a"
    return f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.environ.get("G4_WORK", "/tmp/g4-clustering"))
    ap.add_argument(
        "--results",
        default="results.json",
        help="comma-separated result files under <work>/results, merged per corpus",
    )
    args = ap.parse_args()

    d: dict = {"corpora": {}}
    for fname in args.results.split(","):
        part = json.loads((Path(args.work) / "results" / fname.strip()).read_text())
        d.setdefault("embedding_model", part.get("embedding_model"))
        for cname, entry in part["corpora"].items():
            cur = d["corpora"].setdefault(cname, {**entry, "arms": {}})
            cur.update({k: v for k, v in entry.items() if k != "arms"})
            cur["arms"].update(entry["arms"])
    order = [c for c in ("tab_100", "tab_500", "tab_2000") if c in d["corpora"]]
    order += [c for c in d["corpora"] if c not in order]

    arm_order = [
        "A_production_umap_hdbscan",
        "B_agglomerative_cosine",
        "B2_agglomerative_cosine_kfloor5",
        "C_kmeans_silhouette",
        "A2_supplementary_umap_hdbscan_eps0",
    ]
    for c in d["corpora"].values():
        c["arms"] = {
            k: c["arms"][k]
            for k in arm_order + [a for a in c["arms"] if a not in arm_order]
            if k in c["arms"]
        }

    print("### Quality\n")
    print("| corpus | n | arm | groups shown | non-noise clusters | noise | ARI | purity | intra cos | inter cos | separation |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in order:
        e = d["corpora"][c]
        for a, v in e["arms"].items():
            print(
                f"| {c} | {e['n_base']} | {ARM_LABEL.get(a, a)} | {v['n_clusters_product']} "
                f"| {v['n_clusters_nonnoise']} | {v['n_noise']} | {f(v['ari'])} | {f(v['purity'])} "
                f"| {f(v['coherence']['intra'])} | {f(v['coherence']['inter'])} "
                f"| {f(v['coherence']['separation'])} |"
            )

    print("\n### Singletons and orphans\n")
    print("| corpus | arm | singletons shown | singleton precision | singleton recall | orphan rate | orphans / eligible |")
    print("|---|---|---|---|---|---|---|")
    for c in order:
        e = d["corpora"][c]
        for a, v in e["arms"].items():
            s = v["singletons"]
            print(
                f"| {c} | {ARM_LABEL.get(a, a)} | {s['n_singletons']} "
                f"| {f(s['singleton_precision'])} | {f(s['singleton_recall'])} "
                f"| {f(s['orphan_rate'])} | {s['n_orphans']}/{s['n_eligible']} |"
            )

    print("\n### Decision-39 group survival (+10 docs x 5)\n")
    print("| corpus | arm | survival (groups >= min_cluster_size) | survival (all rows) | membership churn |")
    print("|---|---|---|---|---|")
    for c in order:
        e = d["corpora"][c]
        for a, v in e["arms"].items():
            s = v.get("survival")
            if not s:
                continue
            print(
                f"| {c} | {ARM_LABEL.get(a, a)} | {f(s['survival_ge_min_mean'])} "
                f"| {f(s['survival_all_mean'])} | {f(s['membership_changed_frac_mean'])} |"
            )

    print("\n### Perturbation stability (ARI vs base run, 5 repeats)\n")
    print("| corpus | arm | order-shuffle mean | order-shuffle min | 90% resample mean | 90% resample min |")
    print("|---|---|---|---|---|---|")
    for c in order:
        e = d["corpora"][c]
        for a, v in e["arms"].items():
            s = v.get("stability")
            if not s:
                continue
            print(
                f"| {c} | {ARM_LABEL.get(a, a)} | {f(s['order_shuffle_ari_mean'])} "
                f"| {f(s['order_shuffle_ari_min'])} | {f(s['resample90_ari_mean'])} "
                f"| {f(s['resample90_ari_min'])} |"
            )

    print("\n### Wall time (single fit on the base corpus)\n")
    print("| corpus | arm | parameter search (s) | final fit (s) | chosen parameters |")
    print("|---|---|---|---|---|")
    for c in order:
        e = d["corpora"][c]
        for a, v in e["arms"].items():
            ch = {k: val for k, val in v["chosen"].items() if k != "silhouette"}
            print(
                f"| {c} | {ARM_LABEL.get(a, a)} | {f(v['select_seconds'], 1)} "
                f"| {f(v['fit_seconds'], 1)} | `{json.dumps(ch)}` |"
            )


if __name__ == "__main__":
    main()
