# G4 — clustering evaluation: does the Python geometry sidecar earn its process?

**Verdict: DROP.**

Under the rule pre-registered in `docs/ARCHITECTURE_PLAN.md`
("Addendum 2026-08-07 — G4 clustering evaluation"), the Python UMAP+HDBSCAN
sidecar (decision 25) does not clear the bar. TypeScript can ship agglomerative
or k-means over cosine similarity.

The rule required the production pipeline to beat the best TypeScript-candidate
arm by **≥ 0.10 ARI or ≥ 10 points purity at BOTH 500 and 2000 docs**, and to be
**no worse on decision-39 group survival**. It cleared the ARI leg at 500
(+0.130) and missed it at 2000 (+0.067). It lost the purity leg at every size.
It lost the survival leg at both sizes against the best surviving TypeScript arm.
A single-size win is explicitly a drop under the rule, and this is not even a tie.

This is not close enough to need the tie-breaker, and it is not an artifact of a
weak comparator: the verdict is identical if you delete the two agglomerative
arms and compare only against k-means (see "Did the added arm change anything?").

---

## 1. What was run

| | |
|---|---|
| Embedding model | `nomic-embed-text:latest`, ollama, digest `0a109f422b47e3a3`, 137M params, F16 GGUF, `num_ctx 8192`, 768-d |
| Provider cost | **$0.00** — local ollama, no metered provider was called at any point |
| Document vector | **mean of chunk vectors** (schema fix E5), chunker copied verbatim from `services/ai-engine/app/chatbot/rag.py`: 4000 chars, 400 overlap, max 20 chunks |
| Vectors | embedded once, cached by `sha256(model ‖ chunk_text)` in a sqlite file outside the repo; every arm and every repeat reads byte-identical vectors |
| Labelling | **none**. No LLM was called. The geometry is driven directly (`reduce_dimensions` → `cluster_embeddings`), and that path is asserted to reproduce `cluster_sync`'s flat partition exactly (ARI = 1.000, 29 clusters both ways, n=400) |
| Algorithm assertion | every arm-A run checks `last_reduce_backend == "umap"` and `last_cluster_backend == "hdbscan"` and **aborts** otherwise (decision 48). All runs below passed |
| Runtime image | `docker compose --profile test-unit`, which pins `umap-learn==0.5.5`, `hdbscan==0.8.33`, `scikit-learn==1.4.0` |

Reproduce: `scripts/eval/run.sh [work_dir]` (default `/tmp/g4-clustering`).
Stage 2 runs on the host because ollama is published on `127.0.0.1` only.

### Corpus recipe (tab-shaped — the verdict corpus)

20 Newsgroups, `subset='all'`, `remove=('headers','footers','quotes')`, documents
under 300 chars dropped. `seed = 20260807 + n`.

- **7 groups**, power-law sizes (`rank^-1.1`), ordered by usable pool size so the
  heaviest rank never exhausts the source: `sci.med`, `talk.politics.mideast`,
  `sci.space`, `rec.sport.hockey`, `rec.autos`, `comp.sys.mac.hardware`,
  `comp.graphics`. At n=2000 that is `[825, 385, 247, 180, 141, 115, 97]`.
- **Near-duplicates**: 25% of the two largest groups are boilerplate variants
  (nav header + footer + a rotated body window) of 8 seed pages from that group —
  "many tabs open on one docs site". 302 of the 2000 base documents at n=2000.
- **Navigational stubs**: 8% of documents replaced by their title only. 159 of 2000.
  Base kind counts at n=2000: 1529 base, 302 near-duplicate, 159 stub, 10 outlier.
- **One-off outliers**: exactly 10, one document each from three held-out groups
  (`soc.religion.christian`, `misc.forsale`, `talk.religion.misc`), each with a
  unique ground-truth label so each is a true class of size 1.
- **Increments**: +10 documents, five times, sampled from the same power law.
- No arm ever sees a URL, hostname, or title separate from the text, so purity
  cannot be scored on hostname matching.

`vanilla_498` — 6 balanced groups of 83, no injections — is a **calibration point
only**. It is never the verdict corpus, for the reason the addendum gives.

### Arms

| arm | what it is |
|---|---|
| **A** production | `TabClusterer()` at the defaults `services/ai-engine/app/main.py:43` constructs it with: `min_cluster_size=3, min_samples=2, cluster_selection_epsilon=0.5, umap_n_neighbors=15, umap_n_components=5, umap_min_dist=0.1`, UMAP metric cosine, HDBSCAN metric euclidean on the reduced space, `random_state=42`. **Not** the loosened parameters `tests/unit/test_clustering.py` uses. Flat partition only — see limitations. |
| **B** agglomerative | average linkage, cosine distance, `distance_threshold` chosen by max silhouette over `0.05…0.98` step `0.01`, no ground truth consulted. |
| **B2** agglomerative, k≥5 | same, restricted to cuts producing at least 5 groups. Added after B degenerated to k=2; see below. |
| **C** k-means | L2-normalised rows (spherical k-means ≡ cosine), `k` by max silhouette over `2…min(60, n/4)`. **Not** capped at 10 — `_kmeans_cluster`'s `max_clusters=10` belongs to an `ImportError` fallback that never runs in production and would have been a strawman. |
| A2 *(supplementary, excluded from the verdict)* | arm A with `cluster_selection_epsilon=0.0`, to separate "the algorithm" from "the shipped parameters". |

Selection wall-time is reported separately from fit wall-time, because arm A does
no parameter search at all and that is a real advantage worth seeing.

---

## 2. The decision rule, applied

Best TypeScript candidate is taken per-metric across {B, B2, C}.

| | n=500 | n=2000 |
|---|---|---|
| A ARI | 0.274 | 0.179 |
| best TS ARI | 0.144 (C) | 0.112 (C) |
| **ARI margin** | **+0.130** ✓ (≥ 0.10) | **+0.067** ✗ |
| A purity | 0.698 | 0.667 |
| best TS purity | 0.958 (B2) | 0.977 (B2) |
| **purity margin** | **−0.260** ✗ | **−0.310** ✗ |
| A survival (groups ≥ 3) | 0.782 | 0.916 |
| best TS survival | 1.000 (B) / 0.969 (B2) | 1.000 (B) / 0.992 (B2) |
| **no worse on survival?** | **no** ✗ | **no** ✗ |

The rule requires the quality leg at **both** sizes. It holds at 500 and fails at
2000. The survival condition fails at both. **DROP.**

### Did the added arm change anything?

No. B2 was added after seeing that B's silhouette-selected threshold collapsed to
k=2 (ARI 0.002 at n=500), which would have made A look good against a comparator
that had stopped working. Adding it pushes toward the null hypothesis, which is
the direction that costs the sidecar rather than helps it — but the verdict does
not depend on it. Deleting B and B2 entirely and comparing only against k-means:

| | n=500 | n=2000 |
|---|---|---|
| A ARI − C ARI | +0.130 ✓ | **+0.067 ✗** |
| A purity − C purity | −0.142 ✗ | −0.196 ✗ |

Still fails at 2000, still DROP.

---

## 3. All metrics, all sizes, all arms

### Quality

| corpus | n | arm | groups shown | non-noise clusters | noise | ARI | purity | intra cos | inter cos | separation |
|---|---|---|---|---|---|---|---|---|---|---|
| tab_100 | 100 | A production (UMAP+HDBSCAN) | 2 | 2 | 0 | 0.056 | 0.450 | 0.515 | 0.507 | 0.008 |
| tab_100 | 100 | B agglomerative | 50 | 50 | 0 | 0.117 | 0.890 | 0.712 | 0.509 | 0.203 |
| tab_100 | 100 | B2 agglomerative (k≥5) | 50 | 50 | 0 | 0.117 | 0.890 | 0.712 | 0.509 | 0.203 |
| tab_100 | 100 | C k-means | 23 | 23 | 0 | 0.123 | 0.740 | 0.596 | 0.507 | 0.089 |
| tab_100 | 100 | A2 *(supp.)* | 2 | 2 | 0 | 0.056 | 0.450 | 0.515 | 0.507 | 0.008 |
| **tab_500** | 500 | **A production** | 22 | 12 | 10 | **0.274** | 0.698 | 0.551 | 0.518 | 0.033 |
| tab_500 | 500 | B agglomerative | 2 | 2 | 0 | 0.002 | 0.408 | 0.527 | 0.389 | 0.139 |
| tab_500 | 500 | B2 agglomerative (k≥5) | 235 | 235 | 0 | 0.064 | 0.958 | 0.758 | 0.525 | 0.234 |
| tab_500 | 500 | C k-means | 52 | 52 | 0 | 0.144 | 0.840 | 0.637 | 0.523 | 0.114 |
| tab_500 | 500 | A2 *(supp.)* | 156 | 56 | 100 | 0.089 | 0.922 | 0.670 | 0.524 | 0.146 |
| **tab_2000** | 2000 | **A production** | 26 | 21 | 5 | **0.179** | 0.667 | 0.529 | 0.516 | 0.013 |
| tab_2000 | 2000 | B agglomerative | 2 | 2 | 0 | −0.000 | 0.412 | 0.520 | 0.366 | 0.154 |
| tab_2000 | 2000 | B2 agglomerative (k≥5) | 934 | 934 | 0 | 0.026 | 0.977 | 0.839 | 0.519 | 0.321 |
| tab_2000 | 2000 | C k-means | 60 | 60 | 0 | 0.112 | 0.863 | 0.643 | 0.517 | 0.126 |
| tab_2000 | 2000 | A2 *(supp.)* | 602 | 174 | 428 | 0.081 | 0.935 | 0.640 | 0.518 | 0.122 |
| *vanilla_498* | 498 | A production | 2 | 2 | 0 | 0.127 | 0.325 | 0.514 | 0.495 | 0.019 |
| *vanilla_498* | 498 | B agglomerative | 2 | 2 | 0 | 0.000 | 0.173 | 0.510 | 0.425 | 0.085 |
| *vanilla_498* | 498 | B2 agglomerative (k≥5) | 159 | 159 | 0 | 0.192 | 0.928 | 0.685 | 0.505 | 0.180 |
| *vanilla_498* | 498 | **C k-means** | 6 | 6 | 0 | **0.801** | 0.910 | 0.582 | 0.494 | 0.088 |
| *vanilla_498* | 498 | A2 *(supp.)* | 2 | 2 | 0 | 0.127 | 0.325 | 0.514 | 0.495 | 0.019 |

"groups shown" is what the product renders: `pipeline.py::_group_by_labels` emits
one "Uncategorized" row per noise point, so ARI and purity are scored on that
expanded partition. For arm A at n=500 the two ARIs are identical to 4 decimals
(0.2744 expanded vs 0.2743 with noise as one cluster), so the choice does not
move the verdict.

Coherence is mean pairwise cosine on the **raw 768-d vectors** for every arm, so
the UMAP arm gets no credit for being tight in its own reduced space.

**n=100 is reported but is not part of the verdict, and it is under-powered.**
The pre-registered rule only reads 500 and 2000, which is correct here. All seven
real groups clear `min_cluster_size=3` at n=100 (sizes 39/17/11/8/6/5/4), so the
size is not degenerate in the strict sense — but the ten injected one-offs are
true classes of size 1 and can never be recovered as clusters by any arm, arm A
collapses to two clusters, and every arm's ARI sits between 0.056 and 0.123. At
this size the corpus does not separate the arms, it only shows that none of them
works on a hundred tabs.

### Singletons and orphans

Singletons are documents the product shows alone: HDBSCAN's `-1` points for arm
A (read from the labels, not the API payload), and size-1 clusters for the others.
Orphan rate = documents whose true group had ≥ `min_cluster_size` members present
but which were shown alone anyway. `min_cluster_size` was 3 at every size.

| corpus | arm | singletons shown | precision | recall | orphan rate | orphans / eligible |
|---|---|---|---|---|---|---|
| tab_100 | A production | 0 | n/a | 0.000 | 0.000 | 0/90 |
| tab_100 | B agglomerative | 28 | 0.107 | 0.300 | 0.278 | 25/90 |
| tab_100 | B2 agglomerative (k≥5) | 28 | 0.107 | 0.300 | 0.278 | 25/90 |
| tab_100 | C k-means | 0 | n/a | 0.000 | 0.000 | 0/90 |
| tab_100 | A2 *(supp.)* | 0 | n/a | 0.000 | 0.000 | 0/90 |
| tab_500 | A production | 10 | 0.100 | 0.100 | 0.018 | 9/490 |
| tab_500 | B agglomerative | 1 | 0.000 | 0.000 | 0.002 | 1/490 |
| tab_500 | B2 agglomerative (k≥5) | 148 | 0.034 | 0.500 | 0.292 | 143/490 |
| tab_500 | C k-means | 0 | n/a | 0.000 | 0.000 | 0/490 |
| tab_500 | A2 *(supp.)* | 100 | 0.040 | 0.400 | 0.196 | 96/490 |
| tab_2000 | A production | 5 | 0.000 | 0.000 | 0.003 | 5/1990 |
| tab_2000 | B agglomerative | 1 | 0.000 | 0.000 | 0.001 | 1/1990 |
| tab_2000 | B2 agglomerative (k≥5) | 604 | 0.013 | 0.800 | 0.299 | 596/1990 |
| tab_2000 | C k-means | 0 | n/a | 0.000 | 0.000 | 0/1990 |
| tab_2000 | A2 *(supp.)* | 428 | 0.009 | 0.400 | 0.213 | 424/1990 |
| *vanilla_498* | A production | 0 | n/a | n/a | 0.000 | 0/498 |
| *vanilla_498* | B agglomerative | 0 | n/a | n/a | 0.000 | 0/498 |
| *vanilla_498* | B2 agglomerative (k≥5) | 64 | 0.000 | n/a | 0.129 | 64/498 |
| *vanilla_498* | C k-means | 0 | n/a | n/a | 0.000 | 0/498 |
| *vanilla_498* | A2 *(supp.)* | 0 | n/a | n/a | 0.000 | 0/498 |

### Decision-39 group survival (+10 docs × 5 increments, Jaccard ≥ 0.5, greedy one-to-one)

"groups ≥ min_cluster_size" is the primary number: a one-document "Uncategorized"
row matches itself trivially, and decision 39 exists to protect groups that can
carry a relabel or a pin. Both columns are shown so the choice is visible.

| corpus | arm | survival (groups ≥ 3) | survival (all rows) | membership churn |
|---|---|---|---|---|
| tab_100 | A production | 0.467 | 0.442 | 0.725 |
| tab_100 | B agglomerative | 0.641 | 0.716 | 0.343 |
| tab_100 | B2 agglomerative (k≥5) | 0.829 | 0.909 | 0.169 |
| tab_100 | C k-means | 0.576 | 0.579 | 0.543 |
| tab_100 | A2 *(supp.)* | 0.508 | 0.463 | 0.700 |
| tab_500 | A production | 0.782 | 0.683 | 0.484 |
| tab_500 | B agglomerative | 1.000 | 1.000 | 0.000 |
| tab_500 | B2 agglomerative (k≥5) | 0.969 | 0.988 | 0.041 |
| tab_500 | C k-means | 0.363 | 0.348 | 0.710 |
| tab_500 | A2 *(supp.)* | 0.715 | 0.645 | 0.293 |
| tab_2000 | A production | 0.916 | 0.761 | 0.074 |
| tab_2000 | B agglomerative | 1.000 | 1.000 | 0.000 |
| tab_2000 | B2 agglomerative (k≥5) | 0.992 | 0.997 | 0.005 |
| tab_2000 | C k-means | 0.416 | 0.414 | 0.657 |
| tab_2000 | A2 *(supp.)* | 0.574 | 0.580 | 0.435 |
| *vanilla_498* | A production | 0.586 | 0.540 | 0.493 |
| *vanilla_498* | B agglomerative | 1.000 | 1.000 | 0.000 |
| *vanilla_498* | B2 agglomerative (k≥5) | 0.932 | 0.951 | 0.125 |
| *vanilla_498* | C k-means | 1.000 | 1.000 | 0.022 |
| *vanilla_498* | A2 *(supp.)* | 0.410 | 0.403 | 0.674 |

### Perturbation stability (ARI vs the base run, 5 repeats each)

Not plain reruns: `pipeline.py` pins `random_state=42`, so an identical-input
rerun is bit-identical and would score a vacuous 1.0. These perturb what a real
reindex perturbs — the order documents arrive in, and which documents are present.

| corpus | arm | order-shuffle mean | order-shuffle min | 90% resample mean | 90% resample min |
|---|---|---|---|---|---|
| tab_100 | A production | **0.022** | 0.017 | 0.025 | 0.012 |
| tab_100 | B agglomerative | 1.000 | 1.000 | 0.760 | 0.507 |
| tab_100 | B2 agglomerative (k≥5) | 1.000 | 1.000 | 0.760 | 0.507 |
| tab_100 | C k-means | 0.286 | 0.235 | 0.315 | 0.244 |
| tab_100 | A2 *(supp.)* | 0.021 | 0.015 | 0.025 | 0.012 |
| tab_500 | A production | **0.462** | 0.269 | 0.386 | 0.236 |
| tab_500 | B agglomerative | 1.000 | 1.000 | 0.800 | −0.002 |
| tab_500 | B2 agglomerative (k≥5) | 1.000 | 1.000 | 0.797 | 0.752 |
| tab_500 | C k-means | 0.394 | 0.324 | 0.385 | 0.324 |
| tab_500 | A2 *(supp.)* | 0.691 | 0.673 | 0.630 | 0.279 |
| tab_2000 | A production | **0.746** | 0.538 | 0.627 | 0.403 |
| tab_2000 | B agglomerative | 1.000 | 1.000 | 0.600 | 0.000 |
| tab_2000 | B2 agglomerative (k≥5) | 1.000 | 1.000 | 0.945 | 0.896 |
| tab_2000 | C k-means | 0.450 | 0.432 | 0.437 | 0.408 |
| tab_2000 | A2 *(supp.)* | 0.752 | 0.692 | 0.573 | 0.248 |
| *vanilla_498* | A production | 0.575 | 0.266 | 0.364 | 0.165 |
| *vanilla_498* | B agglomerative | 1.000 | 1.000 | 0.866 | 0.664 |
| *vanilla_498* | B2 agglomerative (k≥5) | 1.000 | 1.000 | 0.759 | 0.608 |
| *vanilla_498* | C k-means | 0.931 | 0.917 | 0.902 | 0.876 |
| *vanilla_498* | A2 *(supp.)* | 0.480 | 0.074 | 0.253 | 0.037 |

### Wall time

| corpus | arm | parameter search (s) | final fit (s) | chosen |
|---|---|---|---|---|
| tab_100 | A production | 0.0 | 29.1 | fixed defaults |
| tab_100 | B agglomerative | 0.0 | 0.0 | threshold 0.37 |
| tab_100 | B2 agglomerative (k≥5) | 0.0 | 0.0 | threshold 0.37 |
| tab_100 | C k-means | 13.2 | 0.5 | k = 23 |
| tab_500 | A production | 0.0 | 22.2 | fixed defaults |
| tab_500 | B agglomerative | 0.1 | 0.0 | threshold 0.59 |
| tab_500 | B2 agglomerative (k≥5) | 0.1 | 0.0 | threshold 0.32 |
| tab_500 | C k-means | 23.7 | 1.2 | k = 52 |
| tab_2000 | A production | 0.0 | 27.0 | fixed defaults |
| tab_2000 | B agglomerative | 1.0 | 0.0 | threshold 0.61 |
| tab_2000 | B2 agglomerative (k≥5) | 0.6 | 0.0 | threshold 0.29 |
| tab_2000 | C k-means | 181.3 | 7.7 | k = 60 |
| *vanilla_498* | A production | 0.0 | 22.0 | fixed defaults |
| *vanilla_498* | C k-means | 30.6 | 0.8 | k = 6 |

Single-threaded inside the test image (UMAP forces `n_jobs=1` when
`random_state` is set — which is exactly the trade the pinned seed buys).
Agglomerative including its full threshold sweep is the fastest arm at every size
by two to three orders of magnitude.

---

## 4. What the numbers say beyond the verdict

**Nobody organizes this corpus well.** The best ARI anywhere on the tab-shaped
corpus is arm A's 0.274 at n=500. This is a choice among mediocre options, and
the sidecar is the most expensive mediocre option. Do not read "DROP" as "the
TypeScript arms are good"; read it as "the sidecar is not buying anything worth a
second runtime". The freeform-geometry path is weak for every implementation,
which is an argument for the plan's own facet path (`GROUP BY facet_value`,
deterministic and explainable) carrying more of the load than geometry does.

**HDBSCAN's headline advantage did not appear.** The stated reason to keep a
Python sidecar is that HDBSCAN handles unknown k, variable density, skewed sizes,
and genuine noise. On a corpus built specifically to exercise all four, arm A
isolated **1 of 10** injected one-offs at n=500 and **0 of 10** at n=2000
(precision 0.100 and 0.000). Its 5 noise points at n=2000 were all documents whose
true group had plenty of members present. Noise detection — the capability
agglomerative and k-means genuinely lack — did not materialise at the shipped
parameters.

**Arm A's clusters are not coherent in the space search uses.** Its intra-minus-
inter cosine separation is 0.008 / 0.033 / 0.013 at 100 / 500 / 2000, against
0.089–0.321 for every non-degenerate alternative. Its groups are shaped by UMAP's
5-d neighbourhood geometry, not by similarity in the 768-d space that hybrid
search ranks in. A group produced this way cannot be explained to a user as
"these pages are alike", and it will not agree with what search returns.

**Arm A is order-dependent, and that is the finding most relevant to decision 39.**
`random_state=42` makes it deterministic only for a fixed row order. Shuffle the
input and the partition changes: ARI 0.022 (n=100), 0.462 (n=500), 0.746 (n=2000)
against the unshuffled run on the same documents. Agglomerative is exactly
order-invariant (1.000 at every size, by construction). Tab ingest order is
whatever order the browser hands over, so this is not a hypothetical.

**The one thing arm A does well is not needing a parameter search.** HDBSCAN
finds its own cluster count; k-means needs 181s of silhouette search at n=2000 to
find k=60, and then finds a bad one. That is a real property, and it is available
in TypeScript from the agglomerative side too — a dendrogram is built once and cut
anywhere, which is why B/B2's "search" costs 0.6–1.0s at n=2000.

**Degenerate wins are visible in this table and should not be mistaken for
quality.** Arm B's perfect 1.000 survival at n=500 and n=2000 is the survival of a
two-cluster partition with ARI ≈ 0. B2's 0.992 survival at n=2000 is the survival
of a 934-cluster partition that shows 604 documents alone, 596 of them from groups
that had members to spare (orphan rate 0.299). Both are stable because they are
not doing anything. Arm A's 0.916 at
n=2000 with 26 groups is a more meaningful number than either — it just is not a
*winning* number under the rule, and the rule is what decides.

**The calibration corpus behaved exactly as the addendum predicted.** On balanced
vanilla 20NG, k-means recovered the ground truth almost perfectly (ARI 0.801, k=6
= true k) while arm A collapsed to two clusters (ARI 0.127, purity 0.325). A
verdict from that corpus would have been a pre-baked landslide. The tab-shaped
corpus is where arm A does *best* relative to the field — it wins ARI at 500 there
and loses it outright on vanilla — and it still fails. Corpus shape did not
manufacture this result; it was chosen to give the sidecar its best shot.

---

## 5. What would have changed the verdict

- **Arm A needed ARI ≥ 0.212 at n=2000** (C's 0.112 + 0.10). It got 0.179. The gap
  is 0.033 ARI — close enough that a different embedding model, a different
  `cluster_selection_epsilon`, or a different corpus draw could plausibly cross it.
  It is *not* close on the other two legs.
- **Or purity ≥ 0.963 at n=2000 and ≥ 0.940 at n=500** (C + 10 points). It got
  0.667 and 0.698. Not close, and unreachable without shredding into hundreds of
  clusters — which is what B2 does to reach 0.977.
- **And, either way, survival ≥ 0.992 at n=2000 and ≥ 0.969 at n=500.** It got
  0.916 and 0.782.
- Tuning arm A does not rescue it either. `cluster_selection_epsilon=0.0` (arm A2)
  raises purity to 0.935 at n=2000 by producing 602 rows of which 428 are single
  documents, and *drops* ARI to 0.081 and survival to 0.574. The shipped epsilon
  is the better of the two settings tested, so "the defaults are just wrong" is
  not an available explanation for this result.
- The near-miss at 2000 is the one place a re-run could matter. If someone wants
  to reopen this, the cheapest decisive experiment is arm A at n=2000 under a
  different embedding model and 3–5 corpus seeds, with confidence intervals. This
  evaluation has neither.

---

## 6. Where I think the pre-registered rule is wrong

Recorded separately, as required, and **not applied**. The verdict above uses the
rule exactly as written.

1. **The purity leg is close to unusable as specified.** Purity rises
   monotonically with cluster count; any arm can win it by shredding. B2 scores
   0.977 at n=2000 while showing 596 of 1990 documents alone. Because the rule
   reads "≥ 0.10 ARI **or** ≥ 10 points purity" against "the best TypeScript
   candidate", a shredding comparator can block the purity leg for free, and
   symmetrically a shredding *production* arm could have won it for free. It did
   not bite here — arm A lost both legs at 2000 — but it would have under
   different numbers. Purity needs a companion constraint (cluster count within
   some factor of the true count, or an orphan-rate ceiling) to mean anything.
2. **"No worse on group survival" does not name a comparator or a validity
   floor.** Arm B has perfect survival at every size by emitting two clusters
   forever. A rule that a do-nothing partition passes is not testing what it
   means to test. Survival should be conditioned on the partition being non-
   degenerate — e.g. ARI above some floor, or cluster count in a sane band.
3. **"Best TypeScript-candidate arm" is ambiguous per-metric.** Here the best-by-
   ARI arm (C) and the best-by-purity arm (B2) are different arms, so "the best
   arm" is not a single object. I resolved it per-metric, which is the reading
   least favourable to keeping and therefore the right one under a `drop` null —
   but the rule should have said so.

None of these change DROP. Under every resolution of every ambiguity above —
per-metric best, single-best-by-ARI, C-only, B-excluded — the quality leg fails at
n=2000.

---

## 7. Limitations

- **The chunk-mean regime is barely exercised by this corpus.** 20 Newsgroups
  posts are short: at the repo's 4000-char chunker only 7/150, 45/550, 143/2050
  documents produce more than one chunk, and `nomic-embed-text`'s 8192-token
  window truncates none of them. So mean-of-chunks and whole-page embedding are
  numerically near-identical here. Using the chunk-mean regime avoids evaluating
  an input that is scheduled for deletion; it does **not** demonstrate anything
  about whether chunk-mean vectors fix the bad labels the plan blames on
  truncation. Real pages are 5–50 KB and the two regimes would diverge there.
  This evaluation says nothing about that.
- **20 Newsgroups topics are more separable than real tabs.** Real tab corpora mix
  docs pages, issues, and blog posts about the *same* library — a much harder and
  less topically-separated problem. Absolute numbers here are optimistic. Only the
  between-arm comparison should be read as transferable, and even that assumes the
  ranking is corpus-invariant, which is not established.
- **The flat partition only.** `cluster()` recursively sub-clusters any cluster
  over 10 tabs to depth 3; `cluster_sync` and therefore this evaluation score the
  top-level partition. Sub-clustering could make arm A's 825-document `sci.med`
  blob more usable in the UI without moving top-level ARI at all. Not measured.
- **No labels, so no label quality.** Labels are what users see; this measures only
  which documents land together. No LLM was called, deliberately.
- **One embedding model, one corpus seed, one run per arm per corpus.** The
  perturbation repeats are 5, but the corpus draw is a single sample and there are
  no confidence intervals on any number in this document. The n=2000 ARI margin
  (0.067 vs the 0.10 bar) is inside the range a different seed could plausibly
  move.
- **`nomic-embed-text` is called without nomic's `search_document:` /
  `search_query:` task prefixes** — whatever ollama's template applies is what was
  used, which is also what production does. Consistent across arms; possibly
  suboptimal for all of them.
- **Silhouette is one "sensible" selection rule and it visibly misbehaves** (k=2
  for arm B at n=500 and n=2000, k=60 — the grid ceiling — for arm C at n=2000).
  A better unsupervised selection rule would raise the TypeScript arms, which only
  strengthens DROP.
- **Arm C's k grid tops out at 60.** It selected the ceiling at n=2000, so the true
  silhouette optimum may be higher. Again: relaxing this can only help the
  TypeScript side.
- **Not measured at all:** memory footprint, cold-start cost of the sidecar
  process, behaviour on a corpus with no clusterable structure, and the
  interaction with decision 26 (Python and TypeScript never writing the same
  SQLite), which is a cost the sidecar carries regardless of these numbers.

---

## 8. Files

| | |
|---|---|
| `scripts/eval/run.sh` | runs all four stages |
| `scripts/eval/g4_corpus.py` | stage 1 — builds the tab-shaped and vanilla corpora |
| `scripts/eval/g4_embed.py` | stage 2 — chunks, embeds via local ollama, caches vectors |
| `scripts/eval/g4_arms.py` | stage 3 — the four arms, the backend assertion, the scoring loop |
| `scripts/eval/g4_metrics.py` | metric definitions (ARI, purity, coherence, singletons, survival) |
| `scripts/eval/g4_report.py` | stage 4 — renders the tables in section 3 |
| `scripts/eval/g4_pool_sizes.py` | one-off diagnostic used to pick the seven groups and their ranks |

Work directory (corpora, 20NG download, vector cache, raw results) defaults to
`/tmp/g4-clustering` and is deliberately outside the repository. No vectors are
committed.

---

*Run 2026-08-07 against branch `plan-completion`. Arm-A backend assertion passed
on every run reported here: `reduce_backend="umap"`, `cluster_backend="hdbscan"`.*
