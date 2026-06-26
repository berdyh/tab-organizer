# AI Clustering Submodule Card

- purpose: cluster scraped tabs and generate labels/summaries.
- product/module functionality: embeddings grouping, dimensionality reduction, HDBSCAN clusters, LLM-generated cluster labels.
- scope boundaries: owns clustering pipeline only; URL/session state lives in Backend Core.
- connected modules/submodules: AI Core, Backend clustering proxy, Web UI clustering page.
- allowed change types: clustering algorithm fixes, label prompt safety, model-dimension handling, tests.
- special operating rules: tab content in prompts must be marked untrusted.
- current stubs/placeholders: small-sample behavior is covered and should remain explicit.
- irrelevant or incomplete code to remove/rework: none known.
- docs that must stay aligned: AI Engine card and README clustering feature text.
- local validation commands/checks: `make test-ai`; focused file `tests/unit/test_clustering.py`.
