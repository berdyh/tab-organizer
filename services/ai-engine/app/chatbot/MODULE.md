# AI Chatbot/RAG Submodule Card

- purpose: retrieval-augmented chat and document search over indexed tab content.
- product/module functionality: LanceDB indexing/search, untrusted-content wrapping, chat prompt assembly, document deletion.
- scope boundaries: owns vector retrieval behavior; embeddings come from AI provider runtime and user-visible chat UI lives in Web UI.
- connected modules/submodules: AI Core, Backend Core, Browser Engine indexing, Web UI chatbot.
- allowed change types: RAG persistence fixes, prompt safety, search behavior, tests.
- special operating rules: scraped page content is untrusted data and must not become system instructions.
- current stubs/placeholders: legacy LanceDB list schema rebuild is compatibility behavior.
- irrelevant or incomplete code to remove/rework: none known.
- docs that must stay aligned: AI Engine card and `docs/AI_CONFIG.md`.
- local validation commands/checks: `make test-ai`; focused file `tests/unit/test_rag_lancedb_persistence.py`.
