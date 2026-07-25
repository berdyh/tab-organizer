"""Parity regression: in-memory keyword search must return exactly the same
result set as the SQLite FTS-backed search for the same corpus and query.

CLAUDE.md invariant: SessionManager's in-memory fallback and its SQLite path
must stay behaviorally equivalent. `_upsert_search_record` only writes an FTS
row when a url record has status == "scraped" AND a non-empty title or
content; `_search_indexed_tabs_in_memory` must mirror that condition exactly
(see services/backend-core/app/sessions/manager.py). This test builds a
corpus with every status/content combination that could diverge and asserts
identical hits across both backends.
"""

from services.backend_core.app.sessions.manager import SessionManager

TERM = "zzuniquesearchterm"


def _build_corpus(manager):
    """Create one session with records in every status/content combination
    that could plausibly diverge between the SQLite FTS write rule and an
    in-memory approximation of it. Returns (session, expected_matching_urls).
    """
    session = manager.create_session("parity corpus")

    url_scraped_content = "https://example.com/scraped-content"
    url_scraped_title_only = "https://example.com/scraped-title-only"
    url_scraped_neither = "https://example.com/scraped-neither"
    url_pending = "https://example.com/pending-doc"
    url_failed = "https://example.com/failed-doc"

    manager.add_urls_to_session(
        session.id,
        [
            url_scraped_content,
            url_scraped_title_only,
            url_scraped_neither,
            url_pending,
            url_failed,
        ],
    )

    # scraped, has content containing the term -> should match
    manager.update_url_status(
        session.id,
        url_scraped_content,
        "scraped",
        metadata={"title": "Ordinary title", "content": f"body text with {TERM} inside"},
    )

    # scraped, only a title containing the term (empty content) -> should match
    manager.update_url_status(
        session.id,
        url_scraped_title_only,
        "scraped",
        metadata={"title": f"{TERM} appears only here", "content": ""},
    )

    # scraped, but neither title nor content set -> must never match, even
    # though the term sits in the URL (URL/domain are UNINDEXED in FTS5, and
    # a title/content-less scraped record gets no FTS row at all).
    manager.update_url_status(
        session.id,
        url_scraped_neither,
        "scraped",
        metadata={},
    )

    # pending, with a title/content that DOES contain the term -> must not
    # match: only status == "scraped" records are searchable.
    manager.update_url_status(
        session.id,
        url_pending,
        "pending",
        metadata={"title": f"{TERM} pending title", "content": f"{TERM} pending body"},
    )

    # failed, with a title/content that DOES contain the term -> must not
    # match either, for the same reason.
    manager.update_url_status(
        session.id,
        url_failed,
        "failed",
        metadata={"title": f"{TERM} failed title", "content": f"{TERM} failed body"},
    )

    expected = {url_scraped_content, url_scraped_title_only}
    return session, expected


def test_keyword_search_parity_across_mixed_status_corpus(tmp_path):
    """Same corpus, same query, both backends: identical result URL sets."""
    mem_manager = SessionManager()
    sql_manager = SessionManager(db_path=str(tmp_path / "search_parity.db"))

    mem_session, expected = _build_corpus(mem_manager)
    sql_session, expected_sql = _build_corpus(sql_manager)
    assert expected == expected_sql  # sanity: corpora were built identically

    mem_hits = {
        hit["url"] for hit in mem_manager.search_indexed_tabs(mem_session.id, TERM, 10)
    }
    sql_hits = {
        hit["url"] for hit in sql_manager.search_indexed_tabs(sql_session.id, TERM, 10)
    }

    assert mem_hits == sql_hits == expected
