
"""Unit tests for PostgreSQL keyword retrieval."""

import pytest
from sqlalchemy.dialects import postgresql

from retrieval.keyword_search import (
    MAX_TOP_K,
    _build_keyword_statement,
    _validate_query,
)


def test_validate_query_trims_whitespace():
    assert _validate_query("  foreign policy  ", 5) == (
        "foreign policy",
        5,
    )


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_validate_query_rejects_empty_query(query):
    with pytest.raises(ValueError, match="non-empty"):
        _validate_query(query, 5)


@pytest.mark.parametrize("top_k", [0, -1, MAX_TOP_K + 1, True, 1.5])
def test_validate_query_rejects_invalid_top_k(top_k):
    with pytest.raises(ValueError):
        _validate_query("MEA report", top_k)


def test_build_statement_uses_postgres_full_text_search():
    statement = _build_keyword_statement("foreign policy", 10)
    compiled = str(statement.compile(dialect=postgresql.dialect()))

    assert "to_tsvector" in compiled
    assert "plainto_tsquery" in compiled
    assert "ts_rank_cd" in compiled
    assert "@@" in compiled
    assert "document_chunks" in compiled
    assert "source_documents" in compiled


def test_build_statement_limits_results():
    statement = _build_keyword_statement("MEA report", 7)
    compiled = statement.compile(dialect=postgresql.dialect())

    assert compiled.params["param_1"] == 7
