
"""Unit tests for pgvector semantic retrieval."""

import pytest
from sqlalchemy.dialects import postgresql

from retrieval.vector_search import (
    EMBEDDING_DIMENSIONS,
    MAX_TOP_K,
    _build_vector_statement,
    _validate_inputs,
)


def valid_vector():
    return [0.01] * EMBEDDING_DIMENSIONS


def test_validate_inputs_accepts_valid_vector():
    vector, top_k = _validate_inputs(valid_vector(), 10)

    assert len(vector) == EMBEDDING_DIMENSIONS
    assert top_k == 10


def test_validate_inputs_rejects_wrong_dimensions():
    with pytest.raises(ValueError, match="dimensions"):
        _validate_inputs([0.1, 0.2], 5)


def test_validate_inputs_rejects_non_finite_values():
    vector = valid_vector()
    vector[0] = float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        _validate_inputs(vector, 5)


@pytest.mark.parametrize(
    "top_k",
    [0, -1, MAX_TOP_K + 1, True, 1.5],
)
def test_validate_inputs_rejects_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="top_k"):
        _validate_inputs(valid_vector(), top_k)


def test_build_vector_statement_uses_cosine_distance():
    statement = _build_vector_statement(valid_vector(), 10)
    compiled = str(statement.compile(dialect=postgresql.dialect()))

    assert "document_chunks" in compiled
    assert "source_documents" in compiled
    assert "<=>" in compiled
    assert "ORDER BY" in compiled
    assert "LIMIT" in compiled


def test_build_vector_statement_excludes_missing_embeddings():
    statement = _build_vector_statement(valid_vector(), 5)
    compiled = str(statement.compile(dialect=postgresql.dialect()))

    assert "embedding IS NOT NULL" in compiled
