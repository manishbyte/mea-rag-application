
"""Tests for retrieval query embeddings."""

import math

import pytest

from retrieval.embeddings import (
    EMBEDDING_DIMENSIONS,
    QueryEmbeddingError,
    embed_query,
    embed_query_async,
)


def test_embed_query_rejects_empty_input():
    with pytest.raises(ValueError):
        embed_query("   ")


def test_embed_query_returns_valid_vector(monkeypatch):
    import retrieval.embeddings as embeddings

    class FakeModel:
        def query_embed(self, texts):
            assert texts == ["MEA annual report"]
            return iter([[0.1] * EMBEDDING_DIMENSIONS])

    monkeypatch.setattr(
        embeddings,
        "_get_embedding_model",
        lambda: FakeModel(),
    )

    vector = embed_query("MEA annual report")

    assert len(vector) == EMBEDDING_DIMENSIONS
    assert all(math.isfinite(value) for value in vector)


def test_embed_query_rejects_wrong_dimensions(monkeypatch):
    import retrieval.embeddings as embeddings

    class FakeModel:
        def query_embed(self, texts):
            return iter([[0.1, 0.2]])

    monkeypatch.setattr(
        embeddings,
        "_get_embedding_model",
        lambda: FakeModel(),
    )

    with pytest.raises(QueryEmbeddingError):
        embed_query("MEA annual report")


def test_embed_query_rejects_non_finite_values(monkeypatch):
    import retrieval.embeddings as embeddings

    class FakeModel:
        def query_embed(self, texts):
            return iter([[float("nan")] * EMBEDDING_DIMENSIONS])

    monkeypatch.setattr(
        embeddings,
        "_get_embedding_model",
        lambda: FakeModel(),
    )

    with pytest.raises(QueryEmbeddingError):
        embed_query("MEA annual report")


def test_embed_query_async_uses_embedding_function(monkeypatch):
    import anyio
    import retrieval.embeddings as embeddings

    expected = [0.25] * EMBEDDING_DIMENSIONS

    monkeypatch.setattr(
        embeddings,
        "embed_query",
        lambda query: expected,
    )

    result = anyio.run(embed_query_async, "MEA annual report")

    assert result == expected
