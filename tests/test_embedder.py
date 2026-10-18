
"""Unit tests for embedding validation and chunk embedding."""

from unittest.mock import patch

import pytest

from ingestion.chunker import TextChunk
from ingestion.embedder import (
    EMBEDDING_DIMENSIONS,
    EmbeddingError,
    embed_chunks,
    embed_texts,
    validate_embedding,
)


def test_validate_embedding_accepts_correct_dimensions():
    vector = [0.1] * EMBEDDING_DIMENSIONS

    result = validate_embedding(vector)

    assert len(result) == EMBEDDING_DIMENSIONS
    assert all(isinstance(value, float) for value in result)


def test_validate_embedding_rejects_wrong_dimensions():
    with pytest.raises(EmbeddingError, match="Expected"):
        validate_embedding([0.1, 0.2])


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_validate_embedding_rejects_non_finite_values(invalid_value):
    vector = [0.1] * EMBEDDING_DIMENSIONS
    vector[0] = invalid_value

    with pytest.raises(EmbeddingError, match="NaN or infinite"):
        validate_embedding(vector)


def test_validate_embedding_rejects_non_numeric_values():
    vector = [0.1] * EMBEDDING_DIMENSIONS
    vector[0] = "invalid"

    with pytest.raises(EmbeddingError, match="non-numeric"):
        validate_embedding(vector)


def test_embed_texts_empty_input_returns_empty_list():
    with patch("ingestion.embedder.get_embedding_model") as get_model:
        assert embed_texts([]) == []
        get_model.assert_not_called()


def test_embed_texts_rejects_empty_passages():
    with pytest.raises(ValueError, match="empty passages"):
        embed_texts(["valid text", "   "])


def test_embed_texts_rejects_invalid_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        embed_texts(["valid text"], batch_size=0)


def test_embed_chunks_preserves_chunk_metadata():
    chunk = TextChunk(
        chunk_index=0,
        page_number=2,
        content="MEA annual report content",
        start_char=10,
        end_char=36,
    )
    fake_embedding = [0.1] * EMBEDDING_DIMENSIONS

    with patch(
        "ingestion.embedder.embed_texts",
        return_value=[fake_embedding],
    ):
        result = embed_chunks([chunk])

    assert len(result) == 1
    assert result[0]["chunk_index"] == 0
    assert result[0]["page_number"] == 2
    assert result[0]["content"] == "MEA annual report content"
    assert result[0]["start_char"] == 10
    assert result[0]["end_char"] == 36
    assert len(result[0]["embedding"]) == EMBEDDING_DIMENSIONS


def test_embed_chunks_empty_input_returns_empty_list():
    with patch("ingestion.embedder.embed_texts") as embed:
        assert embed_chunks([]) == []
        embed.assert_not_called()
