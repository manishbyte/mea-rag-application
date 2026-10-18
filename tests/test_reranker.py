
"""Unit tests for cross-encoder reranking."""

import anyio
import pytest

from retrieval import reranker
from retrieval.reranker import RerankerError, rerank, rerank_async


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.received_pairs = None

    def predict(self, pairs, batch_size, show_progress_bar):
        self.received_pairs = pairs
        return self.scores


def candidate(chunk_id, content, method="hybrid"):
    return {
        "chunk_id": chunk_id,
        "content": content,
        "retrieval_method": method,
        "page_number": 5,
    }


def test_rerank_orders_candidates_by_score(monkeypatch):
    model = FakeCrossEncoder([0.2, 2.1, 1.0])
    monkeypatch.setattr(reranker, "_get_cross_encoder", lambda: model)

    results = rerank(
        "What does the report say about exports?",
        [
            candidate("a", "Unrelated content"),
            candidate("b", "Export performance details"),
            candidate("c", "Trade statistics"),
        ],
    )

    assert [item["chunk_id"] for item in results] == ["b", "c", "a"]
    assert [item["rerank_rank"] for item in results] == [1, 2, 3]
    assert results[0]["rerank_score"] == pytest.approx(2.1)


def test_rerank_preserves_original_metadata(monkeypatch):
    monkeypatch.setattr(
        reranker,
        "_get_cross_encoder",
        lambda: FakeCrossEncoder([1.5]),
    )

    results = rerank("question", [candidate("a", "Relevant content")])

    assert results[0]["page_number"] == 5
    assert results[0]["retrieval_method"] == "hybrid"
    assert results[0]["content"] == "Relevant content"


def test_rerank_respects_top_k(monkeypatch):
    monkeypatch.setattr(
        reranker,
        "_get_cross_encoder",
        lambda: FakeCrossEncoder([0.1, 0.9, 0.5]),
    )

    results = rerank(
        "question",
        [
            candidate("a", "First"),
            candidate("b", "Second"),
            candidate("c", "Third"),
        ],
        top_k=2,
    )

    assert len(results) == 2
    assert [item["chunk_id"] for item in results] == ["b", "c"]


def test_rerank_returns_empty_for_no_candidates(monkeypatch):
    def unexpected_model_load():
        pytest.fail("Model should not load for empty candidates.")

    monkeypatch.setattr(reranker, "_get_cross_encoder", unexpected_model_load)

    assert rerank("question", []) == []


@pytest.mark.parametrize(
    ("query", "candidates", "top_k"),
    [
        ("", [], 5),
        ("question", [{"chunk_id": "a"}], 5),
        ("question", [candidate("a", "content")], 0),
        ("question", [candidate("a", "content")], True),
    ],
)
def test_rerank_rejects_invalid_inputs(query, candidates, top_k):
    with pytest.raises(ValueError):
        rerank(query, candidates, top_k)


def test_rerank_rejects_non_finite_scores(monkeypatch):
    monkeypatch.setattr(
        reranker,
        "_get_cross_encoder",
        lambda: FakeCrossEncoder([float("nan")]),
    )

    with pytest.raises(RerankerError, match="non-finite"):
        rerank("question", [candidate("a", "Some content")])


def test_rerank_preserves_original_order_for_tied_scores(monkeypatch):
    monkeypatch.setattr(
        reranker,
        "_get_cross_encoder",
        lambda: FakeCrossEncoder([1.0, 1.0]),
    )

    results = rerank(
        "question",
        [
            candidate("first", "First candidate"),
            candidate("second", "Second candidate"),
        ],
    )

    assert [item["chunk_id"] for item in results] == [
        "first",
        "second",
    ]


def test_rerank_async(monkeypatch):
    monkeypatch.setattr(
        reranker,
        "_get_cross_encoder",
        lambda: FakeCrossEncoder([0.1, 1.2]),
    )

    results = anyio.run(
        rerank_async,
        "question",
        [
            candidate("a", "First"),
            candidate("b", "Second"),
        ],
    )

    assert results[0]["chunk_id"] == "b"
