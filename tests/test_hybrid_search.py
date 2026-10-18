
"""Unit tests for hybrid retrieval orchestration."""

import anyio
import pytest

from retrieval import hybrid_search as hybrid_module
from retrieval.hybrid_search import (
    HybridSearchError,
    hybrid_search,
)


def make_result(chunk_id, method):
    return {
        "chunk_id": chunk_id,
        "document_id": "document-1",
        "chunk_index": 0,
        "page_number": 1,
        "content": f"Content for {chunk_id}",
        "metadata": {},
        "document_title": "MEA Annual Report 2025",
        "filename": "mea_annual_report.pdf",
        "retrieval_method": method,
    }


async def fake_embed(query):
    return [0.1] * 384


async def fake_vector_search(
    embedding,
    top_k=10,
    *,
    session=None,
):
    return [
        make_result("shared", "vector"),
        make_result("vector-only", "vector"),
    ][:top_k]


async def fake_keyword_search(
    query,
    top_k=10,
    *,
    session=None,
):
    return [
        make_result("shared", "keyword"),
        make_result("keyword-only", "keyword"),
    ][:top_k]


async def fake_rerank(query, candidates, top_k=10):
    results = [dict(item) for item in candidates[:top_k]]
    for rank, result in enumerate(results, start=1):
        result["rerank_rank"] = rank
        result["rerank_score"] = 1.0 / rank
    return results


def install_fakes(monkeypatch):
    monkeypatch.setattr(
        hybrid_module,
        "embed_query_async",
        fake_embed,
    )
    monkeypatch.setattr(
        hybrid_module,
        "vector_search",
        fake_vector_search,
    )
    monkeypatch.setattr(
        hybrid_module,
        "keyword_search",
        fake_keyword_search,
    )
    monkeypatch.setattr(
        hybrid_module,
        "rerank_async",
        fake_rerank,
    )



def test_hybrid_search_combines_retrievers_without_reranking(monkeypatch):
    install_fakes(monkeypatch)

    results = anyio.run(
        lambda: hybrid_search(
            "What does the report say?",
            3,
            10,
            use_reranker=False,
        )
    )

    ids = [result["chunk_id"] for result in results]

    assert "shared" in ids
    assert len(results) == 3
    assert all(
        result["retrieval_method"] == "hybrid"
        for result in results
    )



def test_hybrid_search_uses_reranker(monkeypatch):
    install_fakes(monkeypatch)

    results = anyio.run(
        hybrid_search,
        "What does the report say?",
        2,
        10,
    )

    assert len(results) == 2
    assert all("rerank_score" in result for result in results)
    assert all("rerank_rank" in result for result in results)


def test_hybrid_search_returns_empty_when_no_results(monkeypatch):
    install_fakes(monkeypatch)

    async def no_vector_results(*args, **kwargs):
        return []

    async def no_keyword_results(*args, **kwargs):
        return []

    monkeypatch.setattr(
        hybrid_module,
        "vector_search",
        no_vector_results,
    )
    monkeypatch.setattr(
        hybrid_module,
        "keyword_search",
        no_keyword_results,
    )

    results = anyio.run(
        hybrid_search,
        "A question with no matches",
        5,
        10,
    )

    assert results == []


@pytest.mark.parametrize(
    ("query", "top_k", "candidate_k"),
    [
        ("", 5, 20),
        ("question", 0, 20),
        ("question", 5, 0),
        ("question", 10, 5),
        ("question", True, 20),
    ],
)
def test_hybrid_search_rejects_invalid_inputs(
    query,
    top_k,
    candidate_k,
):
    with pytest.raises(ValueError):
        anyio.run(
            hybrid_search,
            query,
            top_k,
            candidate_k,
        )



def test_hybrid_search_rejects_non_boolean_reranker_setting():
    with pytest.raises(ValueError, match="use_reranker"):
        anyio.run(
            lambda: hybrid_search(
                "question",
                5,
                10,
                use_reranker="yes",
            )
        )



def test_hybrid_search_wraps_unexpected_errors(monkeypatch):
    install_fakes(monkeypatch)

    async def broken_embedding(query):
        raise RuntimeError("simulated embedding failure")

    monkeypatch.setattr(
        hybrid_module,
        "embed_query_async",
        broken_embedding,
    )

    with pytest.raises(HybridSearchError):
        anyio.run(
            hybrid_search,
            "question",
            5,
            10,
        )
