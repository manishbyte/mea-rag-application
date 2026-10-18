
"""Unit tests for Reciprocal Rank Fusion."""

import pytest

from retrieval.rrf import (
    DEFAULT_RRF_K,
    RRFError,
    reciprocal_rank_fusion,
)


def make_result(chunk_id, method, content=None):
    return {
        "chunk_id": chunk_id,
        "content": content or f"Content for {chunk_id}",
        "retrieval_method": method,
    }


def test_rrf_combines_results_from_multiple_retrievers():
    vector_results = [
        make_result("a", "vector"),
        make_result("b", "vector"),
    ]
    keyword_results = [
        make_result("c", "keyword"),
        make_result("a", "keyword"),
    ]

    results = reciprocal_rank_fusion(
        [vector_results, keyword_results]
    )

    assert results[0]["chunk_id"] == "a"
    assert results[0]["retrieval_method"] == "hybrid"
    assert results[0]["source_methods"] == ["vector", "keyword"]


def test_rrf_uses_expected_formula():
    results = reciprocal_rank_fusion(
        [[make_result("a", "vector")]],
        rrf_k=DEFAULT_RRF_K,
    )

    assert results[0]["rrf_score"] == pytest.approx(
        1 / (DEFAULT_RRF_K + 1)
    )


def test_rrf_rewards_chunks_ranked_high_in_both_lists():
    results = reciprocal_rank_fusion(
        [
            [
                make_result("shared", "vector"),
                make_result("vector_only", "vector"),
            ],
            [
                make_result("shared", "keyword"),
                make_result("keyword_only", "keyword"),
            ],
        ]
    )

    assert results[0]["chunk_id"] == "shared"
    assert results[0]["rrf_score"] == pytest.approx(
        2 / (DEFAULT_RRF_K + 1)
    )


def test_rrf_deduplicates_chunks():
    results = reciprocal_rank_fusion(
        [
            [
                make_result("a", "vector"),
                make_result("a", "vector"),
            ]
        ]
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "a"


def test_rrf_preserves_chunk_metadata():
    result = make_result("a", "vector", "MEA report content")
    result["page_number"] = 25
    result["document_title"] = "MEA Annual Report 2025"

    results = reciprocal_rank_fusion([[result]])

    assert results[0]["content"] == "MEA report content"
    assert results[0]["page_number"] == 25
    assert results[0]["document_title"] == "MEA Annual Report 2025"


def test_rrf_respects_top_k():
    results = reciprocal_rank_fusion(
        [[
            make_result("a", "vector"),
            make_result("b", "vector"),
            make_result("c", "vector"),
        ]],
        top_k=2,
    )

    assert len(results) == 2


def test_rrf_handles_empty_lists():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_rejects_missing_chunk_id():
    with pytest.raises(RRFError, match="chunk_id"):
        reciprocal_rank_fusion([[{"content": "Missing ID"}]])


@pytest.mark.parametrize(
    ("top_k", "rrf_k"),
    [
        (0, 60),
        (101, 60),
        (True, 60),
        (10, 0),
        (10, True),
    ],
)
def test_rrf_rejects_invalid_parameters(top_k, rrf_k):
    with pytest.raises(RRFError):
        reciprocal_rank_fusion(
            [[make_result("a", "vector")]],
            top_k=top_k,
            rrf_k=rrf_k,
        )
