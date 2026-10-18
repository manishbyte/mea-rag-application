
"""Hybrid retrieval combining vector search, keyword search, RRF, and reranking."""

from __future__ import annotations

import asyncio
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from retrieval.embeddings import embed_query_async
from retrieval.keyword_search import keyword_search
from retrieval.reranker import rerank_async
from retrieval.rrf import reciprocal_rank_fusion
from retrieval.vector_search import vector_search

DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_K = 20
MAX_TOP_K = 100
MAX_CANDIDATE_K = 100


class HybridSearchError(RuntimeError):
    """Raised when hybrid retrieval fails."""


def _validate_inputs(
    query: str,
    top_k: int,
    candidate_k: int,
) -> tuple[str, int, int]:
    """Validate the query and retrieval limits."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    query = query.strip()

    if len(query) > 4000:
        raise ValueError("Query cannot exceed 4000 characters.")

    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ValueError("top_k must be an integer.")

    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}.")

    if not isinstance(candidate_k, int) or isinstance(candidate_k, bool):
        raise ValueError("candidate_k must be an integer.")

    if not 1 <= candidate_k <= MAX_CANDIDATE_K:
        raise ValueError(
            f"candidate_k must be between 1 and {MAX_CANDIDATE_K}."
        )

    if candidate_k < top_k:
        raise ValueError("candidate_k must be greater than or equal to top_k.")

    return query, top_k, candidate_k


async def hybrid_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    *,
    session: AsyncSession | None = None,
    use_reranker: bool = True,
) -> list[dict[str, Any]]:
    """Retrieve relevant chunks using hybrid search.

    Pipeline:
        1. Embed the query.
        2. Run vector and keyword retrieval.
        3. Fuse ranked results with Reciprocal Rank Fusion.
        4. Optionally rerank the fused candidates with a cross-encoder.

    Args:
        query: User's question.
        top_k: Number of final chunks to return.
        candidate_k: Number of candidates to retain after RRF before
            reranking. Also used as the limit for each initial retriever.
        session: Optional existing SQLAlchemy AsyncSession. When supplied,
            database searches execute sequentially using that session.
        use_reranker: Whether to apply cross-encoder reranking.

    Returns:
        Ranked chunk dictionaries with source metadata and retrieval scores.
        When reranking is enabled, results contain rerank_score and
        rerank_rank. RRF scores remain available as rrf_score.
    """
    query, top_k, candidate_k = _validate_inputs(
        query,
        top_k,
        candidate_k,
    )

    if not isinstance(use_reranker, bool):
        raise ValueError("use_reranker must be a boolean.")

    try:
        query_embedding = await embed_query_async(query)

        if session is None:
            # Each retriever creates its own session, so the two searches
            # can safely run concurrently.
            vector_results, keyword_results = await asyncio.gather(
                vector_search(query_embedding, top_k=candidate_k),
                keyword_search(query, top_k=candidate_k),
            )
        else:
            # An AsyncSession must not be used by concurrent operations.
            vector_results = await vector_search(
                query_embedding,
                top_k=candidate_k,
                session=session,
            )
            keyword_results = await keyword_search(
                query,
                top_k=candidate_k,
                session=session,
            )

        fused_results = reciprocal_rank_fusion(
            [vector_results, keyword_results],
            top_k=candidate_k,
        )

        if not fused_results:
            return []

        if use_reranker:
            return await rerank_async(
                query=query,
                candidates=fused_results,
                top_k=top_k,
            )

        # Without reranking, RRF order is retained.
        return fused_results[:top_k]

    except (ValueError, TypeError):
        raise
    except Exception as exc:
        raise HybridSearchError(
            "Hybrid retrieval failed. Check the embedding model, database "
            "retrievers, RRF configuration, and reranker."
        ) from exc
