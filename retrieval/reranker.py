
"""Cross-encoder reranking for retrieved document chunks."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

from sentence_transformers import CrossEncoder

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_TOP_K = 10
MAX_CANDIDATES = 100
MAX_TOP_K = 100
MAX_QUERY_LENGTH = 4000
MAX_CONTENT_LENGTH = 12000


class RerankerError(RuntimeError):
    """Raised when cross-encoder reranking fails."""


@lru_cache(maxsize=1)
def _get_cross_encoder() -> CrossEncoder:
    """Load and cache the cross-encoder model on first use."""
    return CrossEncoder(
        DEFAULT_RERANKER_MODEL,
        max_length=512,
    )


def _validate_inputs(
    query: str,
    candidates: Sequence[Mapping[str, Any]],
    top_k: int,
) -> tuple[str, list[dict[str, Any]], int]:
    """Validate query, candidates, and requested output size."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    query = query.strip()

    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(
            f"Query cannot exceed {MAX_QUERY_LENGTH} characters."
        )

    if isinstance(candidates, (str, bytes, Mapping)):
        raise ValueError("Candidates must be a sequence of result mappings.")

    try:
        items = list(candidates)
    except TypeError as exc:
        raise ValueError("Candidates must be a sequence.") from exc

    if len(items) > MAX_CANDIDATES:
        raise ValueError(
            f"At most {MAX_CANDIDATES} candidates can be reranked at once."
        )

    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ValueError("top_k must be an integer.")

    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}.")

    validated: list[dict[str, Any]] = []

    for index, candidate in enumerate(items):
        if not isinstance(candidate, Mapping):
            raise ValueError(
                f"Candidate at position {index} must be a mapping."
            )

        content = candidate.get("content")

        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"Candidate at position {index} must contain non-empty "
                "'content'."
            )

        validated.append(dict(candidate))

    return query, validated, top_k


def rerank(
    query: str,
    candidates: Sequence[Mapping[str, Any]],
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """Rerank retrieved chunks by cross-encoder relevance scores.

    Candidates must already have been retrieved, for example by hybrid
    retrieval with RRF. Each candidate must contain a non-empty ``content``.

    The returned candidates preserve their original metadata and receive
    ``rerank_score`` and ``rerank_rank`` fields. Scores are raw model scores,
    not calibrated probabilities.
    """
    query, items, top_k = _validate_inputs(query, candidates, top_k)

    if not items:
        return []

    try:
        model = _get_cross_encoder()

        pairs = [
            [query, item["content"][:MAX_CONTENT_LENGTH]]
            for item in items
        ]

        raw_scores = model.predict(
            pairs,
            batch_size=16,
            show_progress_bar=False,
        )

        scores = [float(score) for score in raw_scores]

        if len(scores) != len(items):
            raise RerankerError(
                "The cross-encoder returned an unexpected number of scores."
            )

        if not all(math.isfinite(score) for score in scores):
            raise RerankerError(
                "The cross-encoder returned non-finite relevance scores."
            )

        scored_items: list[tuple[int, float, dict[str, Any]]] = []

        for index, (item, score) in enumerate(zip(items, scores)):
            result = dict(item)
            result["rerank_score"] = score
            scored_items.append((index, score, result))

        # Keep original order for ties to make results deterministic.
        scored_items.sort(key=lambda entry: (-entry[1], entry[0]))

        reranked: list[dict[str, Any]] = []

        for rank, (_, _, result) in enumerate(
            scored_items[:top_k],
            start=1,
        ):
            result["rerank_rank"] = rank
            reranked.append(result)

        return reranked

    except RerankerError:
        raise
    except Exception as exc:
        raise RerankerError(
            "Cross-encoder reranking failed. Check model availability, "
            "dependencies, and the input candidates."
        ) from exc


async def rerank_async(
    query: str,
    candidates: Sequence[Mapping[str, Any]],
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """Run reranking in a worker thread to avoid blocking the event loop."""
    return await asyncio.to_thread(rerank, query, candidates, top_k)
