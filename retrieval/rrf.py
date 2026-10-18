
"""Reciprocal Rank Fusion for combining ranked retrieval results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

DEFAULT_RRF_K = 60
DEFAULT_TOP_K = 10
MAX_TOP_K = 100


class RRFError(ValueError):
    """Raised when reciprocal rank fusion receives invalid input."""


def _validate_parameters(top_k: int, rrf_k: int) -> None:
    """Validate output size and RRF smoothing constant."""
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise RRFError("top_k must be an integer.")

    if not 1 <= top_k <= MAX_TOP_K:
        raise RRFError(f"top_k must be between 1 and {MAX_TOP_K}.")

    if not isinstance(rrf_k, int) or isinstance(rrf_k, bool):
        raise RRFError("rrf_k must be an integer.")

    if rrf_k < 1:
        raise RRFError("rrf_k must be at least 1.")


def _get_chunk_id(result: Mapping[str, Any]) -> str:
    """Return a stable chunk identifier."""
    chunk_id = result.get("chunk_id")

    if chunk_id is None:
        raise RRFError("Every retrieval result must contain chunk_id.")

    normalized_id = str(chunk_id).strip()

    if not normalized_id:
        raise RRFError("chunk_id cannot be empty.")

    return normalized_id


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Mapping[str, Any]]],
    top_k: int = DEFAULT_TOP_K,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[dict[str, Any]]:
    """Fuse multiple ranked result lists using reciprocal rank fusion.

    Each input list must already be ordered from most to least relevant.
    Results are deduplicated by chunk_id.

    Args:
        ranked_lists: Ranked results, such as vector results and keyword
            results. Each result must contain a chunk_id.
        top_k: Maximum number of fused chunks to return.
        rrf_k: Positive smoothing constant used in the RRF formula.

    Returns:
        Fused results ordered by descending rrf_score. Each result includes
        rrf_score, retrieval_method="hybrid", and source_methods.
    """
    _validate_parameters(top_k, rrf_k)

    if isinstance(ranked_lists, (str, bytes)):
        raise RRFError("ranked_lists must be a sequence of result lists.")

    try:
        lists = list(ranked_lists)
    except TypeError as exc:
        raise RRFError(
            "ranked_lists must be a sequence of result lists."
        ) from exc

    fused: dict[str, dict[str, Any]] = {}
    seen_in_list: list[set[str]] = []

    for ranked_list in lists:
        if isinstance(ranked_list, (str, bytes, Mapping)):
            raise RRFError(
                "Each item in ranked_lists must be a sequence of results."
            )

        try:
            results = list(ranked_list)
        except TypeError as exc:
            raise RRFError(
                "Each item in ranked_lists must be a sequence of results."
            ) from exc

        seen_ids: set[str] = set()
        seen_in_list.append(seen_ids)

        for rank, result in enumerate(results, start=1):
            if not isinstance(result, Mapping):
                raise RRFError("Each retrieval result must be a mapping.")

            chunk_id = _get_chunk_id(result)

            # A chunk contributes at most once per retrieval list.
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)

            if chunk_id not in fused:
                fused[chunk_id] = dict(result)
                fused[chunk_id]["rrf_score"] = 0.0
                fused[chunk_id]["source_methods"] = []

            fused_result = fused[chunk_id]
            fused_result["rrf_score"] += 1.0 / (rrf_k + rank)

            method = result.get("retrieval_method")
            if isinstance(method, str) and method:
                if method not in fused_result["source_methods"]:
                    fused_result["source_methods"].append(method)

    ranked_results = sorted(
        fused.values(),
        key=lambda item: (
            -item["rrf_score"],
            str(item["chunk_id"]),
        ),
    )

    output: list[dict[str, Any]] = []

    for result in ranked_results[:top_k]:
        fused_result = dict(result)
        fused_result["retrieval_method"] = "hybrid"
        output.append(fused_result)

    return output
