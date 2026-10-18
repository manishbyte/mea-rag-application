
"""Retrieval metrics for the MEA RAG evaluation dataset."""

from __future__ import annotations

from typing import Any, Sequence


def _chunk_id(chunk: dict[str, Any]) -> str | None:
    value = chunk.get("chunk_id")
    return str(value).strip() if value is not None else None


def _page_number(chunk: dict[str, Any]) -> int | None:
    value = chunk.get("page_number")

    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def evaluate_retrieval(
    retrieved_chunks: Sequence[dict[str, Any]],
    relevant_chunk_ids: Sequence[str] | None = None,
    relevant_pages: Sequence[int] | None = None,
    k: int = 5,
) -> dict[str, Any]:
    """Evaluate retrieval using chunk IDs, with an explicit page fallback."""

    retrieved = list(retrieved_chunks or [])
    top_k = retrieved[: max(1, k)]

    relevant_ids = {
        str(value).strip()
        for value in (relevant_chunk_ids or [])
        if value is not None and str(value).strip()
    }

    relevant_page_set: set[int] = set()
    for value in relevant_pages or []:
        try:
            relevant_page_set.add(int(value))
        except (TypeError, ValueError):
            continue

    retrieved_ids = [_chunk_id(chunk) for chunk in retrieved]
    top_k_ids = [_chunk_id(chunk) for chunk in top_k]

    retrieved_pages = [_page_number(chunk) for chunk in retrieved]
    top_k_pages = [_page_number(chunk) for chunk in top_k]

    matched_ids = relevant_ids.intersection(
        value for value in retrieved_ids if value is not None
    )
    matched_top_k_ids = relevant_ids.intersection(
        value for value in top_k_ids if value is not None
    )

    retrieved_page_set = {
        page for page in retrieved_pages if page is not None
    }
    top_k_page_set = {
        page for page in top_k_pages if page is not None
    }

    matched_pages = relevant_page_set.intersection(retrieved_page_set)
    matched_top_k_pages = relevant_page_set.intersection(top_k_page_set)

    # Keep chunk-ID results separate from page-level results.
    chunk_id_recall = (
        len(matched_ids) / len(relevant_ids)
        if relevant_ids
        else None
    )

    page_recall = (
        len(matched_pages) / len(relevant_page_set)
        if relevant_page_set
        else None
    )

    chunk_id_hit = bool(matched_ids) if relevant_ids else None
    page_hit = bool(matched_pages) if relevant_page_set else None

    # Prefer exact chunk matching whenever it succeeds.
    # If it fails, use page matching only when reference pages exist.
    if matched_ids:
        evaluation_basis = "chunk_id"
        relevant_ranks = [
            rank
            for rank, chunk in enumerate(retrieved, start=1)
            if _chunk_id(chunk) in relevant_ids
        ]
        hit = True
        recall_at_k = (
            len(matched_top_k_ids) / len(relevant_ids)
        )
    elif relevant_page_set:
        evaluation_basis = "page_fallback"
        relevant_ranks = [
            rank
            for rank, chunk in enumerate(retrieved, start=1)
            if _page_number(chunk) in relevant_page_set
        ]
        hit = bool(matched_pages)
        recall_at_k = (
            len(matched_top_k_pages) / len(relevant_page_set)
        )
    elif relevant_ids:
        evaluation_basis = "chunk_id"
        relevant_ranks = []
        hit = False
        recall_at_k = 0.0
    else:
        evaluation_basis = "unavailable"
        relevant_ranks = []
        hit = None
        recall_at_k = None

    reciprocal_rank = (
        1.0 / relevant_ranks[0]
        if relevant_ranks
        else 0.0
    ) if evaluation_basis != "unavailable" else None

    return {
        "retrieved_count": len(retrieved),

        # Primary metrics use the selected evaluation basis.
        "hit": hit,
        "reciprocal_rank": reciprocal_rank,
        "recall_at_k": recall_at_k,
        "evaluation_basis": evaluation_basis,

        # Always report exact chunk-level metrics independently.
        "chunk_id_hit": chunk_id_hit,
        "chunk_id_recall": chunk_id_recall,
        "matched_chunk_ids": sorted(matched_ids),
        "missing_chunk_ids": sorted(relevant_ids - matched_ids),

        # Always report page-level metrics independently.
        "page_hit": page_hit,
        "page_recall": page_recall,
        "matched_pages": sorted(matched_pages),
        "missing_pages": sorted(relevant_page_set - matched_pages),

        "chunk_id_evaluation_available": bool(relevant_ids),
        "page_evaluation_available": bool(relevant_page_set),
    }
