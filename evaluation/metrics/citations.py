"""Citation integrity metrics for generated answers."""

from __future__ import annotations

import re
from typing import Any, Sequence


_CITATION_PATTERN = re.compile(r"\[(S\d+)\]", re.IGNORECASE)


def evaluate_citations(
    answer: str,
    citations: Sequence[dict[str, Any]] | None,
    retrieved_chunks: Sequence[dict[str, Any]] | None,
    relevant_pages: Sequence[int] | None = None,
) -> dict[str, Any]:
    citations = list(citations or [])
    chunks = list(retrieved_chunks or [])
    answer_markers = {
        marker.upper() for marker in _CITATION_PATTERN.findall(answer or "")
    }

    def citation_id(item: dict[str, Any]) -> str | None:
        value = (
            item.get("source_id")
            or item.get("source")
            or item.get("citation_id")
            or item.get("id")
        )
        if value is None:
            return None
        value = str(value).strip().upper()
        if value and not value.startswith("S"):
            value = f"S{value}"
        return value or None

    def page_number(item: dict[str, Any]) -> int | None:
        value = item.get("page_number", item.get("page"))
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    metadata_ids = {
        value for value in (citation_id(item) for item in citations) if value
    }
    retrieved_ids = {
        f"S{index}" for index in range(1, len(chunks) + 1)
    }
    citation_pages = {
        page for page in (page_number(item) for item in citations)
        if page is not None
    }
    retrieved_pages = {
        page for page in (page_number(item) for item in chunks)
        if page is not None
    }
    relevant_page_set = {int(page) for page in (relevant_pages or [])}

    cited_markers = answer_markers
    matched_metadata_markers = cited_markers.intersection(metadata_ids)
    markers_without_metadata = cited_markers - metadata_ids
    metadata_not_used = metadata_ids - cited_markers
    invalid_pages = citation_pages - retrieved_pages
    relevant_page_hits = citation_pages.intersection(relevant_page_set)

    return {
        "citation_marker_count": len(cited_markers),
        "citation_metadata_count": len(citations),
        "citation_marker_metadata_precision": (
            len(matched_metadata_markers) / len(cited_markers)
            if cited_markers else None
        ),
        "markers_without_metadata": sorted(markers_without_metadata),
        "metadata_not_cited": sorted(metadata_not_used),
        "citation_pages_not_in_retrieval": sorted(invalid_pages),
        "relevant_pages_cited": sorted(relevant_page_hits),
        "relevant_page_citation_recall": (
            len(relevant_page_hits) / len(relevant_page_set)
            if relevant_page_set else None
        ),
        "has_citations": bool(cited_markers or citations),
    }