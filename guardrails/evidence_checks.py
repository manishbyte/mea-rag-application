
"""Validate retrieved evidence before answer generation."""

from __future__ import annotations

from typing import Any


def assess_evidence(
    chunks: list[dict[str, Any]],
) -> tuple[bool, str]:
    """Check whether retrieval returned usable source passages.

    This is a structural check, not a guarantee of semantic relevance.
    """
    if not isinstance(chunks, list):
        return False, "Retrieved evidence must be a list."

    if not chunks:
        return False, "No evidence was retrieved from the report."

    usable_count = 0

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue

        content = chunk.get("content")
        if not isinstance(content, str) or not content.strip():
            continue

        # Require source identity and page information to be present.
        has_source = any(
            chunk.get(key) is not None
            for key in ("document_id", "document_title", "title", "filename")
        )
        has_page = chunk.get("page_number") is not None

        if has_source and has_page:
            usable_count += 1

    if usable_count == 0:
        return (
            False,
            "Retrieved results contain no passages with usable text, "
            "source identity, and page information.",
        )

    return True, f"Found {usable_count} usable evidence passage(s)."
