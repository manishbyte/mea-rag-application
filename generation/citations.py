
"""Citation integrity checks for MEA RAG answers."""

from __future__ import annotations

import re
from typing import Any


CITATION_PATTERN = re.compile(r"\[(S[1-9]\d*)\]")


def build_source_map(
    retrieved_chunks: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Map source identifiers to chunks using the generation prompt's order.

    Source numbering must remain consistent with build_evidence_context()
    in generation/prompts.py.
    """
    source_map: dict[str, dict[str, Any]] = {}

    for index, chunk in enumerate(retrieved_chunks, start=1):
        if not isinstance(chunk, dict):
            continue

        content = chunk.get("content")
        if not isinstance(content, str) or not content.strip():
            continue

        source_map[f"S{index}"] = chunk

    return source_map


def extract_citation_ids(answer: str) -> list[str]:
    """Extract unique citation identifiers in their order of appearance."""
    return list(dict.fromkeys(CITATION_PATTERN.findall(answer)))


def validate_citations(
    answer: str,
    citations: list[dict[str, Any]],
    retrieved_chunks: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    """Validate answer citation references against retrieved evidence.

    Returns:
        A boolean indicating whether validation passed and a list of errors.
    """
    errors: list[str] = []

    if not isinstance(answer, str):
        return False, ["Answer must be a string."]

    if not isinstance(citations, list):
        return False, ["Citations must be a list."]

    if not isinstance(retrieved_chunks, list):
        return False, ["Retrieved evidence must be a list."]

    source_map = build_source_map(retrieved_chunks)
    answer_ids = extract_citation_ids(answer)

    citation_map: dict[str, dict[str, Any]] = {}

    for citation in citations:
        if not isinstance(citation, dict):
            errors.append("Each citation must be an object.")
            continue

        source_id = citation.get("source_id")

        if not isinstance(source_id, str) or not source_id:
            errors.append("A citation is missing its source_id.")
            continue

        if source_id in citation_map:
            errors.append(f"Duplicate citation metadata for [{source_id}].")
            continue

        citation_map[source_id] = citation

        if source_id not in source_map:
            errors.append(
                f"Citation [{source_id}] does not match retrieved evidence."
            )
            continue

        chunk = source_map[source_id]

        # If IDs are supplied, they must match the referenced chunk.
        expected_chunk_id = chunk.get("chunk_id")
        actual_chunk_id = citation.get("chunk_id")

        if (
            expected_chunk_id is not None
            and actual_chunk_id is not None
            and str(expected_chunk_id) != str(actual_chunk_id)
        ):
            errors.append(
                f"Citation [{source_id}] has mismatched chunk metadata."
            )

        expected_document_id = chunk.get("document_id")
        actual_document_id = citation.get("document_id")

        if (
            expected_document_id is not None
            and actual_document_id is not None
            and str(expected_document_id) != str(actual_document_id)
        ):
            errors.append(
                f"Citation [{source_id}] has mismatched document metadata."
            )

        expected_page = chunk.get("page_number")
        actual_page = citation.get("page_number")

        if (
            expected_page is not None
            and actual_page is not None
            and expected_page != actual_page
        ):
            errors.append(
                f"Citation [{source_id}] has a mismatched page number."
            )

    for source_id in answer_ids:
        if source_id not in source_map:
            errors.append(
                f"Answer references [{source_id}], which is not in "
                "the retrieved evidence."
            )
        elif source_id not in citation_map:
            errors.append(
                f"Answer references [{source_id}] without matching "
                "citation metadata."
            )

    for source_id in citation_map:
        if source_id not in answer_ids:
            errors.append(
                f"Citation metadata for [{source_id}] is not referenced "
                "in the answer."
            )

    return not errors, errors
