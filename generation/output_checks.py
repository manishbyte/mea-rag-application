
"""Deterministic output validation for MEA RAG answers."""

from __future__ import annotations

from typing import Any

from generation.citations import validate_citations


MAX_ANSWER_LENGTH = 20000


def validate_output(
    answer: str,
    citations: list[dict[str, Any]],
    retrieved_chunks: list[dict[str, Any]],
    abstained: bool = False,
) -> tuple[bool, list[str]]:
    """Validate the final answer and its citations.

    This performs structural and citation-integrity checks. It does not
    establish semantic entailment between every claim and its evidence.
    """
    errors: list[str] = []

    if not isinstance(answer, str):
        return False, ["Answer must be a string."]

    if not answer.strip():
        errors.append("Answer must not be empty.")
    elif len(answer) > MAX_ANSWER_LENGTH:
        errors.append(
            f"Answer exceeds the maximum length of {MAX_ANSWER_LENGTH} "
            "characters."
        )

    if not isinstance(abstained, bool):
        errors.append("abstained must be a boolean.")

    if not isinstance(citations, list):
        errors.append("Citations must be a list.")
    if not isinstance(retrieved_chunks, list):
        errors.append("Retrieved evidence must be a list.")

    if errors:
        return False, errors

    if abstained:
        if citations:
            errors.append("An abstained answer must not include citations.")

        if answer.strip() and not any(
            phrase in answer.casefold()
            for phrase in (
                "insufficient",
                "cannot answer",
                "can't answer",
                "unable to answer",
                "not enough evidence",
                "not answer reliably",
            )
        ):
            errors.append(
                "An abstained answer must clearly communicate that "
                "the evidence is insufficient."
            )

        return not errors, errors

    if not retrieved_chunks:
        errors.append(
            "A non-abstained answer requires retrieved evidence."
        )

    if not citations:
        errors.append(
            "A non-abstained answer must include at least one citation."
        )

    citations_valid, citation_errors = validate_citations(
        answer=answer,
        citations=citations,
        retrieved_chunks=retrieved_chunks,
    )
    if not citations_valid:
        errors.extend(citation_errors)

    return not errors, errors
