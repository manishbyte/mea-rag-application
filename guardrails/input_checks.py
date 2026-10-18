
"""Input validation for the MEA Annual Report RAG application."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass


MIN_QUESTION_LENGTH = 1
MAX_QUESTION_LENGTH = 4000


@dataclass(frozen=True, slots=True)
class InputCheckResult:
    """Result of validating a user question."""

    is_valid: bool
    normalized_question: str = ""
    reason: str = ""


def validate_question(question: object) -> InputCheckResult:
    """Validate and normalize a question before retrieval.

    Checks:
        - Input must be a string.
        - Leading and trailing whitespace is removed.
        - The question must not be empty.
        - The normalized question must not exceed the length limit.
        - Disallowed Unicode control characters are rejected.

    This function checks input structure, not malicious intent.
    Prompt-injection detection is handled by a separate guardrail.
    """
    if not isinstance(question, str):
        return InputCheckResult(
            is_valid=False,
            reason="Question must be a string.",
        )

    normalized = question.strip()

    if not normalized:
        return InputCheckResult(
            is_valid=False,
            reason="Question must not be empty.",
        )

    if len(normalized) < MIN_QUESTION_LENGTH:
        return InputCheckResult(
            is_valid=False,
            reason="Question is too short.",
        )

    if len(normalized) > MAX_QUESTION_LENGTH:
        return InputCheckResult(
            is_valid=False,
            reason=(
                f"Question cannot exceed {MAX_QUESTION_LENGTH} characters."
            ),
        )

    for character in normalized:
        # Allow ordinary whitespace, including line breaks, but reject
        # other Unicode control characters such as NUL and ESC.
        if unicodedata.category(character) == "Cc":
            if character not in ("\t", "\n", "\r"):
                return InputCheckResult(
                    is_valid=False,
                    reason="Question contains disallowed control characters.",
                )

    if not any(not character.isspace() for character in normalized):
        return InputCheckResult(
            is_valid=False,
            reason="Question must contain non-whitespace characters.",
        )

    return InputCheckResult(
        is_valid=True,
        normalized_question=normalized,
    )
