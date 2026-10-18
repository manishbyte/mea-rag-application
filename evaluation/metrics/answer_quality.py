"""Reference-answer comparison metrics without an external judge model."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


def _tokens(text: str) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", text.casefold())


def _normalize(text: str) -> str:
    return " ".join(_tokens(text))


def _f1(predicted: list[str], expected: list[str]) -> float:
    if not predicted or not expected:
        return 1.0 if predicted == expected else 0.0

    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(left: list[str], right: list[str]) -> int:
    """Return longest common subsequence length using O(len(right)) memory."""
    if len(left) < len(right):
        left, right = right, left

    previous = [0] * (len(right) + 1)

    for token_left in left:
        current = [0]
        for index, token_right in enumerate(right, start=1):
            if token_left == token_right:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current

    return previous[-1]


def evaluate_answer_quality(
    answer: str,
    expected_answer: str | None,
) -> dict[str, Any]:
    """Calculate exact match, token F1, and ROUGE-L F1."""
    if not expected_answer:
        return {
            "reference_available": False,
            "exact_match": None,
            "token_f1": None,
            "rouge_l_f1": None,
        }

    predicted_tokens = _tokens(answer or "")
    expected_tokens = _tokens(expected_answer)

    predicted_normalized = _normalize(answer or "")
    expected_normalized = _normalize(expected_answer)

    lcs = _lcs_length(predicted_tokens, expected_tokens)
    if predicted_tokens and expected_tokens:
        precision = lcs / len(predicted_tokens)
        recall = lcs / len(expected_tokens)
        rouge_l = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    else:
        rouge_l = 1.0 if predicted_tokens == expected_tokens else 0.0

    return {
        "reference_available": True,
        "exact_match": predicted_normalized == expected_normalized,
        "token_f1": round(_f1(predicted_tokens, expected_tokens), 4),
        "rouge_l_f1": round(rouge_l, 4),
    }