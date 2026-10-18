
"""Rule-based prompt-injection detection for the MEA RAG application."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


MAX_INPUT_LENGTH = 4000


@dataclass(frozen=True, slots=True)
class InjectionDetectionResult:
    """Result of checking text for common prompt-injection patterns."""

    is_suspicious: bool
    reason: str = ""
    matched_patterns: tuple[str, ...] = ()


# Each entry has a stable label and a compiled pattern.
# Patterns target instruction-like behavior, not isolated keywords.
INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"\b(?:ignore|disregard|override|forget)\b"
            r".{0,100}\b(?:previous|prior|above|system|developer|all)"
            r"\b.{0,50}\b(?:instructions?|rules?|prompts?|messages?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "system_prompt_extraction",
        re.compile(
            r"\b(?:reveal|show|print|display|repeat|output| disclose|"
            r"expose|leak|tell me)\b"
            r".{0,80}\b(?:system prompt|hidden prompt|developer message|"
            r"system message|internal instructions|secret instructions)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "role_override",
        re.compile(
            r"\b(?:you are now|act as|pretend to be|roleplay as)\b"
            r".{0,60}\b(?:DAN|an unrestricted model|an unrestricted AI|"
            r"a model without restrictions|a system administrator)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "guardrail_bypass",
        re.compile(
            r"\b(?:bypass|disable|circumvent|evade|remove|turn off)\b"
            r".{0,60}\b(?:safety guardrails?|security rules?|"
            r"content policies|safeguards?|restrictions?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "replacement_instructions",
        re.compile(
            r"\b(?:follow|obey|execute)\b"
            r".{0,60}\b(?:these|the following|my|new)\b"
            r".{0,40}\b(?:instructions?|commands?|rules?)\b"
            r".{0,60}\b(?:instead|rather than)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "evidence_bypass",
        re.compile(
            r"\b(?:ignore|disregard|do not use|don't use|stop using)\b"
            r".{0,80}\b(?:retrieved evidence|provided evidence|"
            r"report passages|source documents|citations)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "secret_extraction",
        re.compile(
            r"\b(?:reveal|print|display|output|leak|exfiltrate|send me)\b"
            r".{0,60}\b(?:api keys?|access tokens?|passwords?|"
            r"credentials?|environment variables?|private keys?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
)

# Unicode formatting characters can be used to obscure text.
# Normalize them away before checking patterns.
FORMAT_CHARACTER_CATEGORY = "Cf"


def _normalize_for_detection(text: str) -> str:
    """Normalize Unicode and whitespace for pattern matching."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != FORMAT_CHARACTER_CATEGORY
    )
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def detect_prompt_injection(
    text: object,
) -> InjectionDetectionResult:
    """Detect common prompt-injection attempts in user-provided text.

    This is a heuristic security filter, not a complete classifier.
    It should be combined with input validation, prompt isolation,
    evidence checks, and output validation.
    """
    if not isinstance(text, str):
        return InjectionDetectionResult(
            is_suspicious=True,
            reason="Input must be a string.",
            matched_patterns=("invalid_input_type",),
        )

    if not text.strip():
        return InjectionDetectionResult(
            is_suspicious=False,
        )

    if len(text) > MAX_INPUT_LENGTH:
        return InjectionDetectionResult(
            is_suspicious=True,
            reason="Input exceeds the maximum detection length.",
            matched_patterns=("input_too_long",),
        )

    normalized = _normalize_for_detection(text)

    matched = tuple(
        label
        for label, pattern in INJECTION_PATTERNS
        if pattern.search(normalized)
    )

    if matched:
        return InjectionDetectionResult(
            is_suspicious=True,
            reason="Input matches one or more prompt-injection patterns.",
            matched_patterns=matched,
        )

    return InjectionDetectionResult(
        is_suspicious=False,
    )
