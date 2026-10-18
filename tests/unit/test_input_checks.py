
"""Unit tests for input validation."""

import pytest

from guardrails.input_checks import (
    MAX_QUESTION_LENGTH,
    InputCheckResult,
    validate_question,
)


def test_valid_question_passes():
    result = validate_question("What are India's key diplomatic priorities?")

    assert isinstance(result, InputCheckResult)
    assert result.is_valid is True
    assert result.normalized_question == (
        "What are India's key diplomatic priorities?"
    )
    assert result.reason == ""


def test_leading_and_trailing_whitespace_is_removed():
    result = validate_question("  Summarize the annual report.  ")

    assert result.is_valid is True
    assert result.normalized_question == "Summarize the annual report."


@pytest.mark.parametrize(
    "question",
    [
        "",
        " ",
        "   \t\n ",
    ],
)
def test_empty_or_whitespace_question_is_rejected(question):
    result = validate_question(question)

    assert result.is_valid is False
    assert result.normalized_question == ""
    assert result.reason


@pytest.mark.parametrize(
    "question",
    [
        None,
        123,
        3.14,
        True,
        [],
        {},
        b"What is in the report?",
    ],
)
def test_non_string_input_is_rejected(question):
    result = validate_question(question)

    assert result.is_valid is False
    assert result.reason == "Question must be a string."


def test_question_at_maximum_length_passes():
    question = "a" * MAX_QUESTION_LENGTH

    result = validate_question(question)

    assert result.is_valid is True
    assert len(result.normalized_question) == MAX_QUESTION_LENGTH


def test_question_exceeding_maximum_length_is_rejected():
    question = "a" * (MAX_QUESTION_LENGTH + 1)

    result = validate_question(question)

    assert result.is_valid is False
    assert "cannot exceed" in result.reason


@pytest.mark.parametrize(
    "control_character",
    [
        "\x00",  # NUL
        "\x1b",  # ESC
        "\x07",  # Bell
        "\x08",  # Backspace
    ],
)
def test_disallowed_control_characters_are_rejected(control_character):
    result = validate_question(f"What is in the report?{control_character}")

    assert result.is_valid is False
    assert "control characters" in result.reason


@pytest.mark.parametrize(
    "whitespace",
    [
        "\n",
        "\r",
        "\t",
    ],
)
def test_common_whitespace_characters_are_allowed(whitespace):
    result = validate_question(f"What{whitespace}is in the report?")

    assert result.is_valid is True


def test_unicode_question_passes():
    result = validate_question("भारत की विदेश नीति की प्राथमिकताएँ क्या हैं?")

    assert result.is_valid is True
    assert result.normalized_question.startswith("भारत")


def test_question_with_surrounding_newlines_passes():
    result = validate_question("\nWhat does the report say?\n")

    assert result.is_valid is True
    assert result.normalized_question == "What does the report say?"
