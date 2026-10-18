
"""Tests for final answer output validation."""

from guardrails.output_checks import validate_output


def make_chunk():
    return {
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "page_number": 12,
        "content": "The Ministry held diplomatic consultations.",
        "document_title": "MEA Annual Report 2025",
        "filename": "mea_annual_report_2025.pdf",
    }


def make_citation():
    return {
        "source_id": "S1",
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "page_number": 12,
        "document_title": "MEA Annual Report 2025",
        "filename": "mea_annual_report_2025.pdf",
    }


def test_valid_grounded_answer_passes():
    passed, errors = validate_output(
        answer="The Ministry held diplomatic consultations [S1].",
        citations=[make_citation()],
        retrieved_chunks=[make_chunk()],
    )

    assert passed is True
    assert errors == []


def test_empty_answer_fails():
    passed, errors = validate_output(
        answer=" ",
        citations=[],
        retrieved_chunks=[],
        abstained=True,
    )

    assert passed is False
    assert any("must not be empty" in e for e in errors)


def test_answer_without_citations_fails():
    passed, errors = validate_output(
        answer="The Ministry held diplomatic consultations.",
        citations=[],
        retrieved_chunks=[make_chunk()],
    )

    assert passed is False
    assert any("at least one citation" in e for e in errors)


def test_abstention_without_citations_passes():
    passed, errors = validate_output(
        answer=(
            "I cannot answer this question reliably because "
            "the evidence is insufficient."
        ),
        citations=[],
        retrieved_chunks=[],
        abstained=True,
    )

    assert passed is True
    assert errors == []


def test_abstention_with_citations_fails():
    passed, errors = validate_output(
        answer="The evidence is insufficient.",
        citations=[make_citation()],
        retrieved_chunks=[make_chunk()],
        abstained=True,
    )

    assert passed is False
    assert any("must not include citations" in e for e in errors)


def test_invalid_source_reference_fails():
    passed, errors = validate_output(
        answer="The Ministry did something [S7].",
        citations=[
            {
                **make_citation(),
                "source_id": "S7",
            }
        ],
        retrieved_chunks=[make_chunk()],
    )

    assert passed is False
    assert any("does not match retrieved evidence" in e for e in errors)


def test_non_boolean_abstained_fails():
    passed, errors = validate_output(
        answer="The Ministry held consultations [S1].",
        citations=[make_citation()],
        retrieved_chunks=[make_chunk()],
        abstained="false",
    )

    assert passed is False
    assert any("must be a boolean" in e for e in errors)


def test_oversized_answer_fails():
    passed, errors = validate_output(
        answer="A" * 20001,
        citations=[make_citation()],
        retrieved_chunks=[make_chunk()],
    )

    assert passed is False
    assert any("maximum length" in e for e in errors)
