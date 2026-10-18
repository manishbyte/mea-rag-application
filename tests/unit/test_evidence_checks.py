
"""Unit tests for evidence validation."""

from guardrails.evidence_checks import assess_evidence


def test_empty_evidence_is_insufficient():
    sufficient, reason = assess_evidence([])

    assert sufficient is False
    assert "No evidence" in reason


def test_evidence_without_content_is_insufficient():
    chunks = [
        {
            "document_id": "doc-1",
            "page_number": 10,
            "content": "   ",
        }
    ]

    sufficient, _ = assess_evidence(chunks)

    assert sufficient is False


def test_evidence_without_source_metadata_is_insufficient():
    chunks = [
        {
            "content": "India's foreign policy emphasizes regional cooperation.",
            "page_number": 10,
        }
    ]

    sufficient, _ = assess_evidence(chunks)

    assert sufficient is False


def test_valid_evidence_is_sufficient():
    chunks = [
        {
            "document_id": "doc-1",
            "document_title": "Annual Report 2025",
            "page_number": 10,
            "content": "India's foreign policy emphasizes regional cooperation.",
        }
    ]

    sufficient, reason = assess_evidence(chunks)

    assert sufficient is True
    assert "1 usable evidence passage" in reason


def test_invalid_chunk_does_not_crash_validation():
    sufficient, _ = assess_evidence([None, "invalid", {}])

    assert sufficient is False
