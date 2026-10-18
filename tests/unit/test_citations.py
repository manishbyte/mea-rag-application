
"""Tests for citation integrity validation."""

from generation.citations import (
    build_source_map,
    extract_citation_ids,
    validate_citations,
)


def make_chunk(chunk_id="chunk-1", page_number=12):
    return {
        "chunk_id": chunk_id,
        "document_id": "doc-1",
        "page_number": page_number,
        "content": "The Ministry held diplomatic consultations.",
        "document_title": "MEA Annual Report 2025",
        "filename": "mea_annual_report_2025.pdf",
    }


def make_citation(source_id="S1", chunk_id="chunk-1", page_number=12):
    return {
        "source_id": source_id,
        "chunk_id": chunk_id,
        "document_id": "doc-1",
        "page_number": page_number,
        "document_title": "MEA Annual Report 2025",
        "filename": "mea_annual_report_2025.pdf",
    }


def test_build_source_map():
    chunks = [make_chunk(), make_chunk("chunk-2", 13)]

    source_map = build_source_map(chunks)

    assert list(source_map) == ["S1", "S2"]
    assert source_map["S2"]["chunk_id"] == "chunk-2"


def test_extract_unique_citation_ids():
    assert extract_citation_ids("First [S1], then [S2], then [S1].") == [
        "S1",
        "S2",
    ]


def test_valid_citation_passes():
    passed, errors = validate_citations(
        "The Ministry held consultations [S1].",
        [make_citation()],
        [make_chunk()],
    )

    assert passed is True
    assert errors == []


def test_unknown_source_fails():
    passed, errors = validate_citations(
        "The report says this [S9].",
        [make_citation("S9", "unknown-chunk")],
        [make_chunk()],
    )

    assert passed is False
    assert any("does not match retrieved evidence" in e for e in errors)
    assert any("not in the retrieved evidence" in e for e in errors)


def test_mismatched_chunk_metadata_fails():
    passed, errors = validate_citations(
        "The report says this [S1].",
        [make_citation(chunk_id="different-chunk")],
        [make_chunk()],
    )

    assert passed is False
    assert any("mismatched chunk metadata" in e for e in errors)


def test_missing_citation_metadata_fails():
    passed, errors = validate_citations(
        "The report says this [S1].",
        [],
        [make_chunk()],
    )

    assert passed is False
    assert any("without matching citation metadata" in e for e in errors)


def test_unreferenced_citation_metadata_fails():
    passed, errors = validate_citations(
        "The report says this.",
        [make_citation()],
        [make_chunk()],
    )

    assert passed is False
    assert any("not referenced in the answer" in e for e in errors)


def test_duplicate_citation_metadata_fails():
    passed, errors = validate_citations(
        "The report says this [S1].",
        [make_citation(), make_citation()],
        [make_chunk()],
    )

    assert passed is False
    assert any("Duplicate citation metadata" in e for e in errors)
