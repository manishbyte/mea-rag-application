
"""Tests for PDF loading and text extraction."""

import fitz
import pytest

from ingestion.pdf_loader import (
    EmptyPDFError,
    InvalidPDFError,
    load_pdf,
)


def create_pdf(path, page_texts):
    """Create a small PDF fixture."""
    document = fitz.open()

    for text in page_texts:
        page = document.new_page()
        page.insert_text((72, 72), text)

    document.save(path)
    document.close()

    return path


def test_load_pdf_extracts_text_and_preserves_page_numbers(tmp_path):
    pdf_path = create_pdf(
        tmp_path / "sample.pdf",
        ["First page content", "Second page content"],
    )

    result = load_pdf(pdf_path)

    assert result.filename == "sample.pdf"
    assert result.page_count == 2
    assert len(result.pages) == 2
    assert result.pages[0].page_number == 1
    assert result.pages[1].page_number == 2
    assert "First page content" in result.pages[0].text
    assert "Second page content" in result.pages[1].text
    assert "First page content" in result.full_text


def test_load_pdf_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pdf(tmp_path / "missing.pdf")


def test_load_pdf_rejects_wrong_extension(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Not a PDF", encoding="utf-8")

    with pytest.raises(InvalidPDFError):
        load_pdf(file_path)


def test_load_pdf_rejects_empty_file(tmp_path):
    file_path = tmp_path / "empty.pdf"
    file_path.touch()

    with pytest.raises(InvalidPDFError):
        load_pdf(file_path)


def test_load_pdf_rejects_invalid_pdf_content(tmp_path):
    file_path = tmp_path / "invalid.pdf"
    file_path.write_bytes(b"This is not a PDF document")

    with pytest.raises(InvalidPDFError):
        load_pdf(file_path)


def test_load_pdf_rejects_scanned_or_empty_text_pdf(tmp_path):
    pdf_path = create_pdf(tmp_path / "blank.pdf", ["   "])

    with pytest.raises(EmptyPDFError):
        load_pdf(pdf_path)


def test_load_pdf_validates_minimum_text_length(tmp_path):
    pdf_path = create_pdf(tmp_path / "sample.pdf", ["Some text"])

    with pytest.raises(ValueError):
        load_pdf(pdf_path, min_text_chars=0)
