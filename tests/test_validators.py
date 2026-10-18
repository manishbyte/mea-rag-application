"""Tests for PDF ingestion validation."""

import fitz
import pytest

from ingestion.pdf_loader import LoadedPDF, PDFPage
from ingestion.validators import (
    IngestionValidationError,
    validate_loaded_pdf,
    validate_pdf_file,
)


def create_pdf(path, page_texts):
    document = fitz.open()

    for text in page_texts:
        page = document.new_page()
        page.insert_text((72, 72), text)

    document.save(path)
    document.close()
    return path


def test_validate_pdf_file_accepts_valid_pdf(tmp_path):
    path = create_pdf(tmp_path / "report.pdf", ["Annual report"])

    assert validate_pdf_file(path) == path.resolve()


def test_validate_pdf_file_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_pdf_file(tmp_path / "missing.pdf")


def test_validate_pdf_file_rejects_wrong_extension(tmp_path):
    path = tmp_path / "report.txt"
    path.write_text("Not a PDF", encoding="utf-8")

    with pytest.raises(IngestionValidationError):
        validate_pdf_file(path)


def test_validate_pdf_file_rejects_invalid_signature(tmp_path):
    path = tmp_path / "report.pdf"
    path.write_bytes(b"This is not a PDF")

    with pytest.raises(IngestionValidationError):
        validate_pdf_file(path)


def test_validate_pdf_file_rejects_oversized_file(tmp_path):
    path = create_pdf(tmp_path / "report.pdf", ["Annual report"])

    with pytest.raises(IngestionValidationError, match="exceeds"):
        validate_pdf_file(path, max_size_bytes=10)


def test_validate_pdf_file_rejects_zero_size_limit(tmp_path):
    path = create_pdf(tmp_path / "report.pdf", ["Annual report"])

    with pytest.raises(ValueError):
        validate_pdf_file(path, max_size_bytes=0)


def test_validate_loaded_pdf_accepts_valid_document(tmp_path):
    path = create_pdf(
        tmp_path / "report.pdf",
        ["First page", "Second page"],
    )

    loaded = LoadedPDF(
        file_path=path.resolve(),
        filename=path.name,
        page_count=2,
        pages=(
            PDFPage(page_number=1, text="First page"),
            PDFPage(page_number=2, text="Second page"),
        ),
    )

    validate_loaded_pdf(loaded)


def test_validate_loaded_pdf_rejects_mismatched_page_count(tmp_path):
    path = tmp_path / "report.pdf"

    loaded = LoadedPDF(
        file_path=path,
        filename="report.pdf",
        page_count=2,
        pages=(PDFPage(page_number=1, text="First page"),),
    )

    with pytest.raises(
        IngestionValidationError,
        match="does not match",
    ):
        validate_loaded_pdf(loaded)


def test_validate_loaded_pdf_rejects_nonsequential_pages(tmp_path):
    path = tmp_path / "report.pdf"

    loaded = LoadedPDF(
        file_path=path,
        filename="report.pdf",
        page_count=2,
        pages=(
            PDFPage(page_number=1, text="First page"),
            PDFPage(page_number=3, text="Third page"),
        ),
    )

    with pytest.raises(IngestionValidationError, match="sequential"):
        validate_loaded_pdf(loaded)


def test_validate_loaded_pdf_rejects_insufficient_text(tmp_path):
    path = tmp_path / "report.pdf"

    loaded = LoadedPDF(
        file_path=path,
        filename="report.pdf",
        page_count=1,
        pages=(PDFPage(page_number=1, text="   "),),
    )

    with pytest.raises(IngestionValidationError, match="enough"):
        validate_loaded_pdf(loaded)


def test_validate_loaded_pdf_rejects_too_many_pages(tmp_path):
    path = tmp_path / "report.pdf"

    loaded = LoadedPDF(
        file_path=path,
        filename="report.pdf",
        page_count=2,
        pages=(
            PDFPage(page_number=1, text="First page"),
            PDFPage(page_number=2, text="Second page"),
        ),
    )

    with pytest.raises(IngestionValidationError, match="maximum"):
        validate_loaded_pdf(loaded, max_pages=1)


def test_validate_loaded_pdf_rejects_oversized_page_text(tmp_path):
    path = tmp_path / "report.pdf"

    loaded = LoadedPDF(
        file_path=path,
        filename="report.pdf",
        page_count=1,
        pages=(PDFPage(page_number=1, text="A" * 20),),
    )

    with pytest.raises(IngestionValidationError, match="exceeds"):
        validate_loaded_pdf(loaded, max_page_text_chars=10)