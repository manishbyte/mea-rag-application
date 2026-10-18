
"""Input and extracted-content validation for PDF ingestion."""

from pathlib import Path

from ingestion.pdf_loader import LoadedPDF


DEFAULT_MAX_PDF_SIZE_BYTES = 100 * 1024 * 1024  # 100 MiB


class IngestionValidationError(ValueError):
    """Raised when an ingestion input fails validation."""


def validate_pdf_file(
    file_path: str | Path,
    *,
    max_size_bytes: int = DEFAULT_MAX_PDF_SIZE_BYTES,
) -> Path:
    """Validate PDF path, extension, size, and basic file properties.

    This function does not parse the PDF. Use load_pdf() afterward
    to verify that the document can be opened and its text extracted.
    """
    path = Path(file_path).expanduser()

    if max_size_bytes < 1:
        raise ValueError("max_size_bytes must be at least 1.")

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if not path.is_file():
        raise IngestionValidationError(
            f"Expected a file, received: {path}"
        )

    if path.suffix.lower() != ".pdf":
        raise IngestionValidationError(
            "Only files with a .pdf extension are supported."
        )

    size_bytes = path.stat().st_size

    if size_bytes == 0:
        raise IngestionValidationError("The PDF file is empty.")

    if size_bytes > max_size_bytes:
        raise IngestionValidationError(
            f"PDF size ({size_bytes} bytes) exceeds the "
            f"maximum allowed size ({max_size_bytes} bytes)."
        )

    # A PDF normally begins with the %PDF- signature. Check a small
    # prefix rather than reading the entire file into memory.
    with path.open("rb") as file:
        header = file.read(1024)

    if b"%PDF-" not in header:
        raise IngestionValidationError(
            "The file does not contain a PDF signature."
        )

    return path.resolve()


def validate_loaded_pdf(
    loaded_pdf: LoadedPDF,
    *,
    min_total_text_chars: int = 1,
    max_pages: int = 10_000,
    max_page_text_chars: int = 2_000_000,
) -> None:
    """Validate page count and extracted text before chunking."""
    if min_total_text_chars < 1:
        raise ValueError("min_total_text_chars must be at least 1.")

    if max_pages < 1:
        raise ValueError("max_pages must be at least 1.")

    if max_page_text_chars < 1:
        raise ValueError("max_page_text_chars must be at least 1.")

    if loaded_pdf.page_count < 1:
        raise IngestionValidationError(
            "The PDF must contain at least one page."
        )

    if loaded_pdf.page_count > max_pages:
        raise IngestionValidationError(
            f"PDF has {loaded_pdf.page_count} pages; "
            f"the maximum allowed is {max_pages}."
        )

    if len(loaded_pdf.pages) != loaded_pdf.page_count:
        raise IngestionValidationError(
            "Extracted page count does not match the PDF page count."
        )

    total_text_chars = 0

    for expected_page_number, page in enumerate(
        loaded_pdf.pages,
        start=1,
    ):
        if page.page_number != expected_page_number:
            raise IngestionValidationError(
                "Page numbers must be sequential and one-based."
            )

        if len(page.text) > max_page_text_chars:
            raise IngestionValidationError(
                f"Extracted text on page {page.page_number} exceeds "
                f"the maximum of {max_page_text_chars} characters."
            )

        total_text_chars += len(page.text.strip())

    if total_text_chars < min_total_text_chars:
        raise IngestionValidationError(
            "The PDF does not contain enough extractable text. "
            "It may require OCR if it contains scanned pages."
        )
