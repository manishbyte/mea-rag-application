
"""PDF loading and page-by-page text extraction."""

from dataclasses import dataclass
from pathlib import Path

import fitz


class PDFLoaderError(Exception):
    """Base exception for PDF loading errors."""


class InvalidPDFError(PDFLoaderError):
    """Raised when the input is not a valid, readable PDF."""


class EmptyPDFError(PDFLoaderError):
    """Raised when a PDF contains no extractable text."""


@dataclass(frozen=True, slots=True)
class PDFPage:
    """Extracted text and metadata for one PDF page."""

    page_number: int
    text: str


@dataclass(frozen=True, slots=True)
class LoadedPDF:
    """Extracted content and metadata for a PDF document."""

    file_path: Path
    filename: str
    page_count: int
    pages: tuple[PDFPage, ...]

    @property
    def full_text(self) -> str:
        """Return all page text with page boundaries preserved."""
        return "\n\n".join(page.text for page in self.pages)


def load_pdf(
    file_path: str | Path,
    *,
    min_text_chars: int = 1,
) -> LoadedPDF:
    """Extract text from a PDF while preserving page numbers.

    Args:
        file_path: Path to the PDF file.
        min_text_chars: Minimum total number of non-whitespace
            characters required across all pages.

    Returns:
        LoadedPDF containing the extracted text for each page.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If min_text_chars is less than one.
        InvalidPDFError: If the file is not a readable PDF.
        EmptyPDFError: If insufficient extractable text is present.
    """
    path = Path(file_path).expanduser()

    if min_text_chars < 1:
        raise ValueError("min_text_chars must be at least 1.")

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if not path.is_file():
        raise InvalidPDFError(f"Expected a PDF file, got: {path}")

    if path.suffix.lower() != ".pdf":
        raise InvalidPDFError("Input file must have a .pdf extension.")

    if path.stat().st_size == 0:
        raise InvalidPDFError("The PDF file is empty.")

    extracted_pages: list[PDFPage] = []

    try:
        with fitz.open(path) as document:
            if not document.is_pdf:
                raise InvalidPDFError("The supplied file is not a PDF.")

            if document.needs_pass:
                raise InvalidPDFError(
                    "The PDF is password-protected and cannot be opened."
                )

            if document.page_count == 0:
                raise EmptyPDFError("The PDF contains no pages.")

            page_count = document.page_count

            for index, page in enumerate(document):
                text = page.get_text("text").strip()

                # Page numbers are one-based for citations.
                extracted_pages.append(
                    PDFPage(
                        page_number=index + 1,
                        text=text,
                    )
                )

    except PDFLoaderError:
        raise
    except (fitz.FileDataError, RuntimeError, OSError) as exc:
        raise InvalidPDFError(
            f"Could not read PDF file: {path.name}"
        ) from exc

    total_text_chars = sum(
        len(page.text.strip())
        for page in extracted_pages
    )

    if total_text_chars < min_text_chars:
        raise EmptyPDFError(
            "The PDF contains insufficient extractable text. "
            "It may be scanned and require OCR."
        )

    return LoadedPDF(
        file_path=path.resolve(),
        filename=path.name,
        page_count=page_count,
        pages=tuple(extracted_pages),
    )
