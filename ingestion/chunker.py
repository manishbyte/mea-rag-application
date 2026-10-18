
"""Text normalization and page-aware chunking for RAG ingestion."""

import re
from dataclasses import dataclass

from ingestion.pdf_loader import LoadedPDF, PDFPage


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A chunk of extracted text with source location metadata."""

    chunk_index: int
    page_number: int
    content: str
    start_char: int
    end_char: int


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n+", "\n\n", text)
    return text.strip()


def _split_into_segments(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
) -> list[tuple[str, int, int]]:
    """Split text into overlapping segments without cutting words."""
    segments: list[tuple[str, int, int]] = []
    text_length = len(text)
    start = 0

    while start < text_length:
        tentative_end = min(start + chunk_size, text_length)
        end = tentative_end

        if tentative_end < text_length:
            # Prefer a word boundary when possible.
            boundary = text.rfind(" ", start, tentative_end)

            if boundary > start:
                end = boundary

        segment = text[start:end].strip()

        if segment:
            # Track offsets in the normalized page text.
            actual_start = start
            actual_end = end

            while actual_start < actual_end and text[actual_start].isspace():
                actual_start += 1

            while actual_end > actual_start and text[actual_end - 1].isspace():
                actual_end -= 1

            segments.append(
                (text[actual_start:actual_end], actual_start, actual_end)
            )

        if end >= text_length:
            break

        next_start = max(end - overlap, start + 1)

        # Avoid beginning the next chunk in the middle of a word.
        while next_start < text_length and text[next_start].isspace():
            next_start += 1

        start = next_start

    return segments


def chunk_page(
    page: PDFPage,
    *,
    chunk_size: int = 1200,
    overlap: int = 200,
    first_chunk_index: int = 0,
) -> list[TextChunk]:
    """Chunk one PDF page while preserving its page number."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    if first_chunk_index < 0:
        raise ValueError("first_chunk_index cannot be negative.")

    normalized = normalize_text(page.text)

    if not normalized:
        return []

    segments = _split_into_segments(
        normalized,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    return [
        TextChunk(
            chunk_index=first_chunk_index + index,
            page_number=page.page_number,
            content=content,
            start_char=start_char,
            end_char=end_char,
        )
        for index, (content, start_char, end_char) in enumerate(segments)
    ]


def chunk_pages(
    pages: list[PDFPage] | tuple[PDFPage, ...],
    *,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[TextChunk]:
    """Chunk multiple pages without mixing content across page boundaries."""
    chunks: list[TextChunk] = []
    next_index = 0

    for page in pages:
        page_chunks = chunk_page(
            page,
            chunk_size=chunk_size,
            overlap=overlap,
            first_chunk_index=next_index,
        )

        chunks.extend(page_chunks)
        next_index += len(page_chunks)

    return chunks


def chunk_pdf(
    loaded_pdf: LoadedPDF,
    *,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[TextChunk]:
    """Create page-aware chunks from a loaded PDF."""
    return chunk_pages(
        loaded_pdf.pages,
        chunk_size=chunk_size,
        overlap=overlap,
    )
