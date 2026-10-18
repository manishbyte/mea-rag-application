
"""Tests for page-aware text chunking."""

import pytest

from ingestion.chunker import (
    chunk_page,
    chunk_pages,
    normalize_text,
)
from ingestion.pdf_loader import PDFPage


def test_normalize_text_collapses_extra_whitespace():
    assert normalize_text("  Hello   world \n\n\nNext line ") == (
        "Hello world\n\nNext line"
    )


def test_normalize_text_removes_null_characters():
    assert normalize_text("Hello\x00world") == "Hello world"


def test_normalize_text_handles_empty_text():
    assert normalize_text("  \n\t ") == ""


def test_chunk_page_returns_empty_for_blank_page():
    page = PDFPage(page_number=1, text=" \n ")

    assert chunk_page(page) == []


def test_chunk_page_preserves_page_number():
    page = PDFPage(
        page_number=3,
        text="This is a short page of text.",
    )

    chunks = chunk_page(page)

    assert len(chunks) == 1
    assert chunks[0].page_number == 3
    assert chunks[0].chunk_index == 0
    assert chunks[0].content == "This is a short page of text."


def test_chunk_page_splits_long_text():
    page = PDFPage(
        page_number=1,
        text=" ".join(["word"] * 100),
    )

    chunks = chunk_page(page, chunk_size=50, overlap=10)

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 50 for chunk in chunks)
    assert all(chunk.page_number == 1 for chunk in chunks)


def test_chunk_page_chunks_overlap():
    page = PDFPage(
        page_number=1,
        text=" ".join(f"word{i}" for i in range(30)),
    )

    chunks = chunk_page(page, chunk_size=50, overlap=15)

    assert len(chunks) > 1
    assert any(
        set(first.content.split()) & set(second.content.split())
        for first, second in zip(chunks, chunks[1:])
    )


def test_chunk_pages_preserves_page_boundaries_and_global_indexes():
    pages = [
        PDFPage(page_number=1, text="First page content."),
        PDFPage(page_number=2, text="Second page content."),
        PDFPage(page_number=3, text="   "),
        PDFPage(page_number=4, text="Fourth page content."),
    ]

    chunks = chunk_pages(pages)

    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert [chunk.page_number for chunk in chunks] == [1, 2, 4]


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 0),
        (10, -1),
        (10, 10),
        (10, 11),
    ],
)
def test_chunk_page_rejects_invalid_configuration(
    chunk_size,
    overlap,
):
    page = PDFPage(page_number=1, text="Some text")

    with pytest.raises(ValueError):
        chunk_page(
            page,
            chunk_size=chunk_size,
            overlap=overlap,
        )


def test_chunk_page_rejects_negative_first_chunk_index():
    page = PDFPage(page_number=1, text="Some text")

    with pytest.raises(ValueError):
        chunk_page(page, first_chunk_index=-1)


def test_chunk_page_tracks_character_offsets():
    page = PDFPage(
        page_number=2,
        text="  First paragraph. Second paragraph.  ",
    )

    chunks = chunk_page(page)

    assert len(chunks) == 1
    chunk = chunks[0]
    normalized = normalize_text(page.text)

    assert normalized[chunk.start_char:chunk.end_char] == chunk.content
