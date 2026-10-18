
"""Tests for PDF ingestion orchestration helpers."""

import hashlib
from types import SimpleNamespace
from uuid import uuid4
import anyio
from ingestion import ingest_pdf as ingestion_package

import pytest

from ingestion import ingest_pdf as ingestion_package
import importlib

ingest_module = importlib.import_module("ingestion.ingest_pdf")


def test_calculate_sha256(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    content = b"%PDF-1.4\nsample PDF content"
    pdf_path.write_bytes(content)

    actual = ingest_module._calculate_sha256(pdf_path)

    assert actual == hashlib.sha256(content).hexdigest()


def test_default_title_replaces_underscores():
    assert (
        ingest_module._default_title("mea_annual_report.pdf")
        == "mea annual report"
    )


def test_default_title_replaces_hyphens():
    assert (
        ingest_module._default_title("annual-report-2025.pdf")
        == "annual report 2025"
    )


def test_build_export_records_preserves_chunk_metadata():
    document = SimpleNamespace(
        id=uuid4(),
        filename="report.pdf",
        title="Annual Report",
        sha256="a" * 64,
    )
    chunks = [
        SimpleNamespace(
            chunk_index=0,
            page_number=1,
            content="First chunk",
            metadata_json={"start_char": 0, "end_char": 11},
        ),
        SimpleNamespace(
            chunk_index=1,
            page_number=2,
            content="Second chunk",
            metadata_json={"start_char": 5, "end_char": 17},
        ),
    ]

    records = ingest_module._build_export_records(document, chunks)

    assert len(records) == 2
    assert records[0]["document_id"] == str(document.id)
    assert records[0]["filename"] == "report.pdf"
    assert records[0]["chunk_index"] == 0
    assert records[0]["page_number"] == 1
    assert records[0]["content"] == "First chunk"
    assert records[0]["metadata"] == {
        "start_char": 0,
        "end_char": 11,
    }
    assert records[1]["chunk_index"] == 1
    assert records[1]["page_number"] == 2


@pytest.mark.asyncio
def test_invalid_pdf_fails_before_database_access(
    tmp_path,
    monkeypatch,
):
    missing_pdf = tmp_path / "missing.pdf"

    def database_must_not_be_accessed():
        pytest.fail("Database session factory should not be accessed.")

    monkeypatch.setattr(
        ingest_module,
        "get_session_factory",
        database_must_not_be_accessed,
    )

    async def run_test():
        with pytest.raises(FileNotFoundError):
            await ingest_module.ingest_pdf(missing_pdf)

    anyio.run(run_test)


@pytest.mark.asyncio
def test_invalid_chunk_size_fails_before_database_access(
    tmp_path,
    monkeypatch,
):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nsample")

    def database_must_not_be_accessed():
        pytest.fail("Database session factory should not be accessed.")

    monkeypatch.setattr(
        ingest_module,
        "get_session_factory",
        database_must_not_be_accessed,
    )

    async def run_test():
        with pytest.raises(ValueError, match="chunk_size"):
            await ingest_module.ingest_pdf(
                pdf_path,
                chunk_size=0,
            )

    anyio.run(run_test)


@pytest.mark.asyncio
def test_overlap_must_be_smaller_than_chunk_size(
    tmp_path,
    monkeypatch,
):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nsample")

    def database_must_not_be_accessed():
        pytest.fail("Database session factory should not be accessed.")

    monkeypatch.setattr(
        ingest_module,
        "get_session_factory",
        database_must_not_be_accessed,
    )

    async def run_test():
        with pytest.raises(ValueError, match="overlap"):
            await ingest_module.ingest_pdf(
                pdf_path,
                chunk_size=100,
                overlap=100,
            )

    anyio.run(run_test)


@pytest.mark.asyncio
def test_embedding_batch_size_must_be_positive(
    tmp_path,
    monkeypatch,
):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nsample")

    def database_must_not_be_accessed():
        pytest.fail("Database session factory should not be accessed.")

    monkeypatch.setattr(
        ingest_module,
        "get_session_factory",
        database_must_not_be_accessed,
    )

    async def run_test():
        with pytest.raises(ValueError, match="embedding_batch_size"):
            await ingest_module.ingest_pdf(
                pdf_path,
                embedding_batch_size=0,
            )

    anyio.run(run_test)