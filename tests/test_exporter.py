
"""Tests for JSONL chunk export."""

import json
from uuid import uuid4

import pytest

from ingestion.exporter import (
    ExportError,
    export_chunks_jsonl,
)


def make_record(**overrides):
    record = {
        "chunk_index": 0,
        "page_number": 1,
        "content": "MEA Annual Report 2025",
        "filename": "mea_annual_report.pdf",
        "document_id": str(uuid4()),
    }
    record.update(overrides)
    return record


def test_export_chunks_writes_jsonl(tmp_path):
    destination = tmp_path / "processed" / "chunks.jsonl"
    records = [
        make_record(chunk_index=0, page_number=1),
        make_record(
            chunk_index=1,
            page_number=2,
            content="Second page content",
        ),
    ]

    result = export_chunks_jsonl(records, destination)

    assert result == destination.resolve()
    lines = destination.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    second = json.loads(lines[1])

    assert first["chunk_index"] == 0
    assert first["page_number"] == 1
    assert second["chunk_index"] == 1
    assert second["content"] == "Second page content"


def test_export_chunks_preserves_unicode(tmp_path):
    destination = tmp_path / "chunks.jsonl"

    export_chunks_jsonl(
        [make_record(content="भारत — Ministry of External Affairs")],
        destination,
    )

    raw = destination.read_text(encoding="utf-8")

    assert "भारत" in raw
    assert "Ministry of External Affairs" in raw


def test_export_chunks_handles_empty_records(tmp_path):
    destination = tmp_path / "empty.jsonl"

    export_chunks_jsonl([], destination)

    assert destination.read_text(encoding="utf-8") == ""


@pytest.mark.parametrize(
    "record",
    [
        {"page_number": 1, "content": "Missing chunk index"},
        {"chunk_index": -1, "page_number": 1, "content": "Invalid index"},
        {"chunk_index": 0, "page_number": 0, "content": "Invalid page"},
        {"chunk_index": 0, "page_number": 1, "content": ""},
        {"chunk_index": 0, "page_number": 1, "content": 123},
    ],
)
def test_export_chunks_rejects_invalid_records(tmp_path, record):
    destination = tmp_path / "chunks.jsonl"

    with pytest.raises(ValueError):
        export_chunks_jsonl([record], destination)

    assert not destination.exists()


def test_export_chunks_preserves_existing_file_on_validation_failure(
    tmp_path,
):
    destination = tmp_path / "chunks.jsonl"
    destination.write_text(
        '{"chunk_index":0,"page_number":1,"content":"old"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        export_chunks_jsonl(
            [make_record(content=""),],
            destination,
        )

    assert '"content":"old"' in destination.read_text(encoding="utf-8")


def test_export_chunks_serializes_uuid_values(tmp_path):
    destination = tmp_path / "chunks.jsonl"
    document_id = uuid4()

    export_chunks_jsonl(
        [make_record(document_id=document_id)],
        destination,
    )

    record = json.loads(destination.read_text(encoding="utf-8"))

    assert record["document_id"] == str(document_id)


def test_export_chunks_rejects_non_serializable_values(tmp_path):
    destination = tmp_path / "chunks.jsonl"

    with pytest.raises(ValueError, match="cannot be serialized"):
        export_chunks_jsonl(
            [make_record(unexpected=object())],
            destination,
        )
