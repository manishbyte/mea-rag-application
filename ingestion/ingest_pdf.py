
"""Orchestrate PDF validation, extraction, chunking, embedding,
Neon persistence, and local JSONL export.
"""

import argparse
import asyncio
import hashlib
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from database.models.document import DocumentChunk, SourceDocument
from database.session import dispose_engine, get_session_factory
from ingestion.chunker import chunk_pdf
from ingestion.embedder import embed_chunks_async
from ingestion.exporter import DEFAULT_OUTPUT_PATH, export_chunks_jsonl
from ingestion.pdf_loader import LoadedPDF, load_pdf
from ingestion.validators import validate_loaded_pdf, validate_pdf_file


logger = logging.getLogger(__name__)

DEFAULT_PDF_PATH = Path("data/source/mea_annual_report.pdf")


class IngestionError(RuntimeError):
    """Raised when PDF ingestion cannot be completed."""


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Summary of a completed ingestion operation."""

    document_id: str
    filename: str
    title: str
    sha256: str
    page_count: int
    chunk_count: int
    output_path: str
    already_ingested: bool


def _calculate_sha256(file_path: Path) -> str:
    """Calculate a file's SHA-256 digest in memory-efficient blocks."""
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _default_title(filename: str) -> str:
    """Derive a readable title from a PDF filename."""
    return Path(filename).stem.replace("_", " ").replace("-", " ").strip()


def _build_export_records(
    document: SourceDocument,
    chunks: list[DocumentChunk],
) -> list[dict]:
    """Build local JSONL records from persisted chunks."""
    records = []

    for chunk in chunks:
        records.append(
            {
                "document_id": str(document.id),
                "filename": document.filename,
                "title": document.title,
                "sha256": document.sha256,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "content": chunk.content,
                "metadata": chunk.metadata_json or {},
            }
        )

    return records


async def _export_existing_document(
    session: AsyncSession,
    document: SourceDocument,
    output_path: Path,
) -> IngestionResult:
    """Export an existing document without re-embedding or duplicating it."""
    result = await session.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document.id)
        .order_by(DocumentChunk.chunk_index)
    )
    chunks = list(result.scalars().all())

    if not chunks:
        raise IngestionError(
            f"Document {document.filename!r} already exists but has no "
            "chunks. Inspect the database before retrying ingestion."
        )

    logger.info(
        "Existing document found: %d chunks. Reusing persisted data.",
        len(chunks),
    )
    logger.info("Exporting existing chunks to %s", output_path)

    await asyncio.to_thread(
        export_chunks_jsonl,
        _build_export_records(document, chunks),
        output_path,
    )

    return IngestionResult(
        document_id=str(document.id),
        filename=document.filename,
        title=document.title,
        sha256=document.sha256,
        page_count=len({chunk.page_number for chunk in chunks}),
        chunk_count=len(chunks),
        output_path=str(output_path.resolve()),
        already_ingested=True,
    )


async def ingest_pdf(
    file_path: str | Path = DEFAULT_PDF_PATH,
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    title: str | None = None,
    chunk_size: int = 1200,
    overlap: int = 200,
    embedding_batch_size: int = 32,
) -> IngestionResult:
    """Ingest a PDF into Neon and export its persisted chunks locally.

    Existing documents are detected by SHA-256 before extraction or
    embedding. New document records and chunks are inserted atomically.
    The local export is generated from committed database records.
    """
    validated_path = validate_pdf_file(file_path)
    destination = Path(output_path).expanduser().resolve()

    document_title = (
        title.strip() if title else _default_title(validated_path.name)
    )

    if not document_title:
        raise ValueError("Document title must not be empty.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )
    if embedding_batch_size < 1:
        raise ValueError("embedding_batch_size must be at least 1.")

    logger.info("Calculating PDF SHA-256...")
    sha256 = await asyncio.to_thread(_calculate_sha256, validated_path)

    session_factory = get_session_factory()

    # Check for an existing document before expensive processing.
    logger.info("Checking Neon for an existing document...")

    async with session_factory() as session:
        existing_result = await session.execute(
            select(SourceDocument).where(SourceDocument.sha256 == sha256)
        )
        existing_document = existing_result.scalar_one_or_none()

        if existing_document is not None:
            logger.info(
                "Document already exists. Skipping extraction and embeddings."
            )
            return await _export_existing_document(
                session,
                existing_document,
                destination,
            )

    logger.info("Document is new. Starting PDF extraction...")

    loaded_pdf: LoadedPDF = await asyncio.to_thread(
        load_pdf,
        validated_path,
    )
    validate_loaded_pdf(loaded_pdf)

    logger.info(
        "PDF extraction complete: %d pages.",
        loaded_pdf.page_count,
    )

    logger.info("Chunking PDF text...")
    chunks = chunk_pdf(
        loaded_pdf,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    if not chunks:
        raise IngestionError("No text chunks were produced from the PDF.")

    logger.info("Chunking complete: %d chunks.", len(chunks))
    logger.info(
        "Generating embeddings with batch size %d...",
        embedding_batch_size,
    )

    embedded_chunks = await embed_chunks_async(
        chunks,
        batch_size=embedding_batch_size,
    )

    if len(embedded_chunks) != len(chunks):
        raise IngestionError(
            "Embedding output count does not match the chunk count."
        )

    logger.info(
        "Embedding generation complete: %d embeddings.",
        len(embedded_chunks),
    )
    logger.info("Writing document and chunks to Neon...")

    document_id = None

    try:
        async with session_factory() as session:
            async with session.begin():
                # Recheck inside the write transaction to reduce duplicate
                # inserts if another ingestion starts concurrently.
                existing_result = await session.execute(
                    select(SourceDocument).where(
                        SourceDocument.sha256 == sha256
                    )
                )
                existing_document = existing_result.scalar_one_or_none()

                if existing_document is not None:
                    document_id = existing_document.id
                    logger.info(
                        "Another process inserted this document. "
                        "Reusing its persisted records."
                    )
                else:
                    document = SourceDocument(
                        filename=validated_path.name,
                        title=document_title,
                        sha256=sha256,
                    )
                    session.add(document)
                    await session.flush()

                    document_id = document.id

                    database_chunks = [
                        DocumentChunk(
                            document_id=document_id,
                            chunk_index=item["chunk_index"],
                            page_number=item["page_number"],
                            content=item["content"],
                            embedding=item["embedding"],
                            metadata_json={
                                "start_char": item["start_char"],
                                "end_char": item["end_char"],
                                "source": validated_path.name,
                            },
                        )
                        for item in embedded_chunks
                    ]

                    session.add_all(database_chunks)

    except IntegrityError:
        # A concurrent process could win the unique SHA-256 constraint.
        # Its transaction is unaffected; retrieve its committed record.
        logger.warning(
            "Database uniqueness conflict. Checking whether another "
            "process ingested the same PDF."
        )

        async with session_factory() as session:
            existing_result = await session.execute(
                select(SourceDocument).where(
                    SourceDocument.sha256 == sha256
                )
            )
            existing_document = existing_result.scalar_one_or_none()

            if existing_document is None:
                raise IngestionError(
                    "Database insertion failed because of a uniqueness "
                    "conflict, but no matching document was found."
                )

            return await _export_existing_document(
                session,
                existing_document,
                destination,
            )

    logger.info("Database transaction committed successfully.")

    # Reload persisted records so the export mirrors Neon.
    async with session_factory() as session:
        document_result = await session.execute(
            select(SourceDocument).where(
                SourceDocument.id == document_id
            )
        )
        persisted_document = document_result.scalar_one()

        chunks_result = await session.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        )
        persisted_chunks = list(chunks_result.scalars().all())

        if len(persisted_chunks) != len(embedded_chunks):
            raise IngestionError(
                "Persisted chunk count differs from the generated count. "
                "Database records were committed, but export was skipped."
            )

        records = _build_export_records(
            persisted_document,
            persisted_chunks,
        )

    logger.info("Exporting chunks to %s", destination)

    await asyncio.to_thread(
        export_chunks_jsonl,
        records,
        destination,
    )

    result = IngestionResult(
        document_id=str(document_id),
        filename=validated_path.name,
        title=persisted_document.title,
        sha256=sha256,
        page_count=loaded_pdf.page_count,
        chunk_count=len(persisted_chunks),
        output_path=str(destination),
        already_ingested=False,
    )

    logger.info(
        "Ingestion completed: pages=%d chunks=%d output=%s",
        result.page_count,
        result.chunk_count,
        result.output_path,
    )

    return result


async def _run_cli(args: argparse.Namespace) -> None:
    """Run the ingestion pipeline from the command line."""
    try:
        result = await ingest_pdf(
            args.file,
            output_path=args.output,
            title=args.title,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            embedding_batch_size=args.batch_size,
        )

        print("\nIngestion result:")
        for key, value in asdict(result).items():
            print(f"{key}: {value}")

    finally:
        await dispose_engine()


def main() -> None:
    """Parse command-line arguments and run ingestion."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate, chunk, embed, and ingest a PDF into Neon "
            "while exporting chunk data to local JSONL."
        )
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_PDF_PATH,
        help="Path to the input PDF.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the local JSONL export.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional document title.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Chunk size in characters.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Chunk overlap in characters.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size.",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        asyncio.run(_run_cli(args))
    except Exception:
        logging.exception("PDF ingestion failed.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
