
"""PostgreSQL full-text keyword retrieval for MEA report chunks."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models.document import DocumentChunk, SourceDocument
from database.session import get_session_factory

TEXT_SEARCH_CONFIG = "english"
DEFAULT_TOP_K = 10
MAX_TOP_K = 100


class KeywordSearchError(RuntimeError):
    """Raised when keyword retrieval fails."""


def _validate_query(query: str, top_k: int) -> tuple[str, int]:
    """Validate user query and retrieval limit."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query must be a non-empty string.")

    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ValueError("top_k must be an integer.")

    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}.")

    return query.strip(), top_k


def _build_keyword_statement(query: str, top_k: int):
    """Build a parameterized PostgreSQL full-text search statement."""
    search_query = func.plainto_tsquery(TEXT_SEARCH_CONFIG, query)
    searchable_content = func.to_tsvector(
        TEXT_SEARCH_CONFIG,
        DocumentChunk.content,
    )

    rank = func.ts_rank_cd(searchable_content, search_query).label(
        "keyword_score"
    )

    return (
        select(
            DocumentChunk.id.label("chunk_id"),
            DocumentChunk.document_id.label("document_id"),
            DocumentChunk.chunk_index.label("chunk_index"),
            DocumentChunk.page_number.label("page_number"),
            DocumentChunk.content.label("content"),
            DocumentChunk.metadata_json.label("metadata"),
            SourceDocument.title.label("document_title"),
            SourceDocument.filename.label("filename"),
            rank,
        )
        .join(
            SourceDocument,
            SourceDocument.id == DocumentChunk.document_id,
        )
        .where(searchable_content.op("@@")(search_query))
        .order_by(rank.desc(), DocumentChunk.id.asc())
        .limit(top_k)
    )


async def keyword_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    *,
    session: AsyncSession | None = None,
) -> list[dict[str, Any]]:
    """Retrieve chunks ranked by PostgreSQL full-text relevance.

    If a session is provided, the caller owns its lifecycle. Otherwise,
    this function opens and closes its own async database session.

    Results are ordered by descending PostgreSQL relevance score.
    """
    query, top_k = _validate_query(query, top_k)
    statement = _build_keyword_statement(query, top_k)

    async def execute_search(db: AsyncSession) -> list[dict[str, Any]]:
        result = await db.execute(statement)
        rows = result.mappings().all()

        return [
            {
                "chunk_id": str(row["chunk_id"]),
                "document_id": str(row["document_id"]),
                "chunk_index": row["chunk_index"],
                "page_number": row["page_number"],
                "content": row["content"],
                "metadata": row["metadata"] or {},
                "document_title": row["document_title"],
                "filename": row["filename"],
                "keyword_score": float(row["keyword_score"]),
                "retrieval_method": "keyword",
            }
            for row in rows
        ]

    try:
        if session is not None:
            return await execute_search(session)

        async with get_session_factory()() as db:
            return await execute_search(db)

    except Exception as exc:
        if isinstance(exc, (ValueError,)):
            raise
        raise KeywordSearchError(
            "Keyword retrieval failed. Check database connectivity "
            "and confirm the full-text search migration was applied."
        ) from exc


async def keyword_search_by_document(
    query: str,
    document_id: UUID,
    top_k: int = DEFAULT_TOP_K,
    *,
    session: AsyncSession | None = None,
) -> list[dict[str, Any]]:
    """Search within one document, useful for controlled retrieval tests."""
    query, top_k = _validate_query(query, top_k)
    search_query = func.plainto_tsquery(TEXT_SEARCH_CONFIG, query)
    searchable_content = func.to_tsvector(
        TEXT_SEARCH_CONFIG,
        DocumentChunk.content,
    )
    rank = func.ts_rank_cd(searchable_content, search_query).label(
        "keyword_score"
    )

    statement = (
        select(
            DocumentChunk.id.label("chunk_id"),
            DocumentChunk.document_id.label("document_id"),
            DocumentChunk.chunk_index.label("chunk_index"),
            DocumentChunk.page_number.label("page_number"),
            DocumentChunk.content.label("content"),
            DocumentChunk.metadata_json.label("metadata"),
            SourceDocument.title.label("document_title"),
            SourceDocument.filename.label("filename"),
            rank,
        )
        .join(
            SourceDocument,
            SourceDocument.id == DocumentChunk.document_id,
        )
        .where(
            DocumentChunk.document_id == document_id,
            searchable_content.op("@@")(search_query),
        )
        .order_by(rank.desc(), DocumentChunk.id.asc())
        .limit(top_k)
    )

    async def execute_search(db: AsyncSession) -> list[dict[str, Any]]:
        result = await db.execute(statement)
        return [
            {
                "chunk_id": str(row["chunk_id"]),
                "document_id": str(row["document_id"]),
                "chunk_index": row["chunk_index"],
                "page_number": row["page_number"],
                "content": row["content"],
                "metadata": row["metadata"] or {},
                "document_title": row["document_title"],
                "filename": row["filename"],
                "keyword_score": float(row["keyword_score"]),
                "retrieval_method": "keyword",
            }
            for row in result.mappings().all()
        ]

    try:
        if session is not None:
            return await execute_search(session)

        async with get_session_factory()() as db:
            return await execute_search(db)

    except Exception as exc:
        raise KeywordSearchError(
            "Document-scoped keyword retrieval failed."
        ) from exc
