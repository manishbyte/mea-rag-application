
"""Semantic vector retrieval using PostgreSQL and pgvector."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models.document import DocumentChunk, SourceDocument
from database.session import get_session_factory

EMBEDDING_DIMENSIONS = 384
DEFAULT_TOP_K = 10
MAX_TOP_K = 100


class VectorSearchError(RuntimeError):
    """Raised when semantic vector retrieval fails."""


def _validate_inputs(
    query_embedding: Sequence[float],
    top_k: int,
) -> tuple[list[float], int]:
    """Validate a query vector and the maximum result count."""
    if isinstance(query_embedding, (str, bytes)):
        raise ValueError("Query embedding must be a numeric sequence.")

    try:
        vector = [float(value) for value in query_embedding]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Query embedding must contain only numeric values."
        ) from exc

    if len(vector) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Expected an embedding with {EMBEDDING_DIMENSIONS} dimensions; "
            f"received {len(vector)}."
        )

    if not all(math.isfinite(value) for value in vector):
        raise ValueError("Query embedding contains non-finite values.")

    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ValueError("top_k must be an integer.")

    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}.")

    return vector, top_k


def _build_vector_statement(
    query_embedding: Sequence[float],
    top_k: int,
):
    """Build a pgvector cosine-distance retrieval statement."""
    distance = DocumentChunk.embedding.cosine_distance(
        list(query_embedding)
    ).label("cosine_distance")

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
            distance,
        )
        .join(
            SourceDocument,
            SourceDocument.id == DocumentChunk.document_id,
        )
        .where(DocumentChunk.embedding.is_not(None))
        .order_by(distance.asc(), DocumentChunk.id.asc())
        .limit(top_k)
    )


async def vector_search(
    query_embedding: Sequence[float],
    top_k: int = DEFAULT_TOP_K,
    *,
    session: AsyncSession | None = None,
) -> list[dict[str, Any]]:
    """Retrieve the most semantically similar document chunks.

    Args:
        query_embedding: Query vector produced by retrieval/embeddings.py.
        top_k: Maximum number of chunks to return, between 1 and 100.
        session: Optional existing AsyncSession. If omitted, this function
            creates and closes its own session.

    Returns:
        A list ordered by ascending cosine distance. Each result contains
        chunk/document metadata, cosine distance, cosine similarity, and
        the retrieval method.
    """
    vector, top_k = _validate_inputs(query_embedding, top_k)
    statement = _build_vector_statement(vector, top_k)

    async def execute_search(db: AsyncSession) -> list[dict[str, Any]]:
        result = await db.execute(statement)
        rows = result.mappings().all()

        chunks: list[dict[str, Any]] = []

        for row in rows:
            distance = float(row["cosine_distance"])

            chunks.append(
                {
                    "chunk_id": str(row["chunk_id"]),
                    "document_id": str(row["document_id"]),
                    "chunk_index": row["chunk_index"],
                    "page_number": row["page_number"],
                    "content": row["content"],
                    "metadata": row["metadata"] or {},
                    "document_title": row["document_title"],
                    "filename": row["filename"],
                    "cosine_distance": distance,
                    "similarity_score": 1.0 - distance,
                    "retrieval_method": "vector",
                }
            )

        return chunks

    try:
        if session is not None:
            return await execute_search(session)

        async with get_session_factory()() as db:
            return await execute_search(db)

    except Exception as exc:
        raise VectorSearchError(
            "Vector retrieval failed. Check database connectivity, "
            "pgvector availability, and stored embedding dimensions."
        ) from exc
