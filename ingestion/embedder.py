
"""Batch embedding generation for MEA RAG ingestion."""

import asyncio
from functools import lru_cache
from math import isfinite
from typing import Sequence

from fastembed import TextEmbedding

from ingestion.chunker import TextChunk


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384
DEFAULT_BATCH_SIZE = 32


class EmbeddingError(RuntimeError):
    """Raised when embedding generation or validation fails."""


@lru_cache(maxsize=1)
def get_embedding_model() -> TextEmbedding:
    """Load and cache the embedding model for this process."""
    try:
        return TextEmbedding(model_name=EMBEDDING_MODEL)
    except Exception as exc:
        raise EmbeddingError(
            f"Could not load embedding model: {EMBEDDING_MODEL}"
        ) from exc


def validate_embedding(
    embedding: Sequence[float],
    *,
    dimensions: int = EMBEDDING_DIMENSIONS,
) -> list[float]:
    """Validate and normalize an embedding into a list of floats."""
    if dimensions < 1:
        raise ValueError("dimensions must be at least 1.")

    if len(embedding) != dimensions:
        raise EmbeddingError(
            f"Expected {dimensions} embedding dimensions; "
            f"received {len(embedding)}."
        )

    result: list[float] = []

    for value in embedding:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise EmbeddingError(
                "Embedding contains a non-numeric value."
            ) from exc

        if not isfinite(numeric_value):
            raise EmbeddingError(
                "Embedding contains NaN or infinite values."
            )

        result.append(numeric_value)

    return result


def _embed_texts_sync(
    texts: list[str],
    *,
    batch_size: int,
) -> list[list[float]]:
    """Generate embeddings synchronously in bounded batches."""
    if not texts:
        return []

    model = get_embedding_model()
    embeddings: list[list[float]] = []

    try:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]

            # passage_embed is intended for document passages.
            vectors = model.passage_embed(
                batch,
                batch_size=batch_size,
            )

            for vector in vectors:
                embeddings.append(
                    validate_embedding(vector.tolist())
                )

    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingError(
            "Failed to generate document embeddings."
        ) from exc

    if len(embeddings) != len(texts):
        raise EmbeddingError(
            f"Expected {len(texts)} embeddings, "
            f"received {len(embeddings)}."
        )

    return embeddings


def embed_texts(
    texts: Sequence[str],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
    """Embed a sequence of document texts.

    Empty or whitespace-only passages are rejected to avoid silently
    producing vectors for invalid chunks.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    normalized_texts = [text.strip() for text in texts]

    if any(not text for text in normalized_texts):
        raise ValueError("Texts must not contain empty passages.")

    return _embed_texts_sync(
        normalized_texts,
        batch_size=batch_size,
    )


async def embed_texts_async(
    texts: Sequence[str],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
    """Generate embeddings without blocking the async event loop."""
    return await asyncio.to_thread(
        embed_texts,
        texts,
        batch_size=batch_size,
    )


def embed_chunks(
    chunks: Sequence[TextChunk],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict]:
    """Return chunk metadata paired with its embedding vector."""
    if not chunks:
        return []

    embeddings = embed_texts(
        [chunk.content for chunk in chunks],
        batch_size=batch_size,
    )

    return [
        {
            "chunk_index": chunk.chunk_index,
            "page_number": chunk.page_number,
            "content": chunk.content,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "embedding": embedding,
        }
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]


async def embed_chunks_async(
    chunks: Sequence[TextChunk],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict]:
    """Asynchronously embed chunks and retain their metadata."""
    return await asyncio.to_thread(
        embed_chunks,
        chunks,
        batch_size=batch_size,
    )
