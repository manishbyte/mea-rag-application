
"""Query embedding utilities for semantic retrieval."""

import asyncio
import math
from functools import lru_cache

from fastembed import TextEmbedding

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384


class QueryEmbeddingError(RuntimeError):
    """Raised when a query cannot be embedded correctly."""


@lru_cache(maxsize=1)
def _get_embedding_model() -> TextEmbedding:
    """Create and cache the embedding model."""
    return TextEmbedding(model_name=EMBEDDING_MODEL_NAME)


def embed_query(query: str) -> list[float]:
    """Generate a validated embedding for one user query."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    try:
        model = _get_embedding_model()
        vector_generator = model.query_embed([query.strip()])
        vector = next(iter(vector_generator), None)

        if vector is None:
            raise QueryEmbeddingError(
                "The embedding model returned no query vector."
            )

        embedding = [float(value) for value in vector]

        if len(embedding) != EMBEDDING_DIMENSIONS:
            raise QueryEmbeddingError(
                f"Expected {EMBEDDING_DIMENSIONS} dimensions; "
                f"received {len(embedding)}."
            )

        if not all(math.isfinite(value) for value in embedding):
            raise QueryEmbeddingError(
                "The query embedding contains invalid numeric values."
            )

        return embedding

    except QueryEmbeddingError:
        raise
    except Exception as exc:
        raise QueryEmbeddingError(
            "Failed to generate the query embedding."
        ) from exc


async def embed_query_async(query: str) -> list[float]:
    """Generate an embedding without blocking the async event loop."""
    return await asyncio.to_thread(embed_query, query)
