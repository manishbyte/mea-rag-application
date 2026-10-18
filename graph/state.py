
"""Shared state schema for the MEA Annual Report RAG workflow."""

from __future__ import annotations

from typing import Any, TypedDict


class RAGState(TypedDict, total=False):
    """State passed between LangGraph nodes.

    Nodes return partial dictionaries containing only the fields they update.
    """

    # User input
    question: str

    # Retrieval
    retrieved_chunks: list[dict[str, Any]]
    retrieval_error: str

    # Evidence assessment
    evidence_sufficient: bool
    evidence_reason: str

    # Answer generation
    answer: str
    citations: list[dict[str, Any]]
    abstained: bool

    # Output validation
    validation_passed: bool
    validation_errors: list[str]

    # Operational diagnostics
    workflow_error: str
