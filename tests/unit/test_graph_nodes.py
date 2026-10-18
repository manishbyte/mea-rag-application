
"""Unit tests for RAG retrieval and evidence-assessment nodes."""

from unittest.mock import AsyncMock, patch

import pytest

from graph.nodes import evidence_assessment_node, retrieval_node


@pytest.mark.asyncio
async def test_retrieval_node_returns_chunks():
    chunks = [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "document_title": "MEA Annual Report 2025",
            "filename": "mea-report.pdf",
            "page_number": 10,
            "content": "India emphasizes regional cooperation.",
        }
    ]

    with patch(
        "graph.nodes.hybrid_search",
        new_callable=AsyncMock,
        return_value=chunks,
    ):
        result = await retrieval_node(
            {"question": "What are India's priorities?"}
        )

    assert result["retrieved_chunks"] == chunks
    assert result["retrieval_error"] == ""
    assert result["workflow_error"] == ""


@pytest.mark.asyncio
async def test_retrieval_node_rejects_empty_question():
    result = await retrieval_node({"question": "  "})

    assert result["retrieved_chunks"] == []
    assert result["retrieval_error"]
    assert result["workflow_error"] == ""


@pytest.mark.asyncio
async def test_evidence_node_accepts_valid_evidence():
    chunks = [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "document_title": "MEA Annual Report 2025",
            "page_number": 10,
            "content": "India emphasizes regional cooperation.",
        }
    ]

    result = await evidence_assessment_node(
        {"retrieved_chunks": chunks, "retrieval_error": ""}
    )

    assert result["evidence_sufficient"] is True
    assert result["abstained"] is False


@pytest.mark.asyncio
async def test_evidence_node_abstains_without_evidence():
    result = await evidence_assessment_node(
        {"retrieved_chunks": [], "retrieval_error": ""}
    )

    assert result["evidence_sufficient"] is False
    assert result["abstained"] is True


@pytest.mark.asyncio
async def test_evidence_node_handles_retrieval_failure():
    result = await evidence_assessment_node(
        {
            "retrieved_chunks": [],
            "retrieval_error": "Database unavailable.",
        }
    )

    assert result["evidence_sufficient"] is False
    assert result["abstained"] is True
