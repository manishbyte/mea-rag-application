
"""Unit tests for grounded answer generation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from generation.answer_generator import generate_answer


@pytest.fixture
def valid_chunks():
    return [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "chunk_index": 0,
            "page_number": 12,
            "content": (
                "The Ministry continued diplomatic engagement with "
                "partner countries during the reporting period."
            ),
            "document_title": "MEA Annual Report 2025",
            "filename": "mea_annual_report_2025.pdf",
            "metadata": {},
        }
    ]


@pytest.mark.asyncio
async def test_generate_answer_with_valid_evidence(valid_chunks):
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=AIMessage(
            content=(
                "The Ministry continued diplomatic engagement "
                "with partner countries [S1]."
            )
        )
    )

    with patch(
        "generation.answer_generator.get_llm",
        return_value=mock_llm,
    ):
        result = await generate_answer(
            "What did the Ministry continue?",
            valid_chunks,
        )

    assert result["abstained"] is False
    assert "[S1]" in result["answer"]
    assert len(result["citations"]) == 1
    assert result["citations"][0]["page_number"] == 12
    mock_llm.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_answer_abstains_without_evidence():
    result = await generate_answer(
        "What did the Ministry do?",
        [],
    )

    assert result["abstained"] is True
    assert result["citations"] == []
    assert "insufficient" in result["answer"].lower()


@pytest.mark.asyncio
async def test_generate_answer_rejects_empty_question(valid_chunks):
    with pytest.raises(ValueError, match="non-empty"):
        await generate_answer("  ", valid_chunks)


@pytest.mark.asyncio
async def test_generate_answer_reports_empty_model_response(valid_chunks):
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=""))

    with patch(
        "generation.answer_generator.get_llm",
        return_value=mock_llm,
    ):
        from generation.answer_generator import AnswerGenerationError

        with pytest.raises(AnswerGenerationError, match="empty answer"):
            await generate_answer("Summarize the passage.", valid_chunks)


@pytest.mark.asyncio
async def test_only_cited_sources_are_returned(valid_chunks):
    second_chunk = {
        **valid_chunks[0],
        "chunk_id": "chunk-2",
        "page_number": 13,
        "content": "A separate report passage.",
    }

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content="The report states this [S1].")
    )

    with patch(
        "generation.answer_generator.get_llm",
        return_value=mock_llm,
    ):
        result = await generate_answer(
            "What does the report state?",
            valid_chunks + [second_chunk],
        )

    assert [c["source_id"] for c in result["citations"]] == ["S1"]
