
"""Tests for the shared RAG workflow state."""

from graph.state import RAGState


def test_state_accepts_question_and_retrieved_chunks():
    state: RAGState = {
        "question": "What are India's foreign policy priorities?",
        "retrieved_chunks": [
            {
                "chunk_id": "chunk-1",
                "content": "Example report passage.",
                "page_number": 10,
            }
        ],
    }

    assert state["question"].startswith("What are")
    assert len(state["retrieved_chunks"]) == 1


def test_state_supports_evidence_assessment():
    state: RAGState = {
        "evidence_sufficient": False,
        "evidence_reason": "No relevant evidence found.",
        "abstained": True,
    }

    assert state["evidence_sufficient"] is False
    assert state["abstained"] is True


def test_state_supports_answer_and_citations():
    state: RAGState = {
        "answer": "The report discusses regional cooperation.",
        "citations": [
            {
                "chunk_id": "chunk-1",
                "page_number": 10,
            }
        ],
        "validation_passed": True,
        "validation_errors": [],
    }

    assert state["answer"]
    assert state["citations"][0]["page_number"] == 10
    assert state["validation_passed"] is True


def test_state_can_be_initialized_incrementally():
    state: RAGState = {}

    state["question"] = "Summarize the report."
    state["retrieved_chunks"] = []
    state["retrieval_error"] = ""

    assert state["question"] == "Summarize the report."
    assert state["retrieved_chunks"] == []
