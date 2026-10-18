
"""Conditional routing functions for the MEA RAG graph."""

from __future__ import annotations

from graph.state import RAGState


def route_after_retrieval(state: RAGState) -> str:
    """Stop on input rejection or retrieval failure."""
    if state.get("answer"):
        return "stop"

    if state.get("retrieval_error") or state.get("workflow_error"):
        return "stop"

    return "assess_evidence"


def route_after_evidence_assessment(state: RAGState) -> str:
    """Allow generation only when evidence assessment succeeds."""
    if state.get("evidence_sufficient", False):
        return "generate_answer"

    return "stop"


def route_after_answer_generation(state: RAGState) -> str:
    """Send every generated or fallback answer through validation."""
    return "validate_output"


def route_after_output_validation(state: RAGState) -> str:
    """Finish after validation, whether it passes or fails."""
    return "finish"
