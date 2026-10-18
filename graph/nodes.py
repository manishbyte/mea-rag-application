
"""Guarded retrieval, evidence assessment, and answer-generation nodes."""

from __future__ import annotations

import logging
from typing import Any

from generation.answer_generator import (
    AnswerGenerationError,
    generate_answer,
)
from graph.state import RAGState
from guardrails.evidence_checks import assess_evidence
from guardrails.injection_detection import detect_prompt_injection
from guardrails.input_checks import validate_question
from guardrails.output_checks import validate_output
from retrieval.hybrid_search import HybridSearchError, hybrid_search

logger = logging.getLogger(__name__)

INVALID_QUESTION_ANSWER = (
    "Please enter a valid question about the MEA Annual Report 2025."
)

INJECTION_REJECTION_ANSWER = (
    "I cannot process this question because it appears to contain "
    "instructions that conflict with the application's security rules. "
    "Please rephrase your question about the MEA Annual Report 2025."
)

INSUFFICIENT_EVIDENCE_ANSWER = (
    "I cannot answer this question reliably because the retrieved "
    "MEA report evidence is insufficient."
)

GENERATION_FAILURE_ANSWER = (
    "I cannot provide a reliable answer because the available evidence "
    "is insufficient to provide a validated answer. Please try again."
)

OUTPUT_VALIDATION_FAILURE_ANSWER = (
    "I cannot provide a reliable answer because the available evidence "
    "is insufficient to provide a validated answer. Please try again."
)


async def retrieval_node(state: RAGState) -> dict[str, Any]:
    """Validate input, detect injection, and retrieve evidence if allowed."""
    raw_question = state.get("question", "")

    input_result = validate_question(raw_question)

    if not input_result.is_valid:
        return {
            "question": raw_question if isinstance(raw_question, str) else "",
            "retrieved_chunks": [],
            "retrieval_error": "Input validation failed.",
            "evidence_sufficient": False,
            "evidence_reason": "Input validation failed.",
            "answer": INVALID_QUESTION_ANSWER,
            "citations": [],
            "abstained": True,
            "validation_passed": True,
            "validation_errors": [input_result.reason],
            "workflow_error": "",
        }

    question = input_result.normalized_question
    injection_result = detect_prompt_injection(question)

    if injection_result.is_suspicious:
        return {
            "question": question,
            "retrieved_chunks": [],
            "retrieval_error": "",
            "evidence_sufficient": False,
            "evidence_reason": "Question rejected by injection detection.",
            "answer": INJECTION_REJECTION_ANSWER,
            "citations": [],
            "abstained": True,
            "validation_passed": False,
            "validation_errors": [
                "The question was rejected by the security filter."
            ],
            "workflow_error": "",
        }

    try:
        chunks = await hybrid_search(
            query=question,
            top_k=5,
            candidate_k=20,
        )

        return {
            "question": question,
            "retrieved_chunks": chunks,
            "retrieval_error": "",
            "evidence_sufficient": False,
            "evidence_reason": "",
            "answer": "",
            "citations": [],
            "abstained": False,
            "validation_passed": False,
            "validation_errors": [],
            "workflow_error": "",
        }

    except (HybridSearchError, ValueError):
        logger.exception("Hybrid retrieval failed.")

        return {
            "question": question,
            "retrieved_chunks": [],
            "retrieval_error": "Retrieval failed.",
            "evidence_sufficient": False,
            "evidence_reason": (
                "Evidence assessment skipped because retrieval failed."
            ),
            "answer": INSUFFICIENT_EVIDENCE_ANSWER,
            "citations": [],
            "abstained": True,
            "validation_passed": False,
            "validation_errors": [],
            "workflow_error": "Retrieval failed.",
        }


async def evidence_assessment_node(
    state: RAGState,
) -> dict[str, Any]:
    """Assess retrieved evidence before allowing answer generation."""
    if state.get("retrieval_error", ""):
        return {
            "evidence_sufficient": False,
            "evidence_reason": (
                "Evidence assessment skipped because retrieval failed."
            ),
            "answer": state.get("answer") or INSUFFICIENT_EVIDENCE_ANSWER,
            "abstained": True,
        }

    if state.get("answer") in (
        INVALID_QUESTION_ANSWER,
        INJECTION_REJECTION_ANSWER,
    ):
        return {
            "evidence_sufficient": False,
            "evidence_reason": "Question was rejected by an input guardrail.",
            "abstained": True,
        }

    chunks = state.get("retrieved_chunks", [])
    sufficient, reason = assess_evidence(chunks)

    result: dict[str, Any] = {
        "evidence_sufficient": sufficient,
        "evidence_reason": reason,
        "abstained": not sufficient,
    }

    if not sufficient:
        result["answer"] = INSUFFICIENT_EVIDENCE_ANSWER
        result["citations"] = []

    return result


async def answer_generation_node(
    state: RAGState,
) -> dict[str, Any]:
    """Generate an answer only after evidence assessment succeeds."""
    if not state.get("evidence_sufficient", False):
        return {
            "answer": state.get("answer") or INSUFFICIENT_EVIDENCE_ANSWER,
            "citations": [],
            "abstained": True,
            "workflow_error": state.get("workflow_error", ""),
        }

    question = state.get("question", "")
    chunks = state.get("retrieved_chunks", [])

    input_result = validate_question(question)

    if not input_result.is_valid:
        return {
            "answer": INVALID_QUESTION_ANSWER,
            "citations": [],
            "abstained": True,
            "validation_passed": False,
            "validation_errors": [input_result.reason],
            "workflow_error": "",
        }

    normalized_question = input_result.normalized_question
    injection_result = detect_prompt_injection(normalized_question)

    if injection_result.is_suspicious:
        return {
            "answer": INJECTION_REJECTION_ANSWER,
            "citations": [],
            "abstained": True,
            "validation_passed": False,
            "validation_errors": [
                "The question was rejected by the security filter."
            ],
            "workflow_error": "",
        }

    sufficient, reason = assess_evidence(chunks)

    if not sufficient:
        return {
            "answer": INSUFFICIENT_EVIDENCE_ANSWER,
            "citations": [],
            "abstained": True,
            "evidence_sufficient": False,
            "evidence_reason": reason,
            "workflow_error": "",
        }

    try:
        result = await generate_answer(
            question=normalized_question,
            chunks=chunks,
        )

        return {
            "answer": result["answer"],
            "citations": result["citations"],
            "abstained": result["abstained"],
            "evidence_reason": result.get("evidence_reason", reason),
            "workflow_error": "",
        }

    except (AnswerGenerationError, ValueError):
        logger.exception("Answer generation node failed.")

        return {
            "answer": GENERATION_FAILURE_ANSWER,
            "citations": [],
            "abstained": True,
            "workflow_error": "Answer generation failed.",
        }


async def output_validation_node(
    state: RAGState,
) -> dict[str, Any]:
    """Validate the generated answer and its citation integrity."""
    answer = state.get("answer", "")
    citations = state.get("citations", [])
    chunks = state.get("retrieved_chunks", [])
    abstained = state.get("abstained", False)

    passed, errors = validate_output(
        answer=answer,
        citations=citations,
        retrieved_chunks=chunks,
        abstained=abstained,
    )

    if passed:
        return {
            "validation_passed": True,
            "validation_errors": [],
        }

    logger.warning(
        "Output validation failed. Errors: %s",
        errors,
    )

    return {
        "answer": OUTPUT_VALIDATION_FAILURE_ANSWER,
        "citations": [],
        "abstained": True,
        "validation_passed": False,
        "validation_errors": errors,
    }
