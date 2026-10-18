
"""LangGraph workflow for the MEA Annual Report RAG chatbot."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from graph.edges import (
    route_after_answer_generation,
    route_after_evidence_assessment,
    route_after_output_validation,
    route_after_retrieval,
)
from graph.nodes import (
    answer_generation_node,
    evidence_assessment_node,
    output_validation_node,
    retrieval_node,
)
from graph.state import RAGState


def build_workflow():
    """Build and compile the guarded MEA RAG workflow."""
    builder = StateGraph(RAGState)

    builder.add_node("retrieval", retrieval_node)
    builder.add_node("evidence_assessment", evidence_assessment_node)
    builder.add_node("answer_generation", answer_generation_node)
    builder.add_node("output_validation", output_validation_node)

    builder.add_edge(START, "retrieval")

    builder.add_conditional_edges(
        "retrieval",
        route_after_retrieval,
        {
            "assess_evidence": "evidence_assessment",
            "stop": END,
        },
    )

    builder.add_conditional_edges(
        "evidence_assessment",
        route_after_evidence_assessment,
        {
            "generate_answer": "answer_generation",
            "stop": END,
        },
    )

    builder.add_conditional_edges(
        "answer_generation",
        route_after_answer_generation,
        {
            "validate_output": "output_validation",
        },
    )

    builder.add_conditional_edges(
        "output_validation",
        route_after_output_validation,
        {
            "finish": END,
        },
    )

    return builder.compile()


# Reusable compiled graph for the API and other callers.
rag_workflow = build_workflow()


async def run_workflow(question: str) -> RAGState:
    """Run the RAG workflow asynchronously for one question."""
    return await rag_workflow.ainvoke({"question": question})
