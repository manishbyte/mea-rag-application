
"""Unit tests for graph routing and workflow compilation."""

from graph.edges import (
    route_after_answer_generation,
    route_after_evidence_assessment,
    route_after_output_validation,
    route_after_retrieval,
)
from graph.workflow import build_workflow


def test_invalid_input_response_stops_after_retrieval():
    state = {
        "answer": "Please enter a valid question.",
        "retrieved_chunks": [],
        "retrieval_error": "",
    }

    assert route_after_retrieval(state) == "stop"


def test_retrieval_failure_stops_graph():
    state = {
        "answer": "I cannot answer reliably because evidence is insufficient.",
        "retrieval_error": "Retrieval failed.",
    }

    assert route_after_retrieval(state) == "stop"


def test_successful_retrieval_routes_to_evidence_assessment():
    state = {
        "answer": "",
        "retrieved_chunks": [{"content": "Report passage"}],
        "retrieval_error": "",
        "workflow_error": "",
    }

    assert route_after_retrieval(state) == "assess_evidence"


def test_sufficient_evidence_routes_to_generation():
    assert (
        route_after_evidence_assessment({"evidence_sufficient": True})
        == "generate_answer"
    )


def test_insufficient_evidence_skips_generation():
    assert (
        route_after_evidence_assessment({"evidence_sufficient": False})
        == "stop"
    )


def test_answer_generation_routes_to_output_validation():
    assert route_after_answer_generation({}) == "validate_output"


def test_output_validation_finishes_graph():
    assert route_after_output_validation(
        {"validation_passed": True}
    ) == "finish"

    assert route_after_output_validation(
        {"validation_passed": False}
    ) == "finish"


def test_workflow_compiles_with_expected_nodes():
    workflow = build_workflow()
    graph_nodes = workflow.get_graph().nodes

    assert "retrieval" in graph_nodes
    assert "evidence_assessment" in graph_nodes
    assert "answer_generation" in graph_nodes
    assert "output_validation" in graph_nodes
