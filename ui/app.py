
"""Streamlit frontend for the MEA Annual Report 2025 RAG chatbot."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import streamlit as st

from ui.api_client import APIClient, APIError, get_client
from ui.components import (
    render_citations,
    render_messages,
    show_api_error,
    stream_answer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ENTRY = PROJECT_ROOT / "evaluation" / "run.py"
EVALUATION_DATASET = PROJECT_ROOT / "evaluation" / "datasets" / "mea_v1.jsonl"


st.set_page_config(
    page_title="MEA Annual Report 2025 | RAG Assistant",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {
        max-width: 1100px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
      }
      [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.2);
      }
      .app-subtitle {
        color: #7a8491;
        font-size: 0.95rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------

def initialize_state() -> None:
    defaults = {
        "access_token": None,
        "user": None,
        "page": "Chat",
        "conversation_id": None,
        "messages": [],
        "conversations": [],
        "auth_mode": "Login",
        "evaluation_summary": None,
        "evaluation_return_code": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_state()
client: APIClient = get_client()


# ---------------------------------------------------------------------
# Authentication and conversation helpers
# ---------------------------------------------------------------------

def clear_authentication() -> None:
    """Clear local credentials and conversation data."""
    st.session_state["access_token"] = None
    st.session_state["user"] = None
    st.session_state["conversation_id"] = None
    st.session_state["messages"] = []
    st.session_state["conversations"] = []


def do_logout() -> None:
    token = st.session_state.get("access_token")

    if token:
        try:
            client.logout(token)
        except APIError:
            pass

    clear_authentication()
    st.session_state["page"] = "Chat"
    st.session_state["auth_mode"] = "Login"
    st.rerun()


def refresh_conversations() -> None:
    token = st.session_state.get("access_token")

    if not token:
        return

    try:
        st.session_state["conversations"] = client.list_conversations(token)
    except APIError as exc:
        if exc.status_code == 401:
            clear_authentication()
            st.rerun()

        show_api_error(exc.message)


def load_conversation(conversation_id: str) -> None:
    token = st.session_state.get("access_token")

    if not token:
        return

    try:
        result = client.get_conversation(token, conversation_id)
        conversation = result.get("conversation", {})
        messages = result.get("messages", [])

        st.session_state["conversation_id"] = (
            conversation.get("id", conversation_id)
        )
        st.session_state["messages"] = messages
        st.session_state["page"] = "Chat"
        st.rerun()

    except APIError as exc:
        if exc.status_code == 401:
            clear_authentication()
            st.rerun()

        show_api_error(exc.message)


def start_new_chat() -> None:
    st.session_state["conversation_id"] = None
    st.session_state["messages"] = []
    st.session_state["page"] = "Chat"
    st.rerun()


# ---------------------------------------------------------------------
# Login and registration
# ---------------------------------------------------------------------

def render_authentication() -> None:
    st.title("MEA Annual Report 2025")

    st.markdown(
        '<p class="app-subtitle">A source-grounded assistant for the '
        "Ministry of External Affairs Annual Report.</p>",
        unsafe_allow_html=True,
    )

    _, center, _ = st.columns([1, 1.4, 1])

    with center:
        st.subheader(st.session_state["auth_mode"])

        with st.form("authentication_form", clear_on_submit=False):
            email = st.text_input(
                "Email address",
                placeholder="you@example.com",
                autocomplete="email",
            )

            password = st.text_input(
                "Password",
                type="password",
                autocomplete="current-password",
            )

            if st.session_state["auth_mode"] == "Register":
                confirm_password = st.text_input(
                    "Confirm password",
                    type="password",
                )
            else:
                confirm_password = password

            submitted = st.form_submit_button(
                st.session_state["auth_mode"],
                use_container_width=True,
                type="primary",
            )

        if st.session_state["auth_mode"] == "Login":
            if st.button("Create an account", use_container_width=True):
                st.session_state["auth_mode"] = "Register"
                st.rerun()
        else:
            if st.button("Back to login", use_container_width=True):
                st.session_state["auth_mode"] = "Login"
                st.rerun()

        if not submitted:
            return

        if not email.strip() or not password:
            st.warning("Enter both your email address and password.")
            return

        if st.session_state["auth_mode"] == "Register":
            if password != confirm_password:
                st.error("The passwords do not match.")
                return

            if len(password) < 8:
                st.error("Password must contain at least 8 characters.")
                return

        try:
            if st.session_state["auth_mode"] == "Register":
                with st.spinner("Creating your account…"):
                    client.register(email.strip(), password)

                st.success("Account created. Please log in.")
                st.session_state["auth_mode"] = "Login"
                st.rerun()

            else:
                with st.spinner("Signing you in…"):
                    result = client.login(email.strip(), password)

                token = result.get("access_token")

                if not token:
                    st.error("The API did not return an access token.")
                    return

                st.session_state["access_token"] = token
                st.session_state["user"] = result.get("user", {})
                st.session_state["conversation_id"] = None
                st.session_state["messages"] = []
                st.session_state["page"] = "Chat"

                try:
                    st.session_state["user"] = client.me(token)
                except APIError:
                    pass

                st.rerun()

        except APIError as exc:
            show_api_error(exc.message)


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

def render_sidebar() -> None:
    with st.sidebar:
        st.title("MEA RAG")
        st.caption("Annual Report 2025")

        page = st.radio(
            "Workspace",
            options=["Chat", "Evaluation"],
            index=0 if st.session_state["page"] == "Chat" else 1,
            key="workspace_navigation",
        )

        st.session_state["page"] = page

        st.divider()

        user = st.session_state.get("user") or {}
        email = user.get("email", "Signed in")
        st.caption(f"Logged in as {email}")

        if st.button("＋ New chat", use_container_width=True):
            start_new_chat()

        st.subheader("Conversation history")

        if st.button("Refresh history", use_container_width=True):
            refresh_conversations()

        conversations = st.session_state.get("conversations", [])

        if not conversations:
            st.caption("Your saved conversations will appear here.")
        else:
            for item in conversations:
                if not isinstance(item, dict):
                    continue

                conversation_id = item.get("id")

                if not conversation_id:
                    continue

                title = str(item.get("title") or "New conversation")[:55]

                is_current = (
                    str(conversation_id)
                    == str(st.session_state.get("conversation_id"))
                )

                if st.button(
                    ("● " if is_current else "") + title,
                    key=f"conversation_{conversation_id}",
                    use_container_width=True,
                ):
                    load_conversation(str(conversation_id))

        st.divider()
        st.caption("Answers should be checked against their cited sources.")

        if st.button("Log out", use_container_width=True):
            do_logout()


# ---------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------

def render_chat_page() -> None:
    st.title("MEA Annual Report Assistant")

    st.markdown(
        '<p class="app-subtitle">Ask questions about the 2025 annual report. '
        "Answers are grounded in retrieved report evidence.</p>",
        unsafe_allow_html=True,
    )

    messages = st.session_state.get("messages", [])

    if not messages:
        st.info(
            "Start by asking a question about the MEA Annual Report 2025."
        )

        st.markdown("**Example questions**")

        examples = [
            "What are the major highlights of India's foreign policy?",
            "Summarize the report's key regional engagement initiatives.",
            "What does the report say about India's multilateral diplomacy?",
        ]

        cols = st.columns(len(examples))

        for col, example in zip(cols, examples):
            with col:
                if st.button(example, use_container_width=True):
                    st.session_state["pending_question"] = example
                    st.rerun()

    # Previously saved messages are displayed normally, not streamed again.
    render_messages(messages)

    pending_question = st.session_state.pop("pending_question", None)
    question = st.chat_input("Ask a question about the annual report…")

    if pending_question:
        question = pending_question

    if not question or not question.strip():
        return

    question = question.strip()
    token = st.session_state.get("access_token")

    if not token:
        clear_authentication()
        st.rerun()
        return

    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.chat_message("assistant"):
            with st.spinner(
                "Searching report passages, checking evidence, and preparing an answer…"
            ):
                result = client.send_message(
                    token=token,
                    question=question,
                    conversation_id=st.session_state.get("conversation_id"),
                )

            conversation_id = result.get("conversation_id")

            if conversation_id:
                st.session_state["conversation_id"] = conversation_id

            answer = result.get("answer") or "The API returned no answer."

            # UI-level streaming: reveal the completed answer progressively.
            st.write_stream(stream_answer(answer))

            if result.get("abstained"):
                st.info(
                    "The system abstained because the retrieved evidence "
                    "was insufficient to support a reliable answer."
                )

            if result.get("validation_passed") is False:
                st.warning(
                    "The response did not pass all output validation checks."
                )

            render_citations(result.get("citations"))

        if conversation_id:
            history = client.get_conversation(token, conversation_id)
            st.session_state["messages"] = history.get("messages", [])
        else:
            st.session_state["messages"] = messages + [
                {"role": "user", "content": question},
                {
                    "role": "assistant",
                    "content": answer,
                    "citations": result.get("citations", []),
                },
            ]

        refresh_conversations()
        st.rerun()

    except APIError as exc:
        if exc.status_code == 401:
            clear_authentication()
            st.warning("Your session has expired. Please log in again.")
            st.rerun()

        elif exc.status_code == 422:
            show_api_error(
                "The question could not be processed. Try rephrasing it."
            )

        else:
            show_api_error(exc.message)
            st.caption(
                "Your question may not have been saved. Check conversation "
                "history before retrying."
            )


# ---------------------------------------------------------------------
# Evaluation summary parsing
# ---------------------------------------------------------------------

def extract_evaluation_summary(output: str) -> dict[str, Any] | None:
    """
    Extract the JSON object following the evaluation summary marker.

    Other output, including logger messages, warnings, and the results path,
    is ignored.
    """
    marker = "=== MEA RAG Evaluation Summary ==="
    marker_position = output.find(marker)

    if marker_position != -1:
        search_area = output[marker_position + len(marker):]
    else:
        # Fallback for runners that emit JSON without the heading.
        search_area = output

    decoder = json.JSONDecoder()
    position = 0

    while True:
        json_start = search_area.find("{", position)

        if json_start == -1:
            return None

        try:
            parsed, _ = decoder.raw_decode(search_area[json_start:])

            if (
                isinstance(parsed, dict)
                and (
                    "total_questions" in parsed
                    or "retrieval" in parsed
                    or "answer_quality" in parsed
                )
            ):
                return parsed

            position = json_start + 1

        except json.JSONDecodeError:
            position = json_start + 1


def render_metric(label: str, value: Any, suffix: str = "") -> None:
    """Display a formatted evaluation metric."""
    if isinstance(value, float):
        formatted_value = f"{value:.4f}".rstrip("0").rstrip(".")
    elif value is None:
        formatted_value = "N/A"
    else:
        formatted_value = str(value)

    st.metric(label=label, value=f"{formatted_value}{suffix}")


# ---------------------------------------------------------------------
# Evaluation dashboard
# ---------------------------------------------------------------------

def render_evaluation_page() -> None:
    st.title("RAG Evaluation")
    st.caption(
        "Measure retrieval quality, answer quality, citation accuracy, "
        "security behavior, and response latency."
    )

    if "evaluation_summary" not in st.session_state:
        st.session_state["evaluation_summary"] = None

    if "evaluation_return_code" not in st.session_state:
        st.session_state["evaluation_return_code"] = None

    if st.button(
        "Run Evaluation",
        type="primary",
        use_container_width=True,
    ):
        st.session_state["evaluation_summary"] = None
        st.session_state["evaluation_return_code"] = None

        with st.spinner("Running evaluation… Please wait."):
            try:
                # Use the project's existing evaluation module.
                result = subprocess.run(
                    [sys.executable, "-m", "evaluation.run"],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=1800,
                    env=os.environ.copy(),
                    check=False,
                )

                # Logs can go to stderr while the summary goes to stdout,
                # or vice versa. Inspect both streams.
                combined_output = (
                    (result.stdout or "")
                    + "\n"
                    + (result.stderr or "")
                )

                summary = extract_evaluation_summary(combined_output)

                st.session_state["evaluation_return_code"] = result.returncode
                st.session_state["evaluation_summary"] = summary

                if summary is not None:
                    # A usable summary is more informative than raw logs.
                    # Show it even if the runner returned a non-zero code.
                    if result.returncode == 0:
                        st.success("Evaluation completed successfully.")
                    else:
                        st.warning(
                            "An evaluation summary was produced, but the "
                            "runner reported an error."
                        )

                elif result.returncode != 0:
                    st.error(
                        "Evaluation could not complete successfully. "
                        "Check the application logs for diagnostic details."
                    )

                else:
                    st.error(
                        "The evaluation finished, but its summary could "
                        "not be extracted. Check the runner's output format."
                    )

            except subprocess.TimeoutExpired:
                st.session_state["evaluation_return_code"] = -1
                st.session_state["evaluation_summary"] = None
                st.error("Evaluation timed out. Please try again.")

            except Exception:
                st.session_state["evaluation_return_code"] = -1
                st.session_state["evaluation_summary"] = None
                st.error(
                    "An unexpected error occurred while running evaluation. "
                    "Check the application logs for diagnostic details."
                )

    summary = st.session_state.get("evaluation_summary")

    if not isinstance(summary, dict):
        st.info("Run an evaluation to view the latest results.")
        return

    st.divider()

    # Execution summary
    st.subheader("Execution Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        render_metric("Total questions", summary.get("total_questions", 0))

    with col2:
        render_metric(
            "Successful",
            summary.get("successful_executions", 0),
        )

    with col3:
        render_metric("Failed", summary.get("failed_executions", 0))

    # Safely handle missing metric groups.
    retrieval = summary.get("retrieval") or {}
    quality = summary.get("answer_quality") or {}
    citations = summary.get("citations") or {}
    security = summary.get("security") or {}
    latency = summary.get("latency") or {}

    if not isinstance(retrieval, dict):
        retrieval = {}
    if not isinstance(quality, dict):
        quality = {}
    if not isinstance(citations, dict):
        citations = {}
    if not isinstance(security, dict):
        security = {}
    if not isinstance(latency, dict):
        latency = {}

    # Retrieval quality
    st.subheader("Retrieval Quality")

    col1, col2 = st.columns(2)

    with col1:
        render_metric("Hit rate", retrieval.get("hit_rate"))
        render_metric(
            "Mean reciprocal rank",
            retrieval.get("mean_reciprocal_rank"),
        )

    with col2:
        render_metric("Mean recall@k", retrieval.get("mean_recall_at_k"))
        render_metric("Mean page recall", retrieval.get("mean_page_recall"))

    # Answer quality
    st.subheader("Answer Quality")

    col1, col2, col3 = st.columns(3)

    with col1:
        render_metric("Token F1", quality.get("mean_token_f1"))

    with col2:
        render_metric("ROUGE-L F1", quality.get("mean_rouge_l_f1"))

    with col3:
        render_metric("Exact match rate", quality.get("exact_match_rate"))

    # Citation quality
    st.subheader("Citation Quality")

    col1, col2 = st.columns(2)

    with col1:
        render_metric(
            "Marker metadata precision",
            citations.get("mean_marker_metadata_precision"),
        )

    with col2:
        render_metric(
            "Relevant page recall",
            citations.get("mean_relevant_page_recall"),
        )

    # Security
    st.subheader("Security")

    col1, col2 = st.columns(2)

    with col1:
        render_metric(
            "Behavior pass rate",
            security.get("behavior_pass_rate"),
        )

    with col2:
        behavior_counts = security.get("expected_behavior_counts", {})

        if isinstance(behavior_counts, dict):
            tested_behaviors = sum(
                count
                for count in behavior_counts.values()
                if isinstance(count, (int, float))
            )
        else:
            tested_behaviors = None

        render_metric("Expected behaviors tested", tested_behaviors)

    if isinstance(security.get("expected_behavior_counts"), dict):
        with st.expander("Security behavior breakdown"):
            for behavior, count in security["expected_behavior_counts"].items():
                st.write(f"**{behavior.replace('_', ' ').title()}:** {count}")

    # Latency
    st.subheader("Latency")

    col1, col2 = st.columns(2)

    with col1:
        render_metric(
            "Mean response time",
            latency.get("mean_seconds"),
            " s",
        )

    with col2:
        render_metric(
            "Maximum response time",
            latency.get("max_seconds"),
            " s",
        )


# ---------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------

def main() -> None:
    token = st.session_state.get("access_token")

    if not token:
        render_authentication()
        return

    if not st.session_state.get("user"):
        try:
            st.session_state["user"] = client.me(token)
        except APIError:
            clear_authentication()
            render_authentication()
            return

    render_sidebar()

    if st.session_state["page"] == "Evaluation":
        render_evaluation_page()
    else:
        if not st.session_state.get("conversations"):
            refresh_conversations()

        render_chat_page()


if __name__ == "__main__":
    main()
