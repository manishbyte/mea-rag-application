
"""Reusable Streamlit UI components."""

import re
import time
from typing import Any
from urllib.parse import urlparse

import streamlit as st


def render_citations(citations: Any) -> None:
    """Render citations and page references returned by the API."""
    if not citations:
        return

    if isinstance(citations, dict):
        citations = [citations]

    if not isinstance(citations, list):
        st.caption("Citation metadata is unavailable.")
        return

    with st.expander(f"Sources ({len(citations)})", expanded=False):
        for index, citation in enumerate(citations, start=1):
            if isinstance(citation, str):
                st.markdown(f"**Source {index}:** {citation}")
                continue

            if not isinstance(citation, dict):
                continue

            source = (
                citation.get("source")
                or citation.get("source_name")
                or citation.get("document")
                or citation.get("filename")
                or citation.get("title")
                or f"Source {index}"
            )

            page = (
                citation.get("page")
                or citation.get("page_number")
                or citation.get("page_no")
            )

            url = citation.get("page_url") or citation.get("url")
            snippet = citation.get("snippet") or citation.get("text")

            label = f"**{index}. {source}**"

            if page is not None:
                label += f" — Page {page}"

            if isinstance(url, str):
                parsed = urlparse(url)

                if parsed.scheme in {"http", "https"} and parsed.netloc:
                    st.markdown(f"{label} · [Open source]({url})")
                else:
                    st.markdown(label)
            else:
                st.markdown(label)

            if snippet:
                st.caption(str(snippet)[:800])


def render_messages(messages: list[dict[str, Any]]) -> None:
    """Render persisted conversation messages without replaying them."""
    for message in messages:
        role = message.get("role", "assistant")

        if role not in {"user", "assistant"}:
            continue

        with st.chat_message(role):
            content = message.get("content", "")

            if content:
                st.markdown(content)
            else:
                st.caption("No message content.")

            if role == "assistant":
                render_citations(message.get("citations"))


def stream_answer(text: str):
    """
    Progressively display an already-generated answer.

    This is UI-level streaming: the API has already returned the complete
    answer before this function starts displaying it.
    """
    if not isinstance(text, str) or not text:
        return

    # Keep whitespace attached to each word so paragraphs and line breaks
    # remain intact as the answer is progressively displayed.
    chunks = re.findall(r"\S+\s*", text)

    for chunk in chunks:
        yield chunk
        time.sleep(0.015)


def show_api_error(message: str) -> None:
    """Display an API error."""
    st.error(message)


def show_loading_status(
    message: str = "Searching the annual report and preparing an answer…",
):
    """Display an expandable loading status."""
    return st.status(message, expanded=True)
