
"""Asynchronous, evidence-grounded answer generation using NVIDIA ChatNVIDIA."""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from generation.llm import get_llm
from generation.prompts import (
    SYSTEM_PROMPT,
    build_evidence_context,
    build_user_prompt,
)
from guardrails.evidence_checks import assess_evidence

logger = logging.getLogger(__name__)

SOURCE_PATTERN = re.compile(r"\[(S[1-9]\d*)\]")
def _normalize_citation_markers(answer: str) -> str:
    """Normalize NVIDIA's full-width citation brackets."""
    return re.sub(
        r"【\s*(S[1-9]\d*)\s*】",
        r"[\1]",
        answer,
    )
MAX_GENERATION_ATTEMPTS = 2


class AnswerGenerationError(RuntimeError):
    """Raised when answer generation or citation checks fail."""


def _response_to_text(content: Any) -> str:
    """Convert a LangChain message content value to plain text."""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts: list[str] = []

        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                text_parts.append(item["text"])

        return "\n".join(text_parts).strip()

    return ""


def _extract_citations(
    answer: str,
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return source metadata corresponding to citation markers in the answer."""
    cited_ids = set(SOURCE_PATTERN.findall(answer))

    return [
        source
        for source in sources
        if source.get("source_id") in cited_ids
    ]


def _citation_error(
    answer: str,
    sources: list[dict[str, Any]],
) -> str | None:
    """Return a validation error if the answer's citations are invalid."""
    if not answer:
        return "The language model returned an empty answer."

    cited_ids = set(SOURCE_PATTERN.findall(answer))

    available_ids = {
        source.get("source_id")
        for source in sources
        if source.get("source_id")
    }

    if not cited_ids:
        return "The generated answer contains no recognized citation markers."

    unknown_ids = cited_ids - available_ids

    if unknown_ids:
        return (
            "The generated answer contains unknown citation IDs: "
            + ", ".join(sorted(unknown_ids))
        )

    if not _extract_citations(answer, sources):
        return "Citation markers could not be matched to source metadata."

    return None


async def generate_answer(
    question: str,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Generate an evidence-grounded answer and retry once if citations are invalid."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    question = question.strip()

    if len(question) > 4000:
        raise ValueError("Question cannot exceed 4000 characters.")

    sufficient, reason = assess_evidence(chunks)

    if not sufficient:
        return {
            "answer": (
                "I cannot answer this question reliably because the "
                "retrieved MEA report evidence is insufficient."
            ),
            "citations": [],
            "abstained": True,
            "evidence_reason": reason,
        }

    evidence_context, sources = build_evidence_context(chunks)

    if not evidence_context or not sources:
        return {
            "answer": (
                "I cannot answer this question reliably because no usable "
                "report passages are available."
            ),
            "citations": [],
            "abstained": True,
            "evidence_reason": "No usable evidence context was constructed.",
        }

    available_ids = [
        source["source_id"]
        for source in sources
        if source.get("source_id")
    ]

    base_prompt = build_user_prompt(question, evidence_context)

    retry_instruction = (
        "\n\nIMPORTANT CITATION REQUIREMENTS:\n"
        "- Answer the user's question directly using only the supplied evidence.\n"
        "- Cite factual statements using the exact source IDs provided in the "
        "evidence, such as [S1] or [S2].\n"
        f"- Valid source IDs for this response: {', '.join(available_ids)}.\n"
        "- Do not invent source IDs or citation formats.\n"
        "- Every substantive factual answer must have at least one valid citation.\n"
        "- If the evidence contains the answer, answer it rather than abstaining.\n"
        "- Return the answer with inline citations; do not explain these instructions."
    )

    try:
        llm = get_llm()
        last_validation_error = "No answer was generated."

        for attempt in range(MAX_GENERATION_ATTEMPTS):
            prompt = base_prompt

            if attempt > 0:
                prompt += retry_instruction
                logger.warning(
                    "Retrying answer generation because citation validation failed. "
                    "Attempt %s of %s.",
                    attempt + 1,
                    MAX_GENERATION_ATTEMPTS,
                )

            response = await llm.ainvoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
            )

            answer = _response_to_text(response.content)
            answer = _normalize_citation_markers(answer)
            
            logger.warning(
                "DEBUG raw NVIDIA response type: %s",
                type(response.content).__name__,
            )
            logger.warning(
                "DEBUG raw NVIDIA response: %r",
                str(response.content)[:3000],
            )
            logger.warning(
                "DEBUG parsed answer: %r",
                answer[:3000],
            )
            validation_error = _citation_error(answer, sources)

            if validation_error:
                last_validation_error = validation_error
                logger.warning(
                    "Answer validation failed on attempt %s/%s: %s",
                    attempt + 1,
                    MAX_GENERATION_ATTEMPTS,
                    validation_error,
                )
                continue

            citations = _extract_citations(answer, sources)

            logger.info(
                "Answer generated successfully with %s citation source(s).",
                len(citations),
            )

            return {
                "answer": answer,
                "citations": citations,
                "abstained": False,
                "evidence_reason": reason,
            }

        raise AnswerGenerationError(
            f"Answer failed citation validation after "
            f"{MAX_GENERATION_ATTEMPTS} attempts: {last_validation_error}"
        )

    except AnswerGenerationError:
        logger.exception("Answer generation or citation check failed.")
        raise

    except Exception as exc:
        logger.exception("NVIDIA answer generation failed.")

        raise AnswerGenerationError(
            f"Answer generation failed ({type(exc).__name__})."
        ) from exc
