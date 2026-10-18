
"""NVIDIA ChatNVIDIA connection for the MEA RAG application."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 2048


def get_llm() -> ChatNVIDIA:
    """Return a configured NVIDIA-hosted chat model."""
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()

    if not api_key or api_key == "replace_with_your_nvidia_api_key":
        raise RuntimeError(
            "NVIDIA_API_KEY is missing. Configure it in .env "
            "or your deployment environment."
        )

    model = os.getenv("NVIDIA_MODEL", DEFAULT_MODEL).strip()

    if not model:
        raise ValueError("NVIDIA_MODEL must not be empty.")

    try:
        temperature = float(
            os.getenv("NVIDIA_TEMPERATURE", str(DEFAULT_TEMPERATURE))
        )
        max_tokens = int(
            os.getenv("NVIDIA_MAX_TOKENS", str(DEFAULT_MAX_TOKENS))
        )
    except ValueError as exc:
        raise ValueError(
            "NVIDIA_TEMPERATURE must be numeric and "
            "NVIDIA_MAX_TOKENS must be an integer."
        ) from exc

    if not 0 <= temperature <= 2:
        raise ValueError("NVIDIA_TEMPERATURE must be between 0 and 2.")

    if max_tokens < 1:
        raise ValueError("NVIDIA_MAX_TOKENS must be greater than zero.")

    return ChatNVIDIA(
        model=model,
        nvidia_api_key=api_key,
        temperature=temperature,
        max_completion_tokens=max_tokens,
    )
