from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


DEFAULT_MODEL = "gpt-5.6-luna"


def is_llm_available() -> bool:
    """
    Check whether an OpenAI API key is configured.
    """

    return bool(
        os.getenv("OPENAI_API_KEY")
    )


def generate_ai_recommendation(
    prompt: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Generate a customer retention recommendation
    using the OpenAI Responses API.
    """

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    client = OpenAI(
        api_key=api_key
    )

    response = client.responses.create(
        model=model,
        input=prompt,
    )

    recommendation = (
        response.output_text.strip()
    )

    if not recommendation:
        raise RuntimeError(
            "The LLM returned an empty response."
        )

    return recommendation