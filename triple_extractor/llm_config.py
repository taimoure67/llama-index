"""
LLM configuration for GPT-OSS-120B via an OpenAI-compatible API endpoint.
"""

import os

from llama_index.llms.openai_like import OpenAILike


def get_llm(
    model: str = "gpt-oss-120b",
    api_base: str | None = None,
    api_key: str | None = None,
    context_window: int = 128_000,
    is_chat_model: bool = True,
    is_function_calling_model: bool = False,
    max_tokens: int = 1024,
) -> OpenAILike:
    """
    Create an OpenAILike LLM instance configured for GPT-OSS-120B.

    Credentials are read from environment variables if not provided:
      - OPENAI_API_BASE: base URL of the OpenAI-compatible endpoint
      - OPENAI_API_KEY:  API key for authentication

    Args:
        model: Model name to use (default: "gpt-oss-120b").
        api_base: Base URL for the API endpoint.
        api_key: API key for authentication.
        context_window: Maximum context window size in tokens.
        is_chat_model: Whether the endpoint uses the chat completions format.
        is_function_calling_model: Whether the model supports function calling.
        max_tokens: Maximum number of tokens to generate per response.

    Returns:
        Configured OpenAILike LLM instance.
    """
    resolved_base = api_base or os.environ.get("OPENAI_API_BASE", "")
    resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    if not resolved_base:
        raise ValueError(
            "api_base must be provided or OPENAI_API_BASE environment variable must be set."
        )
    if not resolved_key:
        raise ValueError(
            "api_key must be provided or OPENAI_API_KEY environment variable must be set."
        )

    return OpenAILike(
        model=model,
        api_base=resolved_base,
        api_key=resolved_key,
        context_window=context_window,
        is_chat_model=is_chat_model,
        is_function_calling_model=is_function_calling_model,
        max_tokens=max_tokens,
    )
