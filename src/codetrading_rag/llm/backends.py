"""LLM backend factory: Claude API and LM Studio support."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel

if TYPE_CHECKING:
    from codetrading_rag.config import Config

logger = logging.getLogger(__name__)


def get_llm(config: Config) -> BaseChatModel:
    """Create an LLM instance based on configuration.

    Args:
        config: Application configuration.

    Returns:
        A BaseChatModel instance (Claude or LM Studio via OpenAI-compatible API).

    Raises:
        ValueError: If the configured backend is not supported or missing credentials.
    """
    if config.llm_backend == "claude":
        if not config.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required for Claude backend. "
                "Set it in your .env file."
            )

        from langchain_anthropic import ChatAnthropic

        logger.info("Using Claude LLM (%s)", config.claude_model)
        return ChatAnthropic(
            model=config.claude_model,
            anthropic_api_key=config.anthropic_api_key,
            temperature=0.3,
            max_tokens=4096,
        )

    elif config.llm_backend == "lmstudio":
        if not config.lmstudio_model:
            raise ValueError(
                "LMSTUDIO_MODEL is required for LM Studio backend. "
                "Set it in your .env file."
            )

        from langchain_openai import ChatOpenAI

        logger.info(
            "Using LM Studio LLM (%s at %s)",
            config.lmstudio_model,
            config.lmstudio_base_url,
        )
        return ChatOpenAI(
            base_url=config.lmstudio_base_url,
            api_key="lm-studio",
            model=config.lmstudio_model,
            temperature=0.3,
        )

    else:
        raise ValueError(
            f"Unsupported LLM backend: {config.llm_backend!r}. "
            "Use 'claude' or 'lmstudio'."
        )
