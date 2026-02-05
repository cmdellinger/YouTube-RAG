"""Embedding model factory for OpenAI and HuggingFace backends."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.embeddings import Embeddings

if TYPE_CHECKING:
    from codetrading_rag.config import Config

logger = logging.getLogger(__name__)


def get_embeddings(config: Config) -> Embeddings:
    """Create an embedding model based on configuration.

    Args:
        config: Application configuration.

    Returns:
        An Embeddings instance (OpenAI or HuggingFace).

    Raises:
        ValueError: If the configured backend is not supported.
    """
    if config.embedding_backend == "openai":
        from langchain_openai import OpenAIEmbeddings

        logger.info("Using OpenAI embeddings (text-embedding-3-small)")
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=config.openai_api_key,
        )

    elif config.embedding_backend == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info("Using HuggingFace embeddings (%s)", config.hf_embedding_model)
        return HuggingFaceEmbeddings(
            model_name=config.hf_embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    else:
        raise ValueError(
            f"Unsupported embedding backend: {config.embedding_backend!r}. "
            "Use 'openai' or 'huggingface'."
        )
