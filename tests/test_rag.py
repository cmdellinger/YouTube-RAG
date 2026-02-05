"""Tests for RAG components (prompts, chain setup)."""

from unittest.mock import MagicMock, patch

import pytest

from codetrading_rag.config import Config
from codetrading_rag.rag import chain as rag_chain_module  # noqa: F401 — ensure module is loaded for patching
from codetrading_rag.rag.prompts import PROMPTS, get_prompt


class TestPrompts:
    """Tests for prompt templates."""

    def test_all_modes_exist(self):
        """All expected modes should have prompt templates."""
        assert "strategy" in PROMPTS
        assert "explain" in PROMPTS
        assert "review" in PROMPTS

    def test_get_prompt_valid(self):
        """get_prompt should return templates for valid modes."""
        for mode in ("strategy", "explain", "review"):
            prompt = get_prompt(mode)
            assert prompt is not None
            # All prompts should have {context} and {input} variables
            input_vars = prompt.input_variables
            assert "context" in input_vars or "input" in input_vars

    def test_get_prompt_invalid(self):
        """get_prompt should raise ValueError for unknown modes."""
        with pytest.raises(ValueError, match="Unknown prompt mode"):
            get_prompt("nonexistent")

    def test_prompts_contain_context_placeholder(self):
        """All system prompts should reference {context} for RAG."""
        for mode, prompt in PROMPTS.items():
            messages = prompt.messages
            # The system message should contain {context}
            system_msg = messages[0]
            assert "context" in system_msg.prompt.template, (
                f"Prompt '{mode}' missing {{context}} placeholder"
            )


class TestRAGChain:
    """Tests for the RAG chain (with mocked dependencies)."""

    @patch("codetrading_rag.rag.chain.get_embeddings")
    @patch("codetrading_rag.rag.chain.get_llm")
    @patch("codetrading_rag.rag.chain.VectorStore")
    def test_chain_init(self, mock_store_cls, mock_llm, mock_embeddings):
        """RAGChain should initialize all components."""
        mock_store_instance = MagicMock()
        mock_store_instance.document_count = 0
        mock_store_cls.return_value = mock_store_instance

        from codetrading_rag.rag.chain import RAGChain

        config = Config(llm_backend="claude", anthropic_api_key="test-key")
        chain = RAGChain(config)

        mock_embeddings.assert_called_once_with(config)
        mock_llm.assert_called_once_with(config)
        mock_store_cls.assert_called_once()

    @patch("codetrading_rag.rag.chain.get_embeddings")
    @patch("codetrading_rag.rag.chain.get_llm")
    @patch("codetrading_rag.rag.chain.VectorStore")
    def test_chain_query_empty_store(self, mock_store_cls, mock_llm, mock_embeddings):
        """Query on empty store should return a helpful message."""
        mock_store_instance = MagicMock()
        mock_store_instance.document_count = 0
        mock_store_cls.return_value = mock_store_instance

        from codetrading_rag.rag.chain import RAGChain

        config = Config(llm_backend="claude", anthropic_api_key="test-key")
        chain = RAGChain(config)
        response = chain.query("test question")

        assert "No documents" in response.answer
        assert response.source_documents == []
