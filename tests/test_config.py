"""Tests for configuration module."""

import os
from unittest.mock import patch

from codetrading_rag.config import Config


class TestConfig:
    """Tests for Config dataclass."""

    def test_defaults(self):
        """Config should have sensible defaults."""
        config = Config()
        assert config.llm_backend == "claude"
        assert config.embedding_backend == "huggingface"
        assert config.retriever_k == 5
        assert config.channel_url == "https://www.youtube.com/@CodeTradingCafe"

    def test_from_env(self):
        """Config.from_env should read from environment variables."""
        env = {
            "LLM_BACKEND": "lmstudio",
            "LMSTUDIO_MODEL": "test-model",
            "EMBEDDING_BACKEND": "openai",
            "OPENAI_API_KEY": "sk-test-123",
            "RETRIEVER_K": "10",
        }
        with patch.dict(os.environ, env, clear=False):
            config = Config.from_env()

        assert config.llm_backend == "lmstudio"
        assert config.lmstudio_model == "test-model"
        assert config.embedding_backend == "openai"
        assert config.openai_api_key == "sk-test-123"
        assert config.retriever_k == 10

    def test_validate_claude_no_key(self):
        """Validation should flag missing Anthropic key for Claude backend."""
        config = Config(llm_backend="claude", anthropic_api_key="")
        issues = config.validate()
        assert any("ANTHROPIC_API_KEY" in i for i in issues)

    def test_validate_claude_with_key(self):
        """Validation should pass with Anthropic key set."""
        config = Config(llm_backend="claude", anthropic_api_key="sk-test")
        issues = config.validate()
        assert not any("ANTHROPIC_API_KEY" in i for i in issues)

    def test_validate_lmstudio_no_model(self):
        """Validation should flag missing model for LM Studio backend."""
        config = Config(llm_backend="lmstudio", lmstudio_model="")
        issues = config.validate()
        assert any("LMSTUDIO_MODEL" in i for i in issues)

    def test_validate_openai_embeddings_no_key(self):
        """Validation should flag missing OpenAI key for OpenAI embeddings."""
        config = Config(embedding_backend="openai", openai_api_key="")
        issues = config.validate()
        assert any("OPENAI_API_KEY" in i for i in issues)

    def test_validate_invalid_backend(self):
        """Validation should flag invalid backend names."""
        config = Config(llm_backend="invalid")
        issues = config.validate()
        assert any("Invalid LLM_BACKEND" in i for i in issues)
