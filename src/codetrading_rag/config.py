"""Configuration loader for CodeTrading-RAG.

Loads settings from .env file with sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _load_env() -> None:
    """Load .env file from project root (walks up from this file's location)."""
    # Try to find .env relative to the project root
    current = Path(__file__).resolve().parent
    for _ in range(5):  # Walk up at most 5 levels
        env_path = current / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            return
        current = current.parent
    # Fallback: load from cwd
    load_dotenv()


@dataclass
class Config:
    """Application configuration with defaults."""

    # LLM
    llm_backend: str = "claude"
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-5-20250929"
    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_model: str = ""

    # Embeddings
    embedding_backend: str = "huggingface"
    openai_api_key: str = ""
    hf_embedding_model: str = "sentence-transformers/all-mpnet-base-v2"

    # Vector store
    chroma_db_path: str = "./data/chroma_db"

    # Data directory
    data_dir: str = "./data"

    # Transcript fetching
    transcript_max_retries: int = 3
    transcript_retry_delay: float = 2.0
    transcript_enable_fallbacks: bool = True
    cookies_from_browser: str = ""  # e.g. "chrome", "firefox", "brave" — yt-dlp reads cookies directly

    # Retriever
    retriever_k: int = 5

    # GUI
    gui_port: int = 7860
    gui_share: bool = False

    @classmethod
    def from_env(cls) -> Config:
        """Create a Config instance from environment variables."""
        _load_env()
        return cls(
            llm_backend=os.getenv("LLM_BACKEND", "claude").lower(),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            claude_model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929"),
            lmstudio_base_url=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
            lmstudio_model=os.getenv("LMSTUDIO_MODEL", ""),
            embedding_backend=os.getenv("EMBEDDING_BACKEND", "huggingface").lower(),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            hf_embedding_model=os.getenv(
                "HF_EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"
            ),
            chroma_db_path=os.getenv("CHROMA_DB_PATH", "./data/chroma_db"),
            data_dir=os.getenv("DATA_DIR", "./data"),
            transcript_max_retries=int(os.getenv("TRANSCRIPT_MAX_RETRIES", "3")),
            transcript_retry_delay=float(os.getenv("TRANSCRIPT_RETRY_DELAY", "1.0")),
            transcript_enable_fallbacks=os.getenv("TRANSCRIPT_ENABLE_FALLBACKS", "true").lower() == "true",
            cookies_from_browser=os.getenv("COOKIES_FROM_BROWSER", ""),
            retriever_k=int(os.getenv("RETRIEVER_K", "5")),
            gui_port=int(os.getenv("GUI_PORT", "7860")),
            gui_share=os.getenv("GUI_SHARE", "false").lower() == "true",
        )

    def validate(self) -> list[str]:
        """Return a list of configuration warnings/errors."""
        issues: list[str] = []

        if self.llm_backend == "claude" and not self.anthropic_api_key:
            issues.append("ANTHROPIC_API_KEY is required when LLM_BACKEND=claude")
        if self.llm_backend == "lmstudio" and not self.lmstudio_model:
            issues.append("LMSTUDIO_MODEL is required when LLM_BACKEND=lmstudio")
        if self.embedding_backend == "openai" and not self.openai_api_key:
            issues.append("OPENAI_API_KEY is required when EMBEDDING_BACKEND=openai")
        if self.llm_backend not in ("claude", "lmstudio"):
            issues.append(f"Invalid LLM_BACKEND: {self.llm_backend!r} (use 'claude' or 'lmstudio')")
        if self.embedding_backend not in ("openai", "huggingface"):
            issues.append(
                f"Invalid EMBEDDING_BACKEND: {self.embedding_backend!r} "
                "(use 'openai' or 'huggingface')"
            )

        return issues
