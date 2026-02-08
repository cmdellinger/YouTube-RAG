"""RAG chain: retrieval + LLM response generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from codetrading_rag.config import Config
from codetrading_rag.embeddings.factory import get_embeddings
from codetrading_rag.llm.backends import get_llm
from codetrading_rag.rag.prompts import get_prompt
from codetrading_rag.vectorstore.store import VectorStore

logger = logging.getLogger(__name__)


def _format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a single context string."""
    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        header = f"[Source {i}: {meta.get('video_title', 'Unknown')} ({meta.get('chunk_type', 'unknown')})]"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


@dataclass
class RAGResponse:
    """Response from the RAG chain."""

    answer: str
    source_documents: list[Document] = field(default_factory=list)
    mode: str = "strategy"


class RAGChain:
    """End-to-end RAG pipeline: retrieve context -> generate response."""

    def __init__(self, config: Config) -> None:
        self._config = config

        # Initialize components
        logger.info("Initializing RAG chain...")
        self._embeddings = get_embeddings(config)
        self._llm = get_llm(config)
        self._vector_store = VectorStore(config, self._embeddings)
        logger.info("RAG chain ready (%d documents in store)", self._vector_store.document_count)

    @property
    def document_count(self) -> int:
        """Number of documents in the vector store."""
        return self._vector_store.document_count

    @property
    def vector_store(self) -> VectorStore:
        """Access the underlying vector store."""
        return self._vector_store

    def query(
        self,
        question: str,
        mode: str = "strategy",
        channel_ids: list[str] | None = None,
    ) -> RAGResponse:
        """Run a RAG query.

        Args:
            question: User's question or request.
            mode: Prompt mode - "strategy", "explain", or "review".
            channel_ids: Optional list of channel slugs to filter retrieval.

        Returns:
            RAGResponse with the answer and source documents.
        """
        if self._vector_store.document_count == 0:
            return RAGResponse(
                answer=(
                    "No documents in the vector store. "
                    "Run `codetrading-rag ingest` first to fetch and index videos."
                ),
                source_documents=[],
                mode=mode,
            )

        prompt = get_prompt(mode)
        retriever = self._vector_store.as_retriever(channel_ids=channel_ids)

        # Retrieve documents first so we can return them alongside the answer
        logger.info("Running RAG query (mode=%s): %s", mode, question[:80])
        retrieved_docs = retriever.invoke(question)

        # Build LCEL chain: format context -> prompt -> LLM -> parse output
        chain = prompt | self._llm | StrOutputParser()

        answer = chain.invoke({
            "context": _format_docs(retrieved_docs),
            "input": question,
        })

        return RAGResponse(
            answer=answer,
            source_documents=retrieved_docs,
            mode=mode,
        )
