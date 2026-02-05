"""ChromaDB vector store wrapper with persistence."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_chroma import Chroma
from langchain_core.documents import Document

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStoreRetriever

    from codetrading_rag.config import Config

logger = logging.getLogger(__name__)

COLLECTION_NAME = "codetrading_rag"


class VectorStore:
    """ChromaDB-backed vector store for CodeTrading-RAG documents."""

    def __init__(self, config: Config, embeddings: Embeddings) -> None:
        self._config = config
        self._embeddings = embeddings
        self._persist_dir = Path(config.chroma_db_path)
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        self._store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(self._persist_dir),
        )
        logger.info(
            "VectorStore initialized at %s (%d documents)",
            self._persist_dir,
            self.document_count,
        )

    def create_from_documents(self, documents: list[Document]) -> None:
        """Replace the store contents with a new set of documents.

        Args:
            documents: List of LangChain Documents to embed and store.
        """
        if not documents:
            logger.warning("No documents to store")
            return

        # Create a fresh store
        self._store = Chroma.from_documents(
            documents=documents,
            embedding=self._embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=str(self._persist_dir),
        )
        logger.info("Created vector store with %d documents", len(documents))

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the existing store.

        Args:
            documents: List of LangChain Documents to embed and add.
        """
        if not documents:
            return
        self._store.add_documents(documents)
        logger.info("Added %d documents to vector store", len(documents))

    def as_retriever(self, k: int | None = None) -> VectorStoreRetriever:
        """Return a retriever interface for this store.

        Args:
            k: Number of documents to retrieve (default: config.retriever_k).

        Returns:
            A LangChain VectorStoreRetriever.
        """
        search_k = k or self._config.retriever_k
        return self._store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": search_k},
        )

    def similarity_search(self, query: str, k: int | None = None) -> list[Document]:
        """Perform a similarity search.

        Args:
            query: Search query string.
            k: Number of results (default: config.retriever_k).

        Returns:
            List of matching Documents.
        """
        search_k = k or self._config.retriever_k
        return self._store.similarity_search(query, k=search_k)

    @property
    def document_count(self) -> int:
        """Return the number of documents in the store."""
        try:
            collection = self._store._collection
            return collection.count()
        except Exception:
            return 0
