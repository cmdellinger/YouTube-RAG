"""ChromaDB vector store wrapper with persistence and multi-channel support."""

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

    def add_documents(self, documents: list[Document], batch_size: int = 5000) -> None:
        """Add documents to the existing store in batches.

        Args:
            documents: List of LangChain Documents to embed and add.
            batch_size: Max documents per batch (ChromaDB limit is 5461).
        """
        if not documents:
            return
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            self._store.add_documents(batch)
            logger.info(
                "Added batch %d/%d (%d documents)",
                i // batch_size + 1,
                (len(documents) - 1) // batch_size + 1,
                len(batch),
            )
        logger.info("Added %d documents total to vector store", len(documents))

    def as_retriever(
        self,
        k: int | None = None,
        channel_ids: list[str] | None = None,
    ) -> VectorStoreRetriever:
        """Return a retriever interface for this store.

        Args:
            k: Number of documents to retrieve (default: config.retriever_k).
            channel_ids: Optional list of channel slugs to filter results.

        Returns:
            A LangChain VectorStoreRetriever.
        """
        search_k = k or self._config.retriever_k
        search_kwargs: dict = {"k": search_k}

        if channel_ids:
            if len(channel_ids) == 1:
                search_kwargs["filter"] = {"channel_id": channel_ids[0]}
            else:
                search_kwargs["filter"] = {"channel_id": {"$in": channel_ids}}

        return self._store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        channel_ids: list[str] | None = None,
    ) -> list[Document]:
        """Perform a similarity search.

        Args:
            query: Search query string.
            k: Number of results (default: config.retriever_k).
            channel_ids: Optional list of channel slugs to filter results.

        Returns:
            List of matching Documents.
        """
        search_k = k or self._config.retriever_k
        filter_dict = None

        if channel_ids:
            if len(channel_ids) == 1:
                filter_dict = {"channel_id": channel_ids[0]}
            else:
                filter_dict = {"channel_id": {"$in": channel_ids}}

        return self._store.similarity_search(query, k=search_k, filter=filter_dict)

    def delete_by_channel(self, channel_id: str) -> int:
        """Delete all documents for a specific channel.

        Args:
            channel_id: Channel slug to delete documents for.

        Returns:
            Number of documents deleted.
        """
        try:
            collection = self._store._collection
            results = collection.get(where={"channel_id": channel_id})
            ids = results.get("ids", [])
            if ids:
                collection.delete(ids=ids)
                logger.info("Deleted %d documents for channel '%s'", len(ids), channel_id)
            return len(ids)
        except Exception as exc:
            logger.warning("Error deleting documents for channel '%s': %s", channel_id, exc)
            return 0

    def get_channel_document_count(self, channel_id: str) -> int:
        """Count documents for a specific channel.

        Args:
            channel_id: Channel slug to count documents for.

        Returns:
            Number of documents for the channel.
        """
        try:
            collection = self._store._collection
            results = collection.get(where={"channel_id": channel_id})
            return len(results.get("ids", []))
        except Exception:
            return 0

    @property
    def document_count(self) -> int:
        """Return the number of documents in the store."""
        try:
            collection = self._store._collection
            return collection.count()
        except Exception:
            return 0
