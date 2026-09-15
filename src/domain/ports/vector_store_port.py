"""
VectorStorePort — abstract contract for RAG knowledge retrieval.

Enables persona-scoped knowledge lookup without coupling domain
logic to any specific vector DB (FAISS, Pinecone, Milvus, etc.).
"""
from abc import ABC, abstractmethod
from typing import List


class VectorStorePort(ABC):
    """Abstract interface for semantic knowledge retrieval."""

    @abstractmethod
    async def search(
        self,
        query: str,
        persona_id: str,
        top_k: int = 4,
    ) -> List[str]:
        """
        Search the knowledge base for documents relevant to the query.

        Args:
            query:      The user message or search phrase.
            persona_id: Scopes search to that NPC's knowledge domain.
            top_k:      Maximum number of results to return.

        Returns:
            List of relevant text chunks (already decoded to strings).
        """
        ...

    @abstractmethod
    async def index_documents(
        self,
        documents: List[str],
        persona_id: str,
    ) -> None:
        """
        Add documents to the vector store under a persona namespace.

        Args:
            documents:  Raw text chunks to embed and store.
            persona_id: Namespace/tag for retrieval scoping.
        """
        ...
