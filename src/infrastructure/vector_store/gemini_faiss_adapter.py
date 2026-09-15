"""
GeminiFAISSAdapter — LangChain FAISS implementation of VectorStorePort for Google Gemini.

Provides persona-scoped knowledge retrieval using Gemini embeddings.
"""
import logging
import os
from typing import Dict, List

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.domain.ports.vector_store_port import VectorStorePort

logger = logging.getLogger(__name__)


class GeminiFAISSAdapter(VectorStorePort):
    """
    Concrete VectorStorePort using LangChain's FAISS integration and Gemini Embeddings.
    Maintains a separate FAISS index per persona_id for scoped retrieval.
    """

    def __init__(self, api_key: str, embedding_model: str, index_path: str) -> None:
        if not api_key:
            raise ValueError("Gemini API Key is missing. Please set it in .env")
            
        self._embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=api_key, 
            model=embedding_model
        )
        self._index_path = index_path
        self._indexes: Dict[str, FAISS] = {}
        os.makedirs(index_path, exist_ok=True)
        logger.info("GeminiFAISSAdapter initialized at '%s' with model '%s'", index_path, embedding_model)

    async def search(
        self,
        query: str,
        persona_id: str,
        top_k: int = 4,
    ) -> List[str]:
        """Retrieve top-k relevant documents for the query in persona's knowledge scope."""
        index = self._indexes.get(persona_id)
        if index is None:
            logger.warning("No FAISS index found for persona '%s'. Returning empty context.", persona_id)
            return []

        docs = index.similarity_search(query, k=top_k)
        results = [doc.page_content for doc in docs]
        logger.debug("FAISS search for '%s' (persona=%s) → %d docs", query[:50], persona_id, len(results))
        return results

    async def index_documents(
        self,
        documents: List[str],
        persona_id: str,
    ) -> None:
        """Embed and store documents under the persona's FAISS index."""
        if not documents:
            logger.warning("No documents to index for persona '%s'", persona_id)
            return

        index = FAISS.from_texts(texts=documents, embedding=self._embeddings)
        self._indexes[persona_id] = index

        # Persist to disk for reuse across restarts
        save_path = os.path.join(self._index_path, persona_id)
        index.save_local(save_path)
        logger.info("Indexed %d documents for persona '%s' → saved to '%s'", len(documents), persona_id, save_path)

    def load_persisted_index(self, persona_id: str) -> bool:
        """
        Load a previously saved FAISS index from disk.
        Returns True if successfully loaded, False if not found.
        """
        load_path = os.path.join(self._index_path, persona_id)
        if not os.path.exists(load_path):
            return False
        self._indexes[persona_id] = FAISS.load_local(
            folder_path=load_path,
            embeddings=self._embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("Loaded persisted FAISS index for persona '%s'", persona_id)
        return True
