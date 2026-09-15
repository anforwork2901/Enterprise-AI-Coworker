"""
InitializeSimulation use case — sets up a simulation session.

Loads persona, indexes knowledge base into FAISS, and creates
the initial Conversation state ready for chat.
"""
import logging
from pathlib import Path
from typing import List
from uuid import uuid4

from src.domain.entities.conversation import Conversation
from src.domain.ports.memory_port import MemoryPort
from src.domain.ports.vector_store_port import VectorStorePort
from src.domain.services.persona_loader import PersonaLoader
from src.infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)


class InitializeSimulation:
    """
    Creates a new simulation session and pre-indexes knowledge for the NPC.
    """

    def __init__(
        self,
        memory: MemoryPort,
        vector_store: VectorStorePort,
        persona_loader: PersonaLoader,
        settings: Settings,
    ) -> None:
        self._memory = memory
        self._vector_store = vector_store
        self._persona_loader = persona_loader
        self._kb_dir = Path(settings.knowledge_base_dir)

    async def execute(self, persona_id: str) -> dict:
        """
        Initialize a simulation for the given persona.

        Returns:
            dict with session_id and persona metadata.
        """
        # Validate persona exists
        persona = self._persona_loader.load(persona_id)

        # Create new session
        session_id = str(uuid4())
        conversation = Conversation(session_id=session_id, persona_id=persona_id)
        await self._memory.save(conversation)

        # Index knowledge base (if not already indexed from disk)
        if isinstance(self._vector_store, object) and hasattr(self._vector_store, "load_persisted_index"):
            loaded = self._vector_store.load_persisted_index(persona_id)
            if not loaded:
                await self._index_knowledge_base(persona_id)
        else:
            await self._index_knowledge_base(persona_id)

        logger.info("Simulation initialized: session='%s', persona='%s'", session_id, persona_id)

        return {
            "session_id": session_id,
            "persona_id": persona_id,
            "persona_name": persona.name,
            "persona_role": persona.role,
        }

    # ─── Private ─────────────────────────────────────────────────────────────

    async def _index_knowledge_base(self, persona_id: str) -> None:
        """Load markdown files from knowledge base and index per persona."""
        kb_path = self._kb_dir / "gucci_simulation"
        if not kb_path.exists():
            logger.warning("Knowledge base directory not found: %s", kb_path)
            return

        documents: List[str] = []
        for md_file in kb_path.glob("*.md"):
            text = md_file.read_text(encoding="utf-8")
            # Simple chunking: split by double newlines
            chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
            documents.extend(chunks)

        if documents:
            await self._vector_store.index_documents(documents, persona_id)
            logger.info("Indexed %d chunks for persona '%s'", len(documents), persona_id)
