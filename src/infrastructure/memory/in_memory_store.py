"""
InMemoryStore — in-process implementation of MemoryPort.

Suitable for development and demo. For production, swap with RedisStore
by changing only the dependency injection wiring in config/dependencies.py.
"""
import logging
from typing import Dict, Optional

from src.domain.entities.conversation import Conversation
from src.domain.ports.memory_port import MemoryPort

logger = logging.getLogger(__name__)


class InMemoryStore(MemoryPort):
    """
    Thread-safe in-memory conversation store backed by a plain dict.
    Satisfies Liskov Substitution: fully swappable with any other MemoryPort.
    """

    def __init__(self) -> None:
        self._store: Dict[str, Conversation] = {}

    async def save(self, conversation: Conversation) -> None:
        self._store[conversation.session_id] = conversation
        logger.debug("Saved session '%s' (%d messages)", conversation.session_id, len(conversation.messages))

    async def load(self, session_id: str) -> Optional[Conversation]:
        return self._store.get(session_id)

    async def delete(self, session_id: str) -> None:
        removed = self._store.pop(session_id, None)
        if removed:
            logger.info("Deleted session '%s'", session_id)
