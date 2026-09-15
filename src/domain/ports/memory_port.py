"""
MemoryPort — abstract contract for conversation state persistence.

Supports swap between InMemoryStore (dev) and Redis (prod)
without touching any domain or application code.
"""
from abc import ABC, abstractmethod
from typing import Optional

from src.domain.entities.conversation import Conversation


class MemoryPort(ABC):
    """Abstract interface for storing and retrieving conversation state."""

    @abstractmethod
    async def save(self, conversation: Conversation) -> None:
        """Persist a conversation (full state, not just latest message)."""
        ...

    @abstractmethod
    async def load(self, session_id: str) -> Optional[Conversation]:
        """
        Retrieve a conversation by session ID.
        Returns None if session does not exist.
        """
        ...

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        """Remove a conversation session (cleanup after simulation ends)."""
        ...
