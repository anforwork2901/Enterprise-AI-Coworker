"""
Message value object — a single turn in a conversation.
Immutable once created. Carries emotion snapshot for state tracking.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from src.domain.enums import EmotionState


@dataclass(frozen=True)
class Message:
    """
    Represents a single message in a conversation turn.

    Attributes:
        role:             "user" | "assistant" | "system"
        content:          Text content of the message.
        timestamp:        UTC creation time (auto-set).
        emotion_snapshot: The NPC's emotion state at the time of this message.
    """
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    emotion_snapshot: EmotionState = EmotionState.NEUTRAL
