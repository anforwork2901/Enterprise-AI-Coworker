"""
Conversation aggregate — owns the full chat history and NPC state for a session.

Responsible for:
- Appending messages immutably
- Tracking turn count and emotion progression
- Detecting loop/stuck conditions for the Supervisor
"""
from dataclasses import dataclass, field
from typing import List, Tuple

from src.domain.entities.message import Message
from src.domain.enums import EmotionState


# How many recent user messages to check for repetition
_LOOP_DETECTION_WINDOW = 4
_LOOP_SIMILARITY_THRESHOLD = 0.8  # fraction of overlapping tokens


@dataclass
class Conversation:
    """
    Mutable aggregate representing a live simulation session.

    Attributes:
        session_id:      Unique ID for this conversation session.
        persona_id:      Which NPC is active in this session.
        messages:        Ordered list of all messages.
        emotion_state:   Current NPC emotional disposition.
        turn_count:      Number of completed user-NPC turn pairs.
        is_stuck:        Flagged True when Supervisor detects a loop.
    """
    session_id: str
    persona_id: str
    messages: List[Message] = field(default_factory=list)
    emotion_state: EmotionState = EmotionState.NEUTRAL
    turn_count: int = 0
    is_stuck: bool = False

    def add_message(self, message: Message) -> None:
        """Append a message and increment turn counter on assistant replies."""
        self.messages.append(message)
        if message.role == "assistant":
            self.turn_count += 1

    def update_emotion(self, new_state: EmotionState) -> None:
        """Transition NPC emotional state."""
        self.emotion_state = new_state

    def get_recent_user_messages(self, n: int = _LOOP_DETECTION_WINDOW) -> List[str]:
        """Return contents of the last N user messages for loop detection."""
        user_msgs = [m.content for m in self.messages if m.role == "user"]
        return user_msgs[-n:]

    def get_history_as_tuples(self) -> List[Tuple[str, str]]:
        """
        Return conversation history as (role, content) tuples.
        Convenient for LLM prompt construction.
        """
        return [(m.role, m.content) for m in self.messages]
