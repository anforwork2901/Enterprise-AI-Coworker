"""
NPCResponse value object — the structured output of a single NPC interaction.

Returned by NPCAgent.generate_response() and passed up through the use case layer.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.domain.enums import EmotionState, SafetyLevel


@dataclass(frozen=True)
class NPCResponse:
    """
    Structured response from the NPC engine for a single user turn.

    Attributes:
        assistant_message:  The text the NPC says to the user.
        state_update:       New emotion state after processing this turn.
        safety_flags:       List of safety issue codes (empty = clean).
        safety_level:       Overall safety classification.
        tool_results:       Optional results from any tools the NPC invoked.
        hint_injected:      True if Supervisor injected a hint into this response.
    """
    assistant_message: str
    state_update: EmotionState = EmotionState.NEUTRAL
    safety_flags: List[str] = field(default_factory=list)
    safety_level: SafetyLevel = SafetyLevel.SAFE
    tool_results: Dict[str, Any] = field(default_factory=dict)
    hint_injected: bool = False
