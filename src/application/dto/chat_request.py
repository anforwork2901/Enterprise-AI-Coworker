"""
DTOs for the chat use case — defines the contract between Presentation and Application layers.
"""
from pydantic import BaseModel, Field
from typing import List


class ChatRequest(BaseModel):
    """Input DTO for ChatWithNPC use case."""
    session_id: str = Field(..., description="Unique session identifier.")
    persona_id: str = Field(..., description="NPC persona to chat with.")
    user_message: str = Field(..., min_length=1, max_length=2000)
    simulation_goal: str = Field(
        default="Practice talent and leadership development strategies.",
        description="The learning objective used by the Supervisor.",
    )


class ChatResponse(BaseModel):
    """Output DTO returned to the Presentation layer."""
    session_id: str
    assistant_message: str
    emotion_state: str
    safety_level: str
    safety_flags: List[str]
    hint_injected: bool
    turn_count: int
