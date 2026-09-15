"""
SupervisorService — the "Director" layer that monitors conversation pacing.

Invisible to the user. Detects when a learner is stuck or going in circles,
then generates a subtle in-character hint to guide them back on track.

Single Responsibility: only cares about progression monitoring and hint generation.
"""
import logging
from difflib import SequenceMatcher
from typing import List, Optional

from src.domain.entities.conversation import Conversation
from src.domain.ports.llm_port import LLMPort

logger = logging.getLogger(__name__)

_LOOP_WINDOW = 4           # Number of recent messages to compare
_LOOP_THRESHOLD = 0.72     # Similarity ratio to flag as a loop
_MIN_TURNS_BEFORE_CHECK = 2


class SupervisorService:
    """
    Monitors conversation health and generates director hints.

    Injected with LLMPort to generate natural, in-character hints
    rather than scripted hard-coded messages.
    """

    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm

    async def evaluate(
        self,
        conversation: Conversation,
        simulation_goal: str,
    ) -> Optional[str]:
        """
        Evaluate conversation progress and return a hint if needed.

        Args:
            conversation:    Current conversation state.
            simulation_goal: The learning objective of the simulation.

        Returns:
            A hint string for the NPC to weave in, or None if no intervention needed.
        """
        if conversation.turn_count < _MIN_TURNS_BEFORE_CHECK:
            return None

        if self._is_stuck(conversation):
            conversation.is_stuck = True
            logger.info(
                "Session '%s' detected as stuck at turn %d",
                conversation.session_id,
                conversation.turn_count,
            )
            return await self._generate_hint(conversation, simulation_goal)

        conversation.is_stuck = False
        return None

    # ─── Private ─────────────────────────────────────────────────────────────

    def _is_stuck(self, conversation: Conversation) -> bool:
        """
        Detect if user messages are repeating (circular conversation).
        Uses SequenceMatcher for fuzzy similarity — handles paraphrasing.
        """
        recent = conversation.get_recent_user_messages(_LOOP_WINDOW)
        if len(recent) < 2:
            return False

        # Compare the latest message against all previous in the window
        latest = recent[-1]
        for prior in recent[:-1]:
            ratio = SequenceMatcher(None, latest.lower(), prior.lower()).ratio()
            if ratio >= _LOOP_THRESHOLD:
                logger.debug("Loop detected: similarity=%.2f", ratio)
                return True
        return False

    async def _generate_hint(
        self,
        conversation: Conversation,
        simulation_goal: str,
    ) -> str:
        """
        Use LLM to generate a subtle, in-character hint.
        The hint is designed to be woven naturally into the NPC's next response.
        """
        system_prompt = (
            "You are the Director of a professional simulation. "
            "The learner appears to be stuck. Generate a SHORT, SUBTLE hint "
            "(1-2 sentences max) that can be woven into the NPC's next response. "
            "The hint should guide the learner without being obvious. "
            "Do NOT break the fourth wall. Stay in the simulation context."
        )
        context = (
            f"Simulation goal: {simulation_goal}\n"
            f"Turns completed: {conversation.turn_count}\n"
            f"Last user messages: {conversation.get_recent_user_messages(2)}"
        )
        hint = await self._llm.chat(
            system_prompt=system_prompt,
            history=[],
            user_message=context,
        )
        logger.info("Supervisor generated hint: %s", hint[:100])
        return hint

    @staticmethod
    def build_stuck_detection_summary(messages: List[str]) -> str:
        """Utility for logging/debugging — formats recent messages for inspection."""
        return "\n".join(f"  [{i+1}] {m[:80]}" for i, m in enumerate(messages))
