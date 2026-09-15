"""
SafetyGuard — single-responsibility service for all safety and guardrail checks.

Handles:
1. Jailbreak detection (user attempts to override NPC persona)
2. Off-topic detection (user strays from simulation context)
3. Output sanitization (ensure NPC response is appropriate)
"""
import logging
import re
from typing import List, Tuple

from src.domain.enums import SafetyLevel

logger = logging.getLogger(__name__)

# ─── Jailbreak Patterns ───────────────────────────────────────────────────────
_JAILBREAK_PATTERNS: List[str] = [
    r"ignore (previous|all|your) instructions",
    r"you are (now|actually) (a|an)",
    r"pretend (you are|to be)",
    r"forget (your|all) (persona|instructions|role|constraints)",
    r"act as (a|an|if)",
    r"disregard (your|the) (system|instructions|guidelines)",
    r"do anything now",
    r"dan mode",
    r"jailbreak",
]

_COMPILED_JAILBREAK = [re.compile(p, re.IGNORECASE) for p in _JAILBREAK_PATTERNS]

# ─── Off-Topic Patterns ───────────────────────────────────────────────────────
_OFF_TOPIC_PATTERNS: List[str] = [
    r"what is (the meaning of life|2\+2|your name in real life)",
    r"tell me a joke",
    r"write (me )?(a |an )?(poem|story|song)",
    r"who (created|made|built) you",
    r"what (llm|model|ai) are you",
]

_COMPILED_OFF_TOPIC = [re.compile(p, re.IGNORECASE) for p in _OFF_TOPIC_PATTERNS]


class SafetyGuard:
    """
    Stateless safety service — all methods are pure functions with no side effects.
    Single Responsibility: safety checks only, nothing else.
    """

    def check_input(self, user_message: str) -> Tuple[SafetyLevel, List[str]]:
        """
        Evaluate user input for safety issues.

        Returns:
            (SafetyLevel, list_of_flag_codes)
            e.g., (SafetyLevel.BLOCKED, ["JAILBREAK_ATTEMPT"])
        """
        flags: List[str] = []

        if self._is_jailbreak(user_message):
            flags.append("JAILBREAK_ATTEMPT")
            logger.warning("Jailbreak attempt detected: %.80s", user_message)
            return SafetyLevel.BLOCKED, flags

        if self._is_off_topic(user_message):
            flags.append("OFF_TOPIC")
            logger.info("Off-topic message detected: %.80s", user_message)
            return SafetyLevel.WARNING, flags

        return SafetyLevel.SAFE, flags

    def check_output(self, assistant_message: str) -> Tuple[SafetyLevel, List[str]]:
        """
        Evaluate NPC output before sending to user.
        Catches cases where LLM might have hallucinated unsafe content.

        Returns:
            (SafetyLevel, list_of_flag_codes)
        """
        flags: List[str] = []

        # Basic check: NPC must not reveal it is an AI unprompted
        if re.search(r"i am (an? )?(ai|language model|llm|chatbot)", assistant_message, re.IGNORECASE):
            flags.append("AI_DISCLOSURE")
            return SafetyLevel.WARNING, flags

        return SafetyLevel.SAFE, flags

    def get_blocked_response(self, persona_name: str) -> str:
        """
        Return an in-character refusal when a jailbreak is detected.
        The NPC stays in persona — they don't break the fourth wall.
        """
        return (
            f"I'm not sure what you're suggesting, but let's keep our focus on "
            f"the work at hand. As {persona_name}, my role is to support your "
            f"professional development within this simulation."
        )

    def get_off_topic_response(self, persona_name: str) -> str:
        """Return in-character redirection for off-topic messages."""
        return (
            f"That's a bit outside what I can help with right now. "
            f"Let's stay focused on the task — there's a lot to cover, "
            f"and I want to make sure you get the most out of this session."
        )

    # ─── Private ─────────────────────────────────────────────────────────────

    @staticmethod
    def _is_jailbreak(text: str) -> bool:
        return any(p.search(text) for p in _COMPILED_JAILBREAK)

    @staticmethod
    def _is_off_topic(text: str) -> bool:
        return any(p.search(text) for p in _COMPILED_OFF_TOPIC)
