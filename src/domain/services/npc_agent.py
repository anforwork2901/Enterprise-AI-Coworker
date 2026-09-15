"""
NPCAgent — core domain service that assembles NPC responses.

Responsibilities:
- Build persona-aware system prompts
- Integrate RAG context into prompts
- Drive emotion state transitions
- Coordinate with LLMPort to generate responses

This service is PURE DOMAIN — no LangChain imports allowed here.
LLM calls are delegated to the injected LLMPort.
"""
import logging
from typing import Dict, List, Optional

from src.domain.entities.conversation import Conversation
from src.domain.entities.npc_response import NPCResponse
from src.domain.entities.persona import Persona
from src.domain.enums import EmotionState, SafetyLevel
from src.domain.ports.llm_port import LLMPort
from src.domain.ports.tool_port import ToolPort
from src.domain.ports.vector_store_port import VectorStorePort

logger = logging.getLogger(__name__)

# Emotion transition rules: (current_state, trigger) -> new_state
_EMOTION_TRANSITIONS: Dict[tuple, EmotionState] = {
    (EmotionState.NEUTRAL, "good"):        EmotionState.ENGAGED,
    (EmotionState.NEUTRAL, "bad"):         EmotionState.IMPATIENT,
    (EmotionState.ENGAGED, "good"):        EmotionState.PLEASED,
    (EmotionState.ENGAGED, "bad"):         EmotionState.IMPATIENT,
    (EmotionState.PLEASED, "bad"):         EmotionState.NEUTRAL,
    (EmotionState.IMPATIENT, "bad"):       EmotionState.FRUSTRATED,
    (EmotionState.IMPATIENT, "good"):      EmotionState.NEUTRAL,
    (EmotionState.GUARDED, "good"):        EmotionState.IMPATIENT,
    (EmotionState.GUARDED, "bad"):         EmotionState.FRUSTRATED,
    (EmotionState.FRUSTRATED, "good"):     EmotionState.IMPATIENT,
}


class NPCAgent:
    """
    Core NPC response engine.
    Injected with LLMPort, VectorStorePort, and ToolPort implementations
    (Dependency Inversion Principle — depends on abstractions, not concretions).
    """

    def __init__(
        self,
        llm: LLMPort,
        vector_store: VectorStorePort,
        tools: Optional[List[ToolPort]] = None,
    ) -> None:
        self._llm = llm
        self._vector_store = vector_store
        self._tools: Dict[str, ToolPort] = {t.tool_id: t for t in (tools or [])}

    async def generate_response(
        self,
        persona: Persona,
        conversation: Conversation,
        user_message: str,
        rag_context: List[str],
        hint: Optional[str] = None,
    ) -> NPCResponse:
        """
        Generate an NPC response for the current user turn.

        Args:
            persona:      The active NPC's identity.
            conversation: Current conversation state (history + emotion).
            user_message: The user's latest input.
            rag_context:  Relevant knowledge chunks retrieved via RAG.
            hint:         Optional Supervisor hint to weave into response.

        Returns:
            NPCResponse with message, updated state, and safety flags.
        """
        system_prompt = self._build_system_prompt(persona, conversation, rag_context, hint)
        history = conversation.get_history_as_tuples()

        # ── Tool routing ──────────────────────────────────────────────────────
        allowed_tools = [
            self._tools[tid].get_schema()
            for tid in persona.tools_allowed
            if tid in self._tools
        ]

        if allowed_tools:
            raw = await self._llm.chat_with_tools(
                system_prompt=system_prompt,
                history=history,
                user_message=user_message,
                tools=allowed_tools,
            )
            tool_results = await self._execute_tool_calls(raw.get("tool_calls") or [])
            assistant_message = raw["content"] or self._format_tool_summary(tool_results)
        else:
            assistant_message = await self._llm.chat(
                system_prompt=system_prompt,
                history=history,
                user_message=user_message,
            )
            tool_results = {}

        # ── Emotion state transition ──────────────────────────────────────────
        quality = self._assess_interaction_quality(user_message)
        new_emotion = self._transition_emotion(conversation.emotion_state, quality)

        logger.debug(
            "NPC '%s' emotion: %s → %s (quality=%s)",
            persona.persona_id, conversation.emotion_state, new_emotion, quality,
        )

        return NPCResponse(
            assistant_message=assistant_message,
            state_update=new_emotion,
            safety_flags=[],
            safety_level=SafetyLevel.SAFE,
            tool_results=tool_results,
            hint_injected=hint is not None,
        )

    # ─── Private ─────────────────────────────────────────────────────────────

    @staticmethod
    def _build_system_prompt(
        persona: Persona,
        conversation: Conversation,
        rag_context: List[str],
        hint: Optional[str],
    ) -> str:
        """Assemble the full system prompt with persona identity + context + emotion."""
        context_block = "\n\n".join(rag_context) if rag_context else "No additional context."
        emotion_instruction = _EMOTION_INSTRUCTIONS.get(
            conversation.emotion_state,
            "Respond professionally.",
        )
        hint_block = f"\n\n[DIRECTOR NOTE — subtly guide user]: {hint}" if hint else ""

        return (
            f"{persona.system_prompt}\n\n"
            f"--- RELEVANT KNOWLEDGE ---\n{context_block}\n"
            f"--- YOUR CURRENT MOOD ---\n{emotion_instruction}"
            f"{hint_block}"
        )

    @staticmethod
    def _assess_interaction_quality(user_message: str) -> str:
        """
        Heuristic to classify interaction quality as 'good' or 'bad'.
        Used to drive emotion state transitions.
        In production, this could use a lightweight classifier.
        """
        BAD_SIGNALS = ["shut up", "stupid", "idiot", "useless", "i don't care", "whatever"]
        lower = user_message.lower()
        if any(signal in lower for signal in BAD_SIGNALS):
            return "bad"
        return "good"

    @staticmethod
    def _transition_emotion(
        current: EmotionState,
        quality: str,
    ) -> EmotionState:
        """Apply the emotion transition table."""
        return _EMOTION_TRANSITIONS.get((current, quality), current)

    async def _execute_tool_calls(
        self, tool_calls: list
    ) -> Dict[str, object]:
        """Execute any tool calls returned by the LLM."""
        results: Dict[str, object] = {}
        for call in tool_calls:
            tool_id = call.get("name")
            params = call.get("arguments", {})
            if tool_id in self._tools:
                results[tool_id] = await self._tools[tool_id].execute(params)
                logger.info("Tool '%s' executed with params: %s", tool_id, params)
        return results

    @staticmethod
    def _format_tool_summary(tool_results: Dict[str, object]) -> str:
        """Format tool results into a human-readable summary."""
        if not tool_results:
            return "I've processed your request."
        lines = [f"— {tid}: {result}" for tid, result in tool_results.items()]
        return "Here are the results:\n" + "\n".join(lines)


# ─── Emotion Instructions (prompt tone modifiers) ─────────────────────────────
_EMOTION_INSTRUCTIONS: Dict[EmotionState, str] = {
    EmotionState.NEUTRAL:    "Be professional and measured in your response.",
    EmotionState.ENGAGED:    "You are energized by this collaboration — show genuine enthusiasm.",
    EmotionState.PLEASED:    "You are impressed. Be warm and encouraging, but stay business-focused.",
    EmotionState.IMPATIENT:  "You are running short on patience. Be concise and slightly curt.",
    EmotionState.GUARDED:    "You are wary of this person. Be correct but noticeably reserved.",
    EmotionState.FRUSTRATED: "You are clearly frustrated. Keep it professional, but your tone is cool and clipped.",
}
