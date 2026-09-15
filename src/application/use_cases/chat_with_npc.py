"""
ChatWithNPC — the primary application use case.

Orchestration is now DELEGATED to the LangGraph StateGraph
(src/infrastructure/graphs/npc_graph.py).

This use case has ONE job: translate DTOs ↔ Graph State.
It contains NO business logic and NO orchestration logic.
"""
import logging
from typing import Any, Dict

from src.application.dto.chat_request import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


class ChatWithNPC:
    """
    Application use case: handle one user turn in a simulation chat.

    The compiled LangGraph is injected — this class only translates between
    the Presentation layer's DTOs and the graph's TypedDict state.
    """

    def __init__(self, compiled_graph: Any) -> None:
        """
        Args:
            compiled_graph: A compiled LangGraph StateGraph returned by
                            NPCGraphBuilder.build().
        """
        self._graph = compiled_graph

    async def execute(self, request: ChatRequest) -> ChatResponse:
        """
        Execute one chat turn by invoking the LangGraph.

        Args:
            request: Validated ChatRequest DTO.

        Returns:
            ChatResponse DTO with NPC message and metadata.
        """
        # ── Prepare graph input state ─────────────────────────────────────────
        input_state: Dict[str, Any] = {
            "session_id": request.session_id,
            "persona_id": request.persona_id,
            "user_message": request.user_message,
            "simulation_goal": request.simulation_goal,
        }

        # ── Invoke LangGraph ──────────────────────────────────────────────────
        result = await self._graph.ainvoke(input_state)

        logger.info(
            "Graph completed | session='%s' | emotion=%s | turn=%s",
            result.get("session_id"),
            result.get("emotion_state"),
            result.get("turn_count"),
        )

        # ── Translate graph output → ChatResponse DTO ─────────────────────────
        return ChatResponse(
            session_id=request.session_id,
            assistant_message=result.get("assistant_message", ""),
            emotion_state=result.get("emotion_state", "neutral"),
            safety_level=result.get("safety_level", "safe"),
            safety_flags=result.get("safety_flags", []),
            hint_injected=result.get("hint_injected", False),
            turn_count=result.get("turn_count", 0),
        )
