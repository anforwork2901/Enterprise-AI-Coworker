"""
NPC Conversation Graph — LangGraph StateGraph powering the NPC orchestration.

This is the heart of the AI Co-Worker Engine. Each node in the graph represents
a discrete processing step, and conditional edges determine the flow:

    ┌────────────┐
    │ input_guard │──── BLOCKED ──▶ [blocked_response] ──▶ END
    └─────┬──────┘
          │ SAFE / WARNING
          ▼
    ┌─────────────┐
    │ load_memory  │
    └─────┬───────┘
          ▼
    ┌───────────────┐
    │ retrieve_context│
    └─────┬─────────┘
          ▼
    ┌───────────────┐
    │ supervisor_check│─── HINT ──▶ (hint stored in state)
    └─────┬─────────┘
          │
          ▼
    ┌──────────────────┐
    │ generate_response │
    └─────┬────────────┘
          ▼
    ┌──────────────┐
    │ output_guard  │
    └─────┬────────┘
          ▼
    ┌────────────┐
    │ save_state  │──▶ END
    └────────────┘

Clean Architecture: LangGraph is CONFINED to this file.
Domain services are called via injected references — never imported directly by the graph.
"""
import logging
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from src.domain.entities.conversation import Conversation
from src.domain.entities.message import Message
from src.domain.entities.npc_response import NPCResponse
from src.domain.entities.persona import Persona
from src.domain.enums import EmotionState, SafetyLevel
from src.domain.ports.memory_port import MemoryPort
from src.domain.ports.vector_store_port import VectorStorePort
from src.domain.services.npc_agent import NPCAgent
from src.domain.services.persona_loader import PersonaLoader
from src.domain.services.safety_guard import SafetyGuard
from src.domain.services.supervisor import SupervisorService

logger = logging.getLogger(__name__)


# ─── Graph State Schema ──────────────────────────────────────────────────────
# TypedDict defines the shared state flowing through all nodes.
# Each node can read and write to this state dict.

class NPCGraphState(TypedDict, total=False):
    """
    Shared state passed between LangGraph nodes.

    Inputs (set before graph invocation):
        session_id, persona_id, user_message, simulation_goal

    Intermediate (set by nodes during execution):
        persona, conversation, rag_context, hint,
        npc_response, safety_level, safety_flags

    Output (read after graph completes):
        assistant_message, emotion_state, safety_level, safety_flags,
        hint_injected, turn_count
    """
    # ── Inputs ────────────────────────────────────────────────────────────────
    session_id: str
    persona_id: str
    user_message: str
    simulation_goal: str

    # ── Intermediate ──────────────────────────────────────────────────────────
    persona: Optional[Persona]
    conversation: Optional[Conversation]
    rag_context: List[str]
    hint: Optional[str]
    npc_response: Optional[NPCResponse]

    # ── Output ────────────────────────────────────────────────────────────────
    assistant_message: str
    emotion_state: str
    safety_level: str
    safety_flags: List[str]
    hint_injected: bool
    turn_count: int


# ─── Graph Builder ────────────────────────────────────────────────────────────

class NPCGraphBuilder:
    """
    Builds and compiles the LangGraph StateGraph for NPC conversations.

    All domain services are injected via constructor — the graph itself
    has no knowledge of how to create them (Dependency Inversion Principle).
    """

    def __init__(
        self,
        npc_agent: NPCAgent,
        supervisor: SupervisorService,
        safety_guard: SafetyGuard,
        memory: MemoryPort,
        vector_store: VectorStorePort,
        persona_loader: PersonaLoader,
    ) -> None:
        self._npc_agent = npc_agent
        self._supervisor = supervisor
        self._safety_guard = safety_guard
        self._memory = memory
        self._vector_store = vector_store
        self._persona_loader = persona_loader

    def build(self):
        """
        Construct the StateGraph with all nodes and edges.
        Returns a compiled graph ready for invocation.
        """
        graph = StateGraph(NPCGraphState)

        # ── Register Nodes ────────────────────────────────────────────────────
        graph.add_node("input_guard", self._node_input_guard)
        graph.add_node("blocked_response", self._node_blocked_response)
        graph.add_node("load_memory", self._node_load_memory)
        graph.add_node("retrieve_context", self._node_retrieve_context)
        graph.add_node("supervisor_check", self._node_supervisor_check)
        graph.add_node("generate_response", self._node_generate_response)
        graph.add_node("output_guard", self._node_output_guard)
        graph.add_node("save_state", self._node_save_state)

        # ── Set Entry Point ───────────────────────────────────────────────────
        graph.set_entry_point("input_guard")

        # ── Conditional Edge: input_guard → blocked_response OR load_memory ──
        graph.add_conditional_edges(
            "input_guard",
            self._route_after_safety,
            {
                "blocked": "blocked_response",
                "continue": "load_memory",
            },
        )

        # ── blocked_response → END ───────────────────────────────────────────
        graph.add_edge("blocked_response", END)

        # ── Linear flow: load → retrieve → supervisor → generate → output → save
        graph.add_edge("load_memory", "retrieve_context")
        graph.add_edge("retrieve_context", "supervisor_check")
        graph.add_edge("supervisor_check", "generate_response")
        graph.add_edge("generate_response", "output_guard")
        graph.add_edge("output_guard", "save_state")
        graph.add_edge("save_state", END)

        return graph.compile()

    # ─── Node Implementations ─────────────────────────────────────────────────
    # Each node is an async function: (state) -> partial state update dict

    async def _node_input_guard(self, state: NPCGraphState) -> Dict[str, Any]:
        """Node 1: Check user input for safety issues."""
        safety_level, safety_flags = self._safety_guard.check_input(state["user_message"])
        persona = self._persona_loader.load(state["persona_id"])

        logger.debug("input_guard: level=%s, flags=%s", safety_level, safety_flags)
        return {
            "safety_level": safety_level.value,
            "safety_flags": safety_flags,
            "persona": persona,
        }

    async def _node_blocked_response(self, state: NPCGraphState) -> Dict[str, Any]:
        """Node: Generate in-character refusal for blocked input."""
        persona = state["persona"]
        blocked_msg = self._safety_guard.get_blocked_response(persona.name)
        logger.warning("Blocked message in session '%s'", state["session_id"])
        return {
            "assistant_message": blocked_msg,
            "emotion_state": EmotionState.GUARDED.value,
            "hint_injected": False,
            "turn_count": 0,
        }

    async def _node_load_memory(self, state: NPCGraphState) -> Dict[str, Any]:
        """Node 2: Load or create conversation from memory store."""
        conversation = await self._memory.load(state["session_id"])
        if conversation is None:
            conversation = Conversation(
                session_id=state["session_id"],
                persona_id=state["persona_id"],
            )
            logger.info(
                "New conversation: session='%s', persona='%s'",
                state["session_id"], state["persona_id"],
            )
        return {"conversation": conversation}

    async def _node_retrieve_context(self, state: NPCGraphState) -> Dict[str, Any]:
        """Node 3: RAG — retrieve relevant knowledge chunks."""
        rag_context = await self._vector_store.search(
            query=state["user_message"],
            persona_id=state["persona_id"],
        )
        logger.debug("RAG retrieved %d chunks", len(rag_context))
        return {"rag_context": rag_context}

    async def _node_supervisor_check(self, state: NPCGraphState) -> Dict[str, Any]:
        """Node 4: Supervisor (Director) — detect stuck loops, generate hints."""
        conversation: Conversation = state["conversation"]

        # Add user message so Supervisor has full context
        user_msg = Message(role="user", content=state["user_message"])
        conversation.add_message(user_msg)

        hint = await self._supervisor.evaluate(
            conversation=conversation,
            simulation_goal=state.get("simulation_goal", ""),
        )

        if hint:
            logger.info("Supervisor injecting hint for session '%s'", state["session_id"])

        return {"hint": hint, "conversation": conversation}

    async def _node_generate_response(self, state: NPCGraphState) -> Dict[str, Any]:
        """Node 5: NPC Agent — generate persona-consistent response."""
        npc_response = await self._npc_agent.generate_response(
            persona=state["persona"],
            conversation=state["conversation"],
            user_message=state["user_message"],
            rag_context=state.get("rag_context", []),
            hint=state.get("hint"),
        )
        return {
            "npc_response": npc_response,
            "assistant_message": npc_response.assistant_message,
            "emotion_state": npc_response.state_update.value,
            "hint_injected": npc_response.hint_injected,
        }

    async def _node_output_guard(self, state: NPCGraphState) -> Dict[str, Any]:
        """Node 6: Check NPC output for safety issues."""
        out_level, out_flags = self._safety_guard.check_output(state["assistant_message"])
        safety_flags = list(state.get("safety_flags", []))
        if out_level != SafetyLevel.SAFE:
            safety_flags.extend(out_flags)
        return {"safety_flags": safety_flags}

    async def _node_save_state(self, state: NPCGraphState) -> Dict[str, Any]:
        """Node 7: Persist updated conversation state to memory."""
        conversation: Conversation = state["conversation"]
        npc_response: NPCResponse = state["npc_response"]

        assistant_msg = Message(
            role="assistant",
            content=npc_response.assistant_message,
            emotion_snapshot=npc_response.state_update,
        )
        conversation.add_message(assistant_msg)
        conversation.update_emotion(npc_response.state_update)
        await self._memory.save(conversation)

        logger.info(
            "Turn %d | session='%s' | emotion=%s | hint=%s",
            conversation.turn_count, state["session_id"],
            npc_response.state_update.value, state.get("hint_injected", False),
        )
        return {"turn_count": conversation.turn_count}

    # ─── Routing Functions ────────────────────────────────────────────────────

    @staticmethod
    def _route_after_safety(state: NPCGraphState) -> str:
        """Conditional router: send blocked inputs to blocked_response, else continue."""
        if state.get("safety_level") == SafetyLevel.BLOCKED.value:
            return "blocked"
        return "continue"
