"""
Dependency Injection container — wires all concrete implementations to their ports.

This is the ONLY place in the project where:
1. Concrete infrastructure classes are instantiated
2. Settings are read and passed to adapters
3. The dependency graph is assembled
4. LangGraph is compiled

FastAPI's Depends() system is used for clean, testable DI.
"""
from functools import lru_cache
from typing import Any, List

from src.domain.ports.llm_port import LLMPort
from src.domain.ports.memory_port import MemoryPort
from src.domain.ports.tool_port import ToolPort
from src.domain.ports.vector_store_port import VectorStorePort
from src.domain.services.npc_agent import NPCAgent
from src.domain.services.persona_loader import PersonaLoader
from src.domain.services.safety_guard import SafetyGuard
from src.domain.services.supervisor import SupervisorService
from src.infrastructure.config.settings import Settings, get_settings
from src.infrastructure.graphs.npc_graph import NPCGraphBuilder
from src.infrastructure.llm.gemini_adapter import GeminiAdapter
from src.infrastructure.llm.openai_adapter import OpenAIAdapter
from src.infrastructure.memory.in_memory_store import InMemoryStore
from src.infrastructure.tools.jira_mock import JIRAMock
from src.infrastructure.tools.kpi_calculator import KPICalculator
from src.infrastructure.vector_store.faiss_adapter import FAISSAdapter
from src.infrastructure.vector_store.gemini_faiss_adapter import GeminiFAISSAdapter


# ─── Infrastructure Singletons ────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_llm_adapter() -> LLMPort:
    settings: Settings = get_settings()
    if settings.active_provider == "openai":
        return OpenAIAdapter(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens,
        )
    return GeminiAdapter(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        temperature=settings.openai_temperature,
        max_tokens=settings.openai_max_tokens,
    )


@lru_cache(maxsize=1)
def get_memory_store() -> MemoryPort:
    return InMemoryStore()


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStorePort:
    settings: Settings = get_settings()
    if settings.active_provider == "openai":
        return FAISSAdapter(
            api_key=settings.openai_api_key,
            embedding_model=settings.openai_embedding_model,
            index_path=settings.faiss_index_path,
        )
    return GeminiFAISSAdapter(
        api_key=settings.gemini_api_key,
        embedding_model=settings.embedding_model,
        index_path=settings.faiss_index_path,
    )


@lru_cache(maxsize=1)
def get_tools() -> List[ToolPort]:
    return [KPICalculator(), JIRAMock()]


# ─── Domain Services ──────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_persona_loader() -> PersonaLoader:
    settings: Settings = get_settings()
    return PersonaLoader(personas_dir=settings.personas_dir)


@lru_cache(maxsize=1)
def get_safety_guard() -> SafetyGuard:
    return SafetyGuard()


@lru_cache(maxsize=1)
def get_npc_agent() -> NPCAgent:
    return NPCAgent(
        llm=get_llm_adapter(),
        vector_store=get_vector_store(),
        tools=get_tools(),
    )


@lru_cache(maxsize=1)
def get_supervisor() -> SupervisorService:
    return SupervisorService(llm=get_llm_adapter())


# ─── LangGraph ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_compiled_graph() -> Any:
    """
    Build and compile the LangGraph NPC conversation graph.
    Returns a compiled graph ready for ainvoke().
    """
    builder = NPCGraphBuilder(
        npc_agent=get_npc_agent(),
        supervisor=get_supervisor(),
        safety_guard=get_safety_guard(),
        memory=get_memory_store(),
        vector_store=get_vector_store(),
        persona_loader=get_persona_loader(),
    )
    return builder.build()
