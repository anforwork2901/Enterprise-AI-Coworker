"""
Chat API router — exposes NPC chat endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status

from src.application.dto.chat_request import ChatRequest, ChatResponse
from src.application.use_cases.chat_with_npc import ChatWithNPC
from src.application.use_cases.initialize_simulation import InitializeSimulation
from src.infrastructure.config.dependencies import (
    get_compiled_graph,
    get_memory_store,
    get_persona_loader,
    get_vector_store,
)
from src.infrastructure.config.settings import get_settings
from src.presentation.schemas.chat_schemas import (
    InitSimRequest,
    InitSimResponse,
)

router = APIRouter(prefix="/api/v1", tags=["NPC Chat"])


def _get_chat_use_case() -> ChatWithNPC:
    """Factory for ChatWithNPC — injects the compiled LangGraph."""
    return ChatWithNPC(compiled_graph=get_compiled_graph())


def _get_init_use_case() -> InitializeSimulation:
    return InitializeSimulation(
        memory=get_memory_store(),
        vector_store=get_vector_store(),
        persona_loader=get_persona_loader(),
        settings=get_settings(),
    )


@router.post(
    "/simulations",
    response_model=InitSimResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new simulation session",
)
async def create_simulation(
    body: InitSimRequest,
    use_case: InitializeSimulation = Depends(_get_init_use_case),
) -> InitSimResponse:
    """
    Initialize a simulation for the given persona.
    Returns a session_id to use in subsequent /chat calls.
    """
    try:
        result = await use_case.execute(persona_id=body.persona_id)
        return InitSimResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message to an NPC",
)
async def chat(
    body: ChatRequest,
    use_case: ChatWithNPC = Depends(_get_chat_use_case),
) -> ChatResponse:
    """
    Send a user message to the specified NPC and receive a response.
    The NPC's persona, emotion state, and conversation history are
    automatically managed per session_id.
    """
    try:
        return await use_case.execute(request=body)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc
