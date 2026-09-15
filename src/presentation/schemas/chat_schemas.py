"""
Presentation schemas — Pydantic request/response models for the API layer.
Separate from Application DTOs to allow independent versioning.
"""
from pydantic import BaseModel, Field


class InitSimRequest(BaseModel):
    """Request body for POST /api/v1/simulations."""
    persona_id: str = Field(
        ...,
        description="Persona ID to initialize (e.g., 'gucci_ceo', 'gucci_chro').",
        examples=["gucci_chro"],
    )


class InitSimResponse(BaseModel):
    """Response for POST /api/v1/simulations."""
    session_id: str
    persona_id: str
    persona_name: str
    persona_role: str
