"""
Health check router.
"""
from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health", summary="Health check")
async def health() -> dict:
    return {"status": "ok", "service": "AI Co-Worker NPC Engine"}
