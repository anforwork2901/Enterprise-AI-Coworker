"""
FastAPI application entry point.

Bootstraps the app, registers routers, and configures middleware.
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.infrastructure.config.settings import get_settings
from src.presentation.api.chat_router import router as chat_router
from src.presentation.api.health_router import router as health_router

# ─── Logging ─────────────────────────────────────────────────────────────────
settings = get_settings()
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Co-Worker NPC Engine",
    description=(
        "Powers virtual AI co-workers (NPCs) in interactive workplace simulations. "
        "Each NPC has a distinct persona, memory, emotion state, and business function."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.is_development else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ─────────────────────────────────────────────────────────────────
app.include_router(health_router)
app.include_router(chat_router)

logger.info(
    "AI Co-Worker NPC Engine started | env=%s | provider=%s",
    settings.app_env,
    settings.active_provider,
)
