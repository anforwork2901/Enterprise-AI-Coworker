"""
Application settings loaded from environment variables.
Uses Pydantic BaseSettings for type-safe config with validation.
"""
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Active Provider Selection ───────────────────────────────────────────
    llm_provider: str = Field(default="gemini", description="LLM provider: 'gemini' or 'openai'")

    # ── LLM Provider ─────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="LLM model ID")
    openai_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    openai_max_tokens: int = Field(default=1024, gt=0)
    openai_embedding_model: str = Field(default="text-embedding-3-small", description="OpenAI Embedding model")

    # ── Gemini Provider ──────────────────────────────────────────────────────
    gemini_api_key: str = Field(default="", description="Gemini API key")
    gemini_model: str = Field(default="gemini-2.5-flash", description="Gemini model ID")

    # ── Application ───────────────────────────────────────────────────────────
    app_env: str = Field(default="development")
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")

    # ── Vector Store ──────────────────────────────────────────────────────────
    faiss_index_path: str = Field(default="data/faiss_index")
    embedding_model: str = Field(default="models/gemini-embedding-001", description="Gemini Embedding model ID")

    # ── Data Paths ────────────────────────────────────────────────────────────
    personas_dir: str = Field(default="personas")
    knowledge_base_dir: str = Field(default="data/knowledge_base")

    @property
    def active_provider(self) -> str:
        """
        Determine the active LLM provider.
        If user explicitly set llm_provider to 'openai' or 'gemini', respect it.
        Otherwise auto-detect based on configured API keys.
        """
        provider = self.llm_provider.strip().lower()
        if provider in ("openai", "gemini"):
            return provider
        if self.gemini_api_key and not self.openai_api_key:
            return "gemini"
        if self.openai_api_key and not self.gemini_api_key:
            return "openai"
        return "gemini"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached singleton Settings instance.
    lru_cache ensures env is read only once per process lifetime.
    """
    return Settings()
