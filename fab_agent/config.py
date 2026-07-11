"""Central configuration for fab_agent.

All runtime behaviour that distinguishes one model backend from another is
funnelled through :class:`Settings`. Nothing else in the codebase should read
environment variables directly, so that swapping Gemini for a local Ollama
model is a pure configuration change (see PROJECT.md).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["gemini", "openai", "anthropic", "ollama"]

#: Default OpenAI-compatible endpoint exposed by a local Ollama server.
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"


class Settings(BaseSettings):
    """Application settings, populated from environment / ``.env``.

    Attributes map 1:1 to the variables documented in ``.env.example``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Model backend selection -------------------------------------------
    llm_provider: Provider = Field(
        default="gemini",
        description="Which LLM backend to use.",
    )
    llm_model: str = Field(
        default="gemini-3.5-flash",
        description="Model identifier passed to the provider.",
    )
    llm_api_key: Optional[str] = Field(
        default=None,
        description="API key for the selected provider (unused for Ollama).",
    )
    llm_base_url: Optional[str] = Field(
        default=None,
        description="Override for the provider base URL (OpenAI-compatible).",
    )

    # --- Retrieval ---------------------------------------------------------
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="sentence-transformers model used for dense retrieval.",
    )

    # --- Adjudication ------------------------------------------------------
    rel_level: str = Field(
        default="competitive",
        description="Rules Enforcement Level context for rulings "
        "(e.g. casual, competitive, professional).",
    )

    @model_validator(mode="after")
    def _default_ollama_base_url(self) -> "Settings":
        """Ollama exposes an OpenAI-compatible API on a fixed local port.

        If the user selects the ``ollama`` provider without overriding
        ``LLM_BASE_URL``, fall back to the conventional local endpoint.
        """
        if self.llm_provider == "ollama" and not self.llm_base_url:
            object.__setattr__(self, "llm_base_url", OLLAMA_DEFAULT_BASE_URL)
        return self

    @property
    def is_local(self) -> bool:
        """True when the backend runs on-device (open-weights via Ollama)."""
        return self.llm_provider == "ollama"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance for the process."""
    return Settings()
