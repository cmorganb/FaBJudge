"""Smoke test: the package imports and configuration loads with defaults."""

import fab_agent
from fab_agent.config import Settings, get_settings


def test_package_imports():
    assert fab_agent.__version__


def test_config_loads_defaults():
    settings = get_settings()
    assert isinstance(settings, Settings)
    # Documented defaults from .env.example.
    assert settings.llm_provider == "gemini"
    assert settings.llm_model == "gemini-3.5-flash"
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.rel_level == "competitive"


def test_ollama_base_url_default():
    """Selecting Ollama without an explicit base URL resolves the local one."""
    settings = Settings(llm_provider="ollama", llm_base_url=None)
    assert settings.llm_base_url == "http://localhost:11434/v1"
    assert settings.is_local is True
