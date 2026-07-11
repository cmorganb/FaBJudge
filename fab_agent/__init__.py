"""fab_agent: an LLM-based rules-adjudication agent for Flesh and Blood.

The package is intentionally model-agnostic: the same retrieval and reasoning
architecture runs unchanged across closed models (Gemini, GPT, Claude) and
open-weights models served locally via Ollama, selected purely through
configuration (see :mod:`fab_agent.config`).
"""

__version__ = "0.1.0"

from fab_agent.config import Settings, get_settings

__all__ = ["Settings", "get_settings", "__version__"]
