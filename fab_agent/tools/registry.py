"""Tool registry: names -> (callable, input model), and OpenAI schema export.

The agent (Stage 3) drives these tools through OpenAI-compatible chat-completions
tool calling. ``openai_tool_schemas()`` produces the ``tools`` list to pass to
the model; ``TOOLS`` maps a called tool name back to its function and the
pydantic model used to validate the model's arguments.
"""

from __future__ import annotations

import inspect
from typing import Callable

from pydantic import BaseModel

from fab_agent.tools.cards import GetCardInput, get_card
from fab_agent.tools.clarify import AskClarificationInput, ask_clarification
from fab_agent.tools.precedence import GetPrecedenceInput, get_precedence_context
from fab_agent.tools.rules import SearchRulesInput, search_rules

#: name -> (callable, input model). Insertion order defines schema order.
TOOLS: dict[str, tuple[Callable, type[BaseModel]]] = {
    "search_rules": (search_rules, SearchRulesInput),
    "get_card": (get_card, GetCardInput),
    "get_precedence_context": (get_precedence_context, GetPrecedenceInput),
    "ask_clarification": (ask_clarification, AskClarificationInput),
}


def _schema_for(name: str, fn: Callable, model: type[BaseModel]) -> dict:
    parameters = model.model_json_schema()
    parameters.pop("title", None)  # noise; not needed by the API
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": inspect.getdoc(fn) or "",
            "parameters": parameters,
        },
    }


def openai_tool_schemas() -> list[dict]:
    """Return the OpenAI chat-completions ``tools`` schema list for all tools."""
    return [_schema_for(name, fn, model) for name, (fn, model) in TOOLS.items()]


def call_tool(name: str, arguments: dict):
    """Validate ``arguments`` against the tool's input model and invoke it."""
    if name not in TOOLS:
        raise KeyError(f"unknown tool: {name!r}")
    fn, model = TOOLS[name]
    validated = model.model_validate(arguments)
    return fn(**validated.model_dump())
