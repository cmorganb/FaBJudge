#!/usr/bin/env python3
"""Demonstrate each tool in the fab_agent tool layer.

search_rules and get_card touch the real index / card database; the other two
are pure. Run after the corpus and indexes are built.

Usage:
    uv run python scripts/demo_tools.py
"""

from __future__ import annotations

import json

from fab_agent.tools.cards import get_card
from fab_agent.tools.clarify import ask_clarification
from fab_agent.tools.precedence import get_precedence_context
from fab_agent.tools.registry import openai_tool_schemas
from fab_agent.tools.rules import search_rules


def _rule(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def main() -> int:
    _rule("1. search_rules('does an attack with go again refund an action?', doc_filter=['CR'])")
    for hit in search_rules(
        "does an attack with go again refund an action point", k=3, doc_filter=["CR"]
    ):
        print(f"  [{hit.chunk_id}] ({hit.doc} {hit.rule_id})")
        print(f"      {hit.text[:160]}")

    _rule("2a. get_card('palm of your hand')  — fuzzy match")
    print("  " + get_card("palm of your hand").model_dump_json(indent=2).replace("\n", "\n  "))

    _rule("2b. get_card('dagger')  — no confident match, returns suggestions")
    print("  " + get_card("dagger").model_dump_json(indent=2).replace("\n", "\n  "))

    _rule("3. get_precedence_context('tournament', 'competitive')")
    ctx = get_precedence_context("tournament", "competitive")
    print(f"  setting={ctx.setting}  rel_level={ctx.rel_level}\n")
    print("  " + ctx.text.replace("\n", "\n  "))

    _rule("4. ask_clarification(...)  — sentinel, does not answer")
    req = ask_clarification(
        question="Which of your heroes is the attacking hero this combat?",
        missing_info="the attacking hero is not specified and the ruling depends on it",
    )
    print("  " + req.model_dump_json(indent=2).replace("\n", "\n  "))

    _rule("5. openai_tool_schemas()  — OpenAI-compatible tool JSON")
    schemas = openai_tool_schemas()
    print(f"  {len(schemas)} tools: " + ", ".join(s["function"]["name"] for s in schemas))
    print("\n  search_rules schema:")
    print("  " + json.dumps(schemas[0], indent=2)[:900].replace("\n", "\n  "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
