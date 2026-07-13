#!/usr/bin/env python3
"""Ask the rules agent a question and pretty-print the IRAC verdict.

Usage:
    uv run python scripts/ask.py "If I block with a card, can I later pay a cost with it?"
    uv run python scripts/ask.py "What does Command and Conquer do?" --setting casual
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from typing import Optional

from fab_agent.agent.llm import LLMConnectionError, make_client_from_settings
from fab_agent.agent.loop import RulesAgent

_WRAP = 96


def _para(label: str, text: str) -> None:
    print(f"\n{label}")
    for line in textwrap.wrap(text or "", width=_WRAP):
        print(f"  {line}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="The rules question.")
    parser.add_argument("--setting", choices=["casual", "tournament"], default="tournament")
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args(argv)

    client = make_client_from_settings()
    try:
        client.check_connection()
    except LLMConnectionError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1

    agent = RulesAgent(client=client)
    print(f"Provider: {client.provider} / {client.model}   setting: {args.setting}")
    print(f"Q: {args.query}")
    result = agent.run(args.query, setting=args.setting, max_steps=args.max_steps)

    print("\n" + "=" * _WRAP)
    print("TOOL-CALL TRACE")
    print("=" * _WRAP)
    if not result.trace:
        print("  (no tool calls)")
    for st in result.trace:
        print(f"  [{st.step}] {st.tool_name}({_fmt_args(st.tool_args)})")
        print(f"       → {st.tool_result_summary}")

    if result.error:
        print(f"\n✗ Agent error: {result.error}")

    verdict = result.verdict
    if verdict is None:
        print("\n(no verdict produced)")
        _print_usage(result)
        return 1

    print("\n" + "=" * _WRAP)
    print("VERDICT" + (f"   [confidence: {verdict.confidence}]"))
    print("=" * _WRAP)

    if verdict.clarification_request:
        _para("⚑ CLARIFICATION NEEDED", verdict.clarification_request)

    _para("ISSUE", verdict.issue)
    _para("RULE", verdict.rule)
    _para("APPLICATION", verdict.application)
    _para("CONCLUSION", verdict.conclusion)
    if verdict.precedence_notes:
        _para("PRECEDENCE", verdict.precedence_notes)

    print("\nCITATIONS")
    if not verdict.citations:
        print("  (none)")
    for c in verdict.citations:
        flag = "" if c.valid else "  ⚠ UNVERIFIED (not retrieved this run)"
        print(f"  • [{c.chunk_id}] {c.doc} {c.rule_id}{flag}")
        print(f'      "{c.quoted_snippet}"')
    if result.invalid_citations:
        print(f"\n  ⚠ invalid citations: {result.invalid_citations}")

    _print_usage(result)
    if result.trace_path:
        print(f"Trace saved: {result.trace_path}")
    return 0


def _fmt_args(args: Optional[dict]) -> str:
    if not args:
        return ""
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


def _print_usage(result) -> None:
    u = result.token_usage
    print(f"\nTokens: {u.get('total_tokens', 0)} "
          f"(prompt {u.get('prompt_tokens', 0)}, completion {u.get('completion_tokens', 0)})")


if __name__ == "__main__":
    raise SystemExit(main())
