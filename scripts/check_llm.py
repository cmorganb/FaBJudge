#!/usr/bin/env python3
"""Connectivity smoke test for the configured LLM backend.

Runs the startup connectivity check, then sends a one-token "reply with OK" and
prints the provider, model, latency, token usage, and the reply. Use it to
confirm a backend is reachable before running the agent or the experiments.

Usage:
    uv run python scripts/check_llm.py
    # or point at a specific backend for the Stage-5 comparison:
    uv run python scripts/check_llm.py --provider ollama --model llama3.1:8b
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

from fab_agent.agent.llm import LLMConnectionError, make_client, make_client_from_settings
from fab_agent.config import get_settings


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default=None, help="Override LLM_PROVIDER.")
    parser.add_argument("--model", default=None, help="Override LLM_MODEL.")
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.provider or args.model:
        client = make_client(
            args.provider or settings.llm_provider,
            args.model or settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=None if args.provider else settings.llm_base_url,
        )
    else:
        client = make_client_from_settings(settings)

    print(f"Provider: {client.provider}")
    print(f"Model:    {client.model}")

    try:
        client.check_connection()
    except LLMConnectionError as exc:
        print(f"\n✗ Connectivity check failed:\n  {exc}", file=sys.stderr)
        return 1

    try:
        t0 = time.perf_counter()
        resp = client.chat(
            [{"role": "user", "content": "Reply with OK"}],
            temperature=0.0,
        )
        latency = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        print(f"\n✗ Chat request failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"Latency:  {latency * 1000:.0f} ms")
    print(f"Tokens:   {resp.usage.total_tokens} "
          f"(prompt {resp.usage.prompt_tokens}, completion {resp.usage.completion_tokens})")
    print(f"Reply:    {(resp.content or '').strip()!r}")
    print("\n✓ OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
