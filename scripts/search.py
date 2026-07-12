#!/usr/bin/env python3
"""Manual retrieval sanity check.

Runs a query through :class:`fab_agent.retrieval.hybrid.HybridRetriever` and
prints the top results (chunk_id, score, source, and the first 200 characters).

Usage:
    uv run python scripts/search.py "how much life does a hero start with"
    uv run python scripts/search.py "1.2.3a" --mode bm25 --k 5
    uv run python scripts/search.py "unsporting conduct penalty" --doc PPG TRP
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from fab_agent.retrieval.hybrid import HybridRetriever


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="The search query.")
    parser.add_argument("--k", type=int, default=8, help="Results to show (default 8).")
    parser.add_argument("--mode", choices=["hybrid", "bm25", "dense"], default="hybrid")
    parser.add_argument("--doc", nargs="+", default=None, metavar="DOC",
                        help="Restrict to document codes, e.g. --doc PPG TRP.")
    args = parser.parse_args(argv)

    retriever = HybridRetriever.load()
    results = retriever.retrieve(args.query, k=args.k, mode=args.mode, doc_filter=args.doc)

    header = f'Query: {args.query!r}   mode={args.mode}   k={args.k}'
    if args.doc:
        header += f'   doc_filter={args.doc}'
    print(header)
    print("=" * 78)
    if not results:
        print("(no results)")
        return 0

    for r in results:
        snippet = " ".join(r.text.split())[:200]
        title = f"  «{r.title}»" if r.title else ""
        print(f"[{r.rank:>2}] {r.chunk_id:<28} score={r.score:.4f}  "
              f"src={r.retriever_source}{title}")
        print(f"     {snippet}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
