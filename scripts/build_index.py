#!/usr/bin/env python3
"""Rebuild the lexical (BM25) and semantic (Chroma) indexes from chunks.jsonl.

Both indexes are rebuilt from scratch (delete-and-rebuild) so the operation is
idempotent and deterministic. Prints chunk counts, embedding dimension, and
build times.

Usage:
    uv run python scripts/build_index.py
    uv run python scripts/build_index.py --batch-size 128
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from fab_agent.config import get_settings
from fab_agent.retrieval.index import (
    BM25_PATH,
    BM25Index,
    CHROMA_PATH,
    CHUNKS_PATH,
    ChromaIndex,
    INDEX_DIR,
    load_chunks,
)

ROOT = Path(__file__).resolve().parents[1]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Embedding batch size (default: 64).")
    args = parser.parse_args(argv)

    if not CHUNKS_PATH.exists():
        print(f"Missing {CHUNKS_PATH}. Run the chunker first "
              f"(python -m fab_agent.corpus.chunker).", file=sys.stderr)
        return 1

    settings = get_settings()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks):,} chunks from {CHUNKS_PATH.relative_to(ROOT)}")
    doc_counts: dict[str, int] = {}
    for c in chunks:
        doc_counts[c.get("doc", "?")] = doc_counts.get(c.get("doc", "?"), 0) + 1

    # ---------------------------------------------------------------- BM25 ---
    print("\n[1/2] Building BM25 lexical index …")
    t0 = time.perf_counter()
    bm25 = BM25Index.build(chunks)
    if BM25_PATH.exists():  # explicit delete-and-rebuild
        BM25_PATH.unlink()
    bm25.save(BM25_PATH)
    bm25_time = time.perf_counter() - t0
    vocab = len({tok for doc in bm25.tokenized_corpus for tok in doc})
    print(f"      saved {BM25_PATH.relative_to(ROOT)}  "
          f"({len(bm25.chunk_ids):,} docs, vocab {vocab:,})  in {bm25_time:.2f}s")

    # -------------------------------------------------------------- Chroma ---
    print(f"\n[2/2] Building Chroma semantic index "
          f"(model: {settings.embedding_model}) …")
    t0 = time.perf_counter()
    chroma = ChromaIndex.build(
        chunks, settings.embedding_model, path=CHROMA_PATH,
        batch_size=args.batch_size, show_progress=True,
    )
    chroma_time = time.perf_counter() - t0
    print(f"      persisted {CHROMA_PATH.relative_to(ROOT)}  "
          f"({chroma.count():,} vectors, dim {chroma.dimension})  in {chroma_time:.2f}s")

    # --------------------------------------------------------------- Stats ---
    print("\n" + "=" * 68)
    print("Index build complete")
    print("=" * 68)
    print(f"  Chunks indexed:     {len(chunks):,}")
    print("  By document:        " + ", ".join(f"{k}={v}" for k, v in sorted(doc_counts.items())))
    print(f"  BM25 vocabulary:    {vocab:,} unique tokens")
    print(f"  Embedding model:    {settings.embedding_model}")
    print(f"  Embedding dim:      {chroma.dimension}")
    print(f"  BM25 build time:    {bm25_time:.2f}s")
    print(f"  Chroma build time:  {chroma_time:.2f}s")
    print(f"  Total build time:   {bm25_time + chroma_time:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
