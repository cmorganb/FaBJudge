"""The single retrieval entry point used by the agent and the eval harness.

:class:`HybridRetriever` fuses lexical (BM25) and dense (Chroma) retrieval and
exposes a ``mode`` switch so the same code path can be run three ways for the
Stage 5 ablation study:

* ``"hybrid"`` — BM25 (top 20) + vector (top 20) merged with Reciprocal Rank
  Fusion (RRF, k=60), de-duplicated by ``chunk_id`` (the default and the
  system's real behaviour).
* ``"bm25"``  — lexical only.
* ``"dense"`` — dense/semantic only.

Keeping lexical-only and dense-only reachable through one parameter lets the
evaluation replay identical queries against each configuration and attribute
differences to the retrieval strategy rather than to code differences — the
"grounding architecture held constant" principle from PROJECT.md.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Literal, Optional, Union

from pydantic import Field

from fab_agent.corpus.chunker import Chunk
from fab_agent.retrieval.index import (
    BM25_PATH,
    BM25Index,
    CHROMA_PATH,
    CHUNKS_PATH,
    ChromaIndex,
    Hit,
)

Mode = Literal["hybrid", "bm25", "dense"]
Source = Literal["bm25", "dense", "both"]

#: Depth pulled from each retriever before fusion.
POOL = 20
#: RRF constant; larger values flatten the contribution of top ranks.
RRF_K = 60
#: When a doc_filter is set we over-fetch so enough candidates survive filtering.
FILTER_FETCH = 200


class RetrievedChunk(Chunk):
    """A :class:`Chunk` annotated with its retrieval score, rank, and origin."""

    score: float
    rank: int
    retriever_source: Source = Field(
        description="'bm25', 'dense', or 'both' (found by both retrievers)."
    )


class HybridRetriever:
    """Fuses BM25 and Chroma retrieval behind a single ``retrieve`` call."""

    def __init__(
        self,
        chunks: Iterable[Union[Chunk, dict]],
        bm25: BM25Index,
        chroma: ChromaIndex,
    ):
        self.chunks: dict[str, Chunk] = {}
        for c in chunks:
            chunk = c if isinstance(c, Chunk) else Chunk.model_validate(c)
            self.chunks[chunk.chunk_id] = chunk
        self.bm25 = bm25
        self.chroma = chroma

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def load(
        cls,
        *,
        chunks_path: Path = CHUNKS_PATH,
        bm25_path: Path = BM25_PATH,
        chroma_path: Path = CHROMA_PATH,
        model_name: Optional[str] = None,
    ) -> "HybridRetriever":
        """Load the persisted indexes and chunk store from disk."""
        if model_name is None:
            from fab_agent.config import get_settings

            model_name = get_settings().embedding_model

        with Path(chunks_path).open("r", encoding="utf-8") as fh:
            chunks = [json.loads(line) for line in fh if line.strip()]

        return cls(
            chunks=chunks,
            bm25=BM25Index.load(bm25_path),
            chroma=ChromaIndex.load(model_name, path=chroma_path),
        )

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        query: str,
        k: int = 8,
        mode: Mode = "hybrid",
        doc_filter: Optional[list[str]] = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the top ``k`` chunks for ``query``.

        Args:
            query: Natural-language or keyword query.
            k: Number of chunks to return.
            mode: ``"hybrid"`` (RRF of both), ``"bm25"``, or ``"dense"``. The
                lexical-/dense-only modes exist for the Stage 5 ablation.
            doc_filter: If given, restrict results to these document codes
                (e.g. ``["PPG"]``); the router uses this to scope a query.
        """
        if mode == "bm25":
            hits = self._candidates(self.bm25, query, doc_filter)[:k]
            return [self._wrap(h.chunk_id, h.score, i, "bm25")
                    for i, h in enumerate(hits, start=1)]

        if mode == "dense":
            hits = self._candidates(self.chroma, query, doc_filter)[:k]
            # Chroma reports cosine distance; convert to a similarity score.
            return [self._wrap(h.chunk_id, 1.0 - h.score, i, "dense")
                    for i, h in enumerate(hits, start=1)]

        if mode == "hybrid":
            bm25_hits = self._candidates(self.bm25, query, doc_filter)[:POOL]
            dense_hits = self._candidates(self.chroma, query, doc_filter)[:POOL]
            return self._fuse(bm25_hits, dense_hits, k)

        raise ValueError(f"unknown mode: {mode!r}")

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _candidates(self, index, query: str,
                    doc_filter: Optional[list[str]]) -> list[Hit]:
        fetch = FILTER_FETCH if doc_filter else POOL
        hits = index.query(query, k=fetch)
        if doc_filter:
            allowed = set(doc_filter)
            hits = [h for h in hits
                    if (c := self.chunks.get(h.chunk_id)) is not None and c.doc in allowed]
        return hits

    def _fuse(self, bm25_hits: list[Hit], dense_hits: list[Hit],
              k: int) -> list[RetrievedChunk]:
        scores: dict[str, float] = defaultdict(float)
        sources: dict[str, set] = defaultdict(set)
        for source, hits in (("bm25", bm25_hits), ("dense", dense_hits)):
            for rank, hit in enumerate(hits, start=1):
                scores[hit.chunk_id] += 1.0 / (RRF_K + rank)
                sources[hit.chunk_id].add(source)

        # Deterministic: sort by fused score desc, chunk_id asc as tiebreak.
        order = sorted(scores, key=lambda cid: (-scores[cid], cid))[:k]
        results = []
        for i, cid in enumerate(order, start=1):
            src = sorted(sources[cid])
            source: Source = "both" if len(src) == 2 else src[0]  # type: ignore[assignment]
            results.append(self._wrap(cid, scores[cid], i, source))
        return results

    def _wrap(self, chunk_id: str, score: float, rank: int,
              source: Source) -> RetrievedChunk:
        base = self.chunks[chunk_id]
        return RetrievedChunk(**base.model_dump(), score=float(score),
                              rank=rank, retriever_source=source)
