"""Tests for HybridRetriever, built over the 10-chunk fixture from test_index."""

import pytest

from fab_agent.config import get_settings
from fab_agent.retrieval.hybrid import HybridRetriever, RetrievedChunk
from fab_agent.retrieval.index import BM25Index, ChromaIndex
from tests.test_index import FIXTURE


@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    model = get_settings().embedding_model
    chroma_path = tmp_path_factory.mktemp("chroma") / "chroma"
    bm25 = BM25Index.build(FIXTURE)
    chroma = ChromaIndex.build(FIXTURE, model, path=chroma_path,
                               batch_size=8, show_progress=False)
    return HybridRetriever(FIXTURE, bm25, chroma)


def _ids(results):
    return [r.chunk_id for r in results]


# --------------------------------------------------------------------------- #
# Modes (ablation switch)
# --------------------------------------------------------------------------- #
def test_bm25_mode_matches_rule_number(retriever):
    results = retriever.retrieve("1.2.3a", k=3, mode="bm25")
    assert results[0].chunk_id == "CR-1.2.3a"
    assert all(r.retriever_source == "bm25" for r in results)


def test_dense_mode_matches_paraphrase(retriever):
    results = retriever.retrieve(
        "how much health does a hero have at the start of a match", k=3, mode="dense"
    )
    assert results[0].chunk_id == "CR-2.1"
    assert all(r.retriever_source == "dense" for r in results)


# --------------------------------------------------------------------------- #
# Hybrid fusion / union behavior
# --------------------------------------------------------------------------- #
def test_hybrid_results_are_subset_of_union(retriever):
    query = "a hero blocks by committing cards from hand to reduce damage"
    hybrid = set(_ids(retriever.retrieve(query, k=8, mode="hybrid")))
    bm25 = set(_ids(retriever.retrieve(query, k=20, mode="bm25")))
    dense = set(_ids(retriever.retrieve(query, k=20, mode="dense")))
    # Fusion never invents chunks outside what the two retrievers found.
    assert hybrid <= (bm25 | dense)
    # The chunk both retrievers agree on is present and labelled "both".
    assert "CR-7.5" in hybrid
    both = next(r for r in retriever.retrieve(query, k=8) if r.chunk_id == "CR-7.5")
    assert both.retriever_source == "both"


def test_hybrid_surfaces_semantic_and_lexical_hits(retriever):
    # A lexical rule-number hit and a semantic paraphrase hit both survive fusion.
    lexical = retriever.retrieve("rule 1.2.3a", k=8, mode="hybrid")
    assert "CR-1.2.3a" in _ids(lexical)

    semantic = retriever.retrieve("what is a player's starting life total", k=8, mode="hybrid")
    assert "CR-2.1" in _ids(semantic)


def test_hybrid_deduplicates_and_ranks(retriever):
    results = retriever.retrieve("arsenal card face down", k=8, mode="hybrid")
    ids = _ids(results)
    assert len(ids) == len(set(ids))  # de-duplicated by chunk_id
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)  # descending fused score
    assert all(isinstance(r, RetrievedChunk) for r in results)


# --------------------------------------------------------------------------- #
# doc_filter
# --------------------------------------------------------------------------- #
def test_doc_filter_restricts_results(retriever):
    results = retriever.retrieve("penalty for taking too long", k=8,
                                 mode="hybrid", doc_filter=["PPG"])
    assert results, "expected at least one PPG result"
    assert all(r.doc == "PPG" for r in results)
    assert "PPG-2.1" in _ids(results)


def test_doc_filter_excludes_other_docs(retriever):
    # The same query scoped to TRP must not return the PPG slow-play chunk.
    results = retriever.retrieve("penalty for taking too long", k=8,
                                 mode="hybrid", doc_filter=["TRP"])
    assert all(r.doc == "TRP" for r in results)
    assert "PPG-2.1" not in _ids(results)


def test_doc_filter_multiple_docs(retriever):
    results = retriever.retrieve("rules enforcement and penalties", k=10,
                                 mode="hybrid", doc_filter=["PPG", "TRP"])
    assert results
    assert all(r.doc in {"PPG", "TRP"} for r in results)
