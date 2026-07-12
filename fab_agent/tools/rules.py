"""Tool: search_rules — hybrid retrieval over the rules corpus."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

# Lazily-initialised shared retriever (loading Chroma + the embedding model is
# expensive, so we do it once per process and never in unit tests).
_RETRIEVER = None


def get_retriever():
    global _RETRIEVER
    if _RETRIEVER is None:
        from fab_agent.retrieval.hybrid import HybridRetriever

        _RETRIEVER = HybridRetriever.load()
    return _RETRIEVER


class SearchRulesInput(BaseModel):
    query: str = Field(description="Natural-language or keyword query about the rules.")
    k: int = Field(default=8, description="Number of passages to return.")
    doc_filter: Optional[list[str]] = Field(
        default=None,
        description="Restrict to document codes: 'CR' (game rules), 'PPG' "
        "(penalties), 'TRP' (tournament policy), 'CARD' (card text). Omit to "
        "search everything.",
    )


class RuleHit(BaseModel):
    chunk_id: str
    rule_id: str
    doc: str
    text: str


def search_rules(
    query: str,
    k: int = 8,
    doc_filter: Optional[list[str]] = None,
    *,
    retriever=None,
) -> list[RuleHit]:
    """Search the Flesh and Blood rules corpus for passages relevant to a query.

    Returns up to `k` passages, each with its citation anchor (`chunk_id`,
    `rule_id`) and document code. Use `doc_filter` to scope the search — e.g.
    ['PPG'] for infractions/penalties, ['CR'] for game rules, ['CARD'] for card
    text. This is the primary way to gather grounding before issuing a ruling.
    """
    retriever = retriever or get_retriever()
    hits = retriever.retrieve(query, k=k, doc_filter=doc_filter)
    return [
        RuleHit(chunk_id=h.chunk_id, rule_id=h.rule_id, doc=h.doc, text=h.text)
        for h in hits
    ]
