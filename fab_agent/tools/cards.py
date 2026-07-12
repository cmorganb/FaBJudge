"""Tool: get_card — fuzzy card lookup over the normalized card database."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from rapidfuzz import fuzz, process, utils

CARDS_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "cards.jsonl"

#: Length-sensitive edit similarity (0-100) at/above which a fuzzy match is
#: accepted as the intended card. Using the length-sensitive ``ratio`` (rather
#: than a partial scorer) keeps short fragments like "dagger" from confidently
#: matching a longer card name; those return suggestions instead.
CONFIDENT_SCORE = 72.0
#: WRatio floor (0-100) for a name to be offered as a suggestion at all.
SUGGESTION_CUTOFF = 60.0

_CARDS: Optional[list[dict]] = None


def _load_cards() -> list[dict]:
    global _CARDS
    if _CARDS is None:
        with CARDS_PATH.open("r", encoding="utf-8") as fh:
            _CARDS = [json.loads(line) for line in fh if line.strip()]
    return _CARDS


class GetCardInput(BaseModel):
    name: str = Field(description="The card name (approximate spelling is fine).")


class GetCardOutput(BaseModel):
    name: str
    pitch: Optional[int] = None
    types: list[str] = Field(default_factory=list)
    text: str = ""
    matched: bool
    suggestions: list[str] = Field(default_factory=list)


def _found(card: dict) -> GetCardOutput:
    return GetCardOutput(
        name=card.get("name", ""),
        pitch=card.get("pitch"),
        types=card.get("types", []),
        text=card.get("text", ""),
        matched=True,
        suggestions=[],
    )


def get_card(name: str, *, cards: Optional[list[dict]] = None) -> GetCardOutput:
    """Look up a Flesh and Blood card by name and return its printed text.

    Matching is fuzzy, so approximate names work (e.g. 'palm of your hand'
    resolves to 'In the Palm of Your Hand'). If no confident match is found,
    `matched` is False and up to three close `suggestions` are returned; ask the
    user to confirm or re-run with a suggested name rather than guessing.
    """
    cards = cards if cards is not None else _load_cards()

    # First occurrence wins for names shared across pitch variants.
    lower_to_card: dict[str, dict] = {}
    for card in cards:
        lower_to_card.setdefault((card.get("name") or "").lower(), card)

    query = name.strip().lower()
    if not query:
        return GetCardOutput(name=name, matched=False, suggestions=[])

    if query in lower_to_card:  # exact (case-insensitive)
        return _found(lower_to_card[query])

    names_lower = list(lower_to_card)
    # Rank candidates with WRatio (token-aware; robust to word order/partials).
    ranked = process.extract(
        query, names_lower, scorer=fuzz.WRatio,
        processor=utils.default_process, limit=3, score_cutoff=SUGGESTION_CUTOFF,
    )
    if not ranked:
        return GetCardOutput(name=name, matched=False, suggestions=[])

    best = ranked[0][0]
    # Accept when the length-sensitive edit similarity is high, or when the full
    # card name appears inside a longer user phrase ("play in the palm ...").
    edit_score = fuzz.ratio(query, best, processor=utils.default_process)
    if edit_score >= CONFIDENT_SCORE or best in query:
        return _found(lower_to_card[best])

    return GetCardOutput(
        name=name,
        matched=False,
        suggestions=[lower_to_card[choice]["name"] for choice, _score, _idx in ranked],
    )
