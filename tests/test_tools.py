"""Unit tests for the tool layer. No LLM, no network, no real index."""

import pytest
from pydantic import ValidationError

from fab_agent.config import get_settings
from fab_agent.retrieval.hybrid import RetrievedChunk
from fab_agent.tools.cards import get_card
from fab_agent.tools.clarify import ask_clarification
from fab_agent.tools.precedence import PRECEDENCE_PATH, get_precedence_context
from fab_agent.tools.registry import TOOLS, call_tool, openai_tool_schemas
from fab_agent.tools.rules import search_rules

# --------------------------------------------------------------------------- #
# search_rules — thin wrapper over the retriever
# --------------------------------------------------------------------------- #
class FakeRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, query, k=8, mode="hybrid", doc_filter=None):
        self.calls.append({"query": query, "k": k, "mode": mode, "doc_filter": doc_filter})
        return [
            RetrievedChunk(chunk_id="CR-1.2.3", doc="CR", rule_id="1.2.3",
                           text="A rule about targeting.", score=0.9, rank=1,
                           retriever_source="both"),
            RetrievedChunk(chunk_id="PPG-2.1", doc="PPG", rule_id="2.1",
                           text="An infraction.", score=0.5, rank=2,
                           retriever_source="dense"),
        ]


def test_search_rules_maps_hits_and_passes_through_args():
    fake = FakeRetriever()
    out = search_rules("go again", k=3, doc_filter=["PPG"], retriever=fake)

    assert fake.calls == [{"query": "go again", "k": 3, "mode": "hybrid", "doc_filter": ["PPG"]}]
    assert [h.chunk_id for h in out] == ["CR-1.2.3", "PPG-2.1"]
    first = out[0]
    assert (first.rule_id, first.doc, first.text) == ("1.2.3", "CR", "A rule about targeting.")
    # Only the four documented fields are exposed.
    assert set(first.model_dump()) == {"chunk_id", "rule_id", "doc", "text"}


# --------------------------------------------------------------------------- #
# get_card — fuzzy matching
# --------------------------------------------------------------------------- #
FIX_CARDS = [
    {"name": "In the Palm of Your Hand", "pitch": 3, "types": ["Wizard", "Instant"],
     "text": "Your next attack this turn gains go again."},
    {"name": "Sink Below", "pitch": 3, "types": ["Ninja", "Action"],
     "text": "Instant - Draw a card."},
    {"name": "Command and Conquer", "pitch": 2, "types": ["Warrior", "Action"],
     "text": "Dominate. Go again"},
]


def test_get_card_exact_match():
    out = get_card("In the Palm of Your Hand", cards=FIX_CARDS)
    assert out.matched is True
    assert out.name == "In the Palm of Your Hand"
    assert out.pitch == 3
    assert out.types == ["Wizard", "Instant"]
    assert out.suggestions == []


def test_get_card_case_insensitive_exact():
    out = get_card("sink below", cards=FIX_CARDS)
    assert out.matched is True and out.name == "Sink Below"


def test_get_card_fuzzy_match():
    out = get_card("palm of your hand", cards=FIX_CARDS)
    assert out.matched is True
    assert out.name == "In the Palm of Your Hand"


def test_get_card_name_inside_longer_phrase():
    out = get_card("I want to cast in the palm of your hand now", cards=FIX_CARDS)
    assert out.matched is True and out.name == "In the Palm of Your Hand"


def test_get_card_no_confident_match_returns_suggestions():
    out = get_card("sink", cards=FIX_CARDS)
    assert out.matched is False
    assert "Sink Below" in out.suggestions
    assert len(out.suggestions) <= 3


def test_get_card_unknown_returns_empty():
    out = get_card("Zzzqqq Nonexistent Widget", cards=FIX_CARDS)
    assert out.matched is False
    assert out.suggestions == []


# --------------------------------------------------------------------------- #
# get_precedence_context
# --------------------------------------------------------------------------- #
def test_precedence_template_exists():
    assert PRECEDENCE_PATH.exists()


def test_precedence_tournament_mentions_trp_ppg_and_rel():
    ctx = get_precedence_context("tournament", "competitive")
    assert ctx.setting == "tournament"
    assert ctx.rel_level == "competitive"
    assert "TRP" in ctx.text and "PPG" in ctx.text
    assert "Comprehensive Rules" in ctx.text
    assert "competitive" in ctx.text
    # Placeholders substituted; editor comments stripped.
    assert "{setting}" not in ctx.text and "{rel_level}" not in ctx.text
    assert "<!--" not in ctx.text


def test_precedence_casual_excludes_tournament_scope():
    ctx = get_precedence_context("casual", "casual")
    assert "Procedures and Penalty Guide" not in ctx.text
    assert "casual" in ctx.text.lower()
    assert "Comprehensive Rules" in ctx.text


def test_precedence_defaults_rel_level_from_config():
    ctx = get_precedence_context("casual")
    assert ctx.rel_level == get_settings().rel_level


# --------------------------------------------------------------------------- #
# ask_clarification
# --------------------------------------------------------------------------- #
def test_ask_clarification_is_a_sentinel():
    req = ask_clarification("Which hero is attacking?", "the attacking hero is unspecified")
    assert req.type == "clarification_request"
    assert req.question == "Which hero is attacking?"
    assert req.missing_info == "the attacking hero is unspecified"


def test_ask_clarification_description_warns_against_overuse():
    desc = {s["function"]["name"]: s["function"]["description"] for s in openai_tool_schemas()}
    text = desc["ask_clarification"].lower()
    assert "only" in text
    assert "penal" in text  # documents that unnecessary clarifications are penalized


# --------------------------------------------------------------------------- #
# registry / schema
# --------------------------------------------------------------------------- #
def test_tools_dict_shape():
    assert list(TOOLS) == ["search_rules", "get_card", "get_precedence_context", "ask_clarification"]
    for name, (fn, model) in TOOLS.items():
        assert callable(fn)
        assert hasattr(model, "model_json_schema")


def test_openai_tool_schemas_snapshot():
    schemas = openai_tool_schemas()
    assert [s["function"]["name"] for s in schemas] == [
        "search_rules", "get_card", "get_precedence_context", "ask_clarification",
    ]
    by = {s["function"]["name"]: s["function"] for s in schemas}

    assert all(s["type"] == "function" for s in schemas)
    assert all(f["description"] for f in by.values())
    assert all(f["parameters"]["type"] == "object" for f in by.values())

    assert set(by["search_rules"]["parameters"]["properties"]) == {"query", "k", "doc_filter"}
    assert by["search_rules"]["parameters"]["required"] == ["query"]

    assert set(by["get_card"]["parameters"]["properties"]) == {"name"}
    assert by["get_card"]["parameters"]["required"] == ["name"]

    gp = by["get_precedence_context"]["parameters"]
    assert set(gp["properties"]) == {"setting", "rel_level"}
    assert gp["required"] == ["setting"]
    assert set(gp["properties"]["setting"]["enum"]) == {"casual", "tournament"}

    ac = by["ask_clarification"]["parameters"]
    assert set(ac["properties"]) == {"question", "missing_info"}
    assert set(ac["required"]) == {"question", "missing_info"}


def test_call_tool_validates_and_dispatches():
    res = call_tool("get_precedence_context", {"setting": "casual", "rel_level": "casual"})
    assert res.setting == "casual"

    sentinel = call_tool("ask_clarification", {"question": "q?", "missing_info": "m"})
    assert sentinel.type == "clarification_request"

    with pytest.raises(KeyError):
        call_tool("nonexistent_tool", {})
    with pytest.raises(ValidationError):
        call_tool("get_precedence_context", {"setting": "not-a-setting"})
