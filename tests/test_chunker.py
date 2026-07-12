"""Tests for the chunker on a synthetic 3-level-nested markdown fixture."""

from fab_agent.corpus.chunker import (
    MAX_CHARS,
    Chunk,
    ancestor_ids,
    chunk_cards,
    chunk_document,
    rule_depth,
    split_text,
)

# A long rule body (> MAX_CHARS) built from paragraphs so it splits cleanly.
_LONG_BODY = "\n\n".join(
    f"Paragraph {i} of a very long rule that must be split into pieces "
    "because it exceeds the maximum chunk size threshold set for retrieval." * 3
    for i in range(6)
)

FIXTURE = f"""---
document: Test Doc
version: v0
---

# 1 Game Concepts

**1.0** General

**1.0.1** A short base rule.

**1.0.1a** A short lettered sub-rule.

**1.0.2** {_LONG_BODY}

# 2 Combat

**2.1** Attacks

**2.1.3** Some rule text that is moderately sized but under the threshold.
"""


def _by_id(chunks):
    out = {}
    for c in chunks:
        out.setdefault(c.rule_id, []).append(c)
    return out


# --------------------------------------------------------------------------- #
# Identifier hierarchy
# --------------------------------------------------------------------------- #
def test_ancestor_ids():
    assert ancestor_ids("1.2.3a") == ["1", "1.2", "1.2.3"]
    assert ancestor_ids("1.2.3") == ["1", "1.2"]
    assert ancestor_ids("1.2") == ["1"]
    assert ancestor_ids("1") == []


def test_rule_depth():
    assert rule_depth("1") == 1
    assert rule_depth("1.2") == 2
    assert rule_depth("1.2.3") == 3
    assert rule_depth("1.2.3a") == 3  # letter shares parent depth


# --------------------------------------------------------------------------- #
# Document chunking
# --------------------------------------------------------------------------- #
def test_ids_titles_and_parents():
    chunks = chunk_document(FIXTURE, "CR")
    by_id = _by_id(chunks)

    # No chapter chunks; only numbered rules.
    assert "1" not in by_id and "2" not in by_id

    base = by_id["1.0.1"][0]
    assert base.chunk_id == "CR-1.0.1"
    assert base.doc == "CR"
    assert base.parent_ids == ["CR-1", "CR-1.0"]
    assert base.title == "General"  # nearest section heading

    letter = by_id["1.0.1a"][0]
    assert letter.chunk_id == "CR-1.0.1a"
    assert letter.parent_ids == ["CR-1", "CR-1.0", "CR-1.0.1"]
    assert letter.title == "General"

    # Section chunk's own title is the chapter.
    section = by_id["1.0"][0]
    assert section.parent_ids == ["CR-1"]
    assert section.title == "Game Concepts"

    # Second chapter's section.
    deep = by_id["2.1.3"][0]
    assert deep.parent_ids == ["CR-2", "CR-2.1"]
    assert deep.title == "Attacks"


def test_short_rule_gets_parent_context_window():
    chunks = chunk_document(FIXTURE, "CR")
    base = _by_id(chunks)["1.0.1"][0]
    assert base.char_count < 200
    # context comes from the nearest ancestor with text (section 1.0 -> "General")
    assert base.context_window == "General"

    letter = _by_id(chunks)["1.0.1a"][0]
    assert letter.context_window == "A short base rule."  # parent 1.0.1


def test_long_rule_is_split_keeping_rule_id():
    chunks = chunk_document(FIXTURE, "CR")
    parts = _by_id(chunks)["1.0.2"]
    assert len(parts) > 1
    assert [c.chunk_id for c in parts] == [f"CR-1.0.2-p{i}" for i in range(1, len(parts) + 1)]
    # rule_id stays the same for citation; each piece under the limit.
    assert all(c.rule_id == "1.0.2" for c in parts)
    assert all(c.char_count <= MAX_CHARS for c in parts)
    # Split pieces do not carry a context window.
    assert all(c.context_window is None for c in parts)


def test_char_count_is_derived_from_text():
    c = Chunk(chunk_id="CR-9.9", doc="CR", rule_id="9.9", text="hello world")
    assert c.char_count == len("hello world")


def test_split_text_respects_boundaries():
    assert split_text("short") == ["short"]
    pieces = split_text(_LONG_BODY, max_chars=500)
    assert len(pieces) > 1
    assert all(len(p) <= 500 for p in pieces)


# --------------------------------------------------------------------------- #
# Card chunking
# --------------------------------------------------------------------------- #
def test_duplicate_rule_numbers_get_unique_chunk_ids():
    # A document that reuses a rule number across two sections (as the CR does
    # with its Acknowledgments back-matter) must still yield unique chunk_ids.
    md = (
        "# 2 Object Properties\n\n**2.1** Color\n\n"
        "# 2 Acknowledgments\n\n**2.1** Community Contributors listed here.\n"
    )
    chunks = chunk_document(md, "CR")
    ids = [c.chunk_id for c in chunks]
    assert ids == ["CR-2.1", "CR-2.1-dup2"]
    assert len(ids) == len(set(ids))
    # rule_id is preserved for both (citation still resolves to 2.1).
    assert all(c.rule_id == "2.1" for c in chunks)


def test_card_chunks_have_unique_slug_ids():
    cards = [
        {"name": "Sink Below", "pitch": 3, "type_text": "Ninja Action",
         "text": "Instant - Draw a card."},
        {"name": "Sink Below", "pitch": 2, "type_text": "Ninja Action",
         "text": "Instant - Draw a card."},
        {"name": "Command and Conquer!", "pitch": None, "type_text": "Warrior Action",
         "text": "Go again"},
    ]
    chunks = chunk_cards(cards)
    ids = [c.chunk_id for c in chunks]

    assert ids == ["CARD-sink-below", "CARD-sink-below-2", "CARD-command-and-conquer"]
    assert all(c.doc == "CARD" for c in chunks)
    assert chunks[0].rule_id == "Sink Below"
    assert "Ninja Action" in chunks[0].text
    assert "pitch 3" in chunks[0].text
