"""Tests for corpus normalization on small synthetic fixtures (no real files).

Covers the artifacts the normalizer must survive: hard-wrapped rule text,
repeated header/footer lines, nested rule identifiers, BOM / latin-1 encodings,
hyphenation at line breaks, and card de-duplication.
"""

import json

from fab_agent.corpus.extract import (
    decode_bytes,
    extract_rules_document,
    normalize_cards,
    normalize_lines,
    segment,
)


def _rules(raw: bytes):
    """Extract and return {identifier: Record} for rule records only."""
    result = extract_rules_document(raw, source_file="fixture.txt",
                                    document_title="Fixture", version="v0")
    return result, {r.identifier: r for r in result.records if r.kind == "rule"}


# --------------------------------------------------------------------------- #
# Hard-wrapped rule text
# --------------------------------------------------------------------------- #
def test_hard_wrapped_paragraph_is_rejoined():
    raw = (
        "1.1 Combat\n"
        "1.1.1 When an attack hits, the defending hero loses life equal to the\n"
        "attack's power unless an effect modifies that amount.\n"
        "1.1.2 A separate, later rule.\n"
    ).encode("utf-8")
    _, rules = _rules(raw)

    assert set(rules) >= {"1.1", "1.1.1", "1.1.2"}
    assert rules["1.1.1"].paragraphs == [
        "When an attack hits, the defending hero loses life equal to the "
        "attack's power unless an effect modifies that amount."
    ]
    # Never joined across the next identifier.
    assert "separate" not in rules["1.1.1"].paragraphs[0]
    assert rules["1.1.2"].paragraphs[0] == "A separate, later rule."


def test_example_line_is_not_merged_into_rule_sentence():
    raw = (
        "1.1 A restriction applies here.\n"
        "Example: like this one.\n"
        "1.2 Next.\n"
    ).encode("utf-8")
    _, rules = _rules(raw)
    assert rules["1.1"].paragraphs == [
        "A restriction applies here.",
        "Example: like this one.",
    ]


def test_hyphenation_at_line_break_is_repaired():
    raw = ("1.1 A hyphen-\nation example word.\n").encode("utf-8")
    _, rules = _rules(raw)
    assert rules["1.1"].paragraphs[0] == "A hyphenation example word."


# --------------------------------------------------------------------------- #
# Repeated headers / footers / page numbers
# --------------------------------------------------------------------------- #
def test_repeated_headers_and_page_numbers_removed():
    parts = []
    for i in range(1, 7):
        parts.append("FLESH AND BLOOD COMPREHENSIVE RULES")  # running header
        parts.append(f"1.{i} Section {i} content ends here.")
        parts.append(str(i))  # page number
    text = "\n".join(parts)

    lines, removed = normalize_lines(text, repeat_threshold=5)

    assert "FLESH AND BLOOD COMPREHENSIVE RULES" in removed
    assert "FLESH AND BLOOD COMPREHENSIVE RULES" not in lines
    # No bare page-number lines survive.
    assert not any(l.strip().isdigit() for l in lines)
    # Rule content is preserved.
    assert any(l.startswith("1.3 Section 3") for l in lines)


def test_repeated_line_that_is_a_rule_id_is_not_removed():
    # A short line is only a header candidate if it is NOT a rule identifier.
    text = "\n".join(["1.1 Repeated rule text."] * 6)
    lines, removed = normalize_lines(text, repeat_threshold=5)
    assert removed == []
    assert lines  # content kept


# --------------------------------------------------------------------------- #
# Nested rule identifiers
# --------------------------------------------------------------------------- #
def test_nested_identifiers_each_anchor_their_own_record():
    raw = (
        "1 Game Concepts\n"
        "1.0 General\n"
        "1.0.1 The base rule.\n"
        "1.0.1a A lettered sub-rule.\n"
        "1.0.1b Another sub-rule.\n"
        "1.0.2 The next rule.\n"
    ).encode("utf-8")
    result, rules = _rules(raw)

    assert [r.identifier for r in result.records if r.kind == "rule"] == [
        "1.0", "1.0.1", "1.0.1a", "1.0.1b", "1.0.2",
    ]
    chapters = [r for r in result.records if r.kind == "chapter"]
    assert chapters[0].identifier == "1"
    assert chapters[0].title == "Game Concepts"
    assert rules["1.0.1a"].paragraphs[0] == "A lettered sub-rule."


def test_trailing_period_identifier_is_recognized():
    raw = "1.2.3. A rule written with a trailing period.\n".encode("utf-8")
    _, rules = _rules(raw)
    assert "1.2.3" in rules
    assert rules["1.2.3"].paragraphs[0] == "A rule written with a trailing period."


# --------------------------------------------------------------------------- #
# Encodings: BOM + latin-1
# --------------------------------------------------------------------------- #
def test_bom_encoded_file_decodes_cleanly():
    raw = "1.2.3 A rule with an em dash — and a BOM prefix.\n".encode("utf-8-sig")
    assert raw[:3] == b"\xef\xbb\xbf"

    result, rules = _rules(raw)
    assert "1.2.3" in rules
    assert rules["1.2.3"].paragraphs[0].startswith("A rule with an em dash")
    # BOM must not leak into identifiers or output.
    assert "﻿" not in result.markdown
    assert "﻿" not in rules["1.2.3"].identifier


def test_latin1_fallback_decoding():
    raw = "1.1 Café rule text.".encode("latin-1")
    assert decode_bytes(raw) == "1.1 Café rule text."


# --------------------------------------------------------------------------- #
# Output structure
# --------------------------------------------------------------------------- #
def test_markdown_has_metadata_block():
    raw = "1.1 A rule.\n".encode("utf-8")
    result = extract_rules_document(
        raw, source_file="cr.txt", document_title="CR", version="v2.6.0"
    )
    md = result.markdown
    assert md.startswith("---\n")
    assert "document: CR" in md
    assert "source_file: cr.txt" in md
    assert "version: v2.6.0" in md
    assert "processed_at:" in md
    assert "rules_detected: 1" in md
    assert "**1.1** A rule." in md


def test_segment_returns_no_crossing_and_orders_records():
    lines, _ = normalize_lines("1.1 First.\n1.2 Second.\n")
    records, _ = segment(lines)
    assert [r.identifier for r in records] == ["1.1", "1.2"]


# --------------------------------------------------------------------------- #
# Card normalization / de-duplication
# --------------------------------------------------------------------------- #
def test_card_dedup_by_name_pitch_text_merges_printings():
    data = [
        {
            "name": "Sink Below", "pitch": "3", "types": ["Ninja", "Action"],
            "functional_text": "Instant - Draw a card.",
            "printings": [{"set_id": "WTR", "id": "WTR100", "edition": "A", "rarity": "C"}],
        },
        {  # reprint: same name/pitch/text, different printing
            "name": "Sink Below", "pitch": "3", "types": ["Ninja", "Action"],
            "functional_text": "Instant - Draw a card.",
            "printings": [{"set_id": "WTR", "id": "WTR100", "edition": "U", "rarity": "C"}],
        },
        {  # different pitch -> distinct card
            "name": "Sink Below", "pitch": "2", "types": ["Ninja", "Action"],
            "functional_text": "Instant - Draw a card.",
            "printings": [{"set_id": "WTR", "id": "WTR101", "edition": "A", "rarity": "C"}],
        },
    ]
    cards = normalize_cards(json.dumps(data).encode("utf-8"))

    assert len(cards) == 2
    pitch3 = next(c for c in cards if c["pitch"] == 3)
    assert pitch3["name"] == "Sink Below"
    assert pitch3["types"] == ["Ninja", "Action"]
    assert len(pitch3["printings"]) == 2  # both printings merged
    assert {p["edition"] for p in pitch3["printings"]} == {"A", "U"}


def test_card_non_numeric_pitch_becomes_none():
    data = [{"name": "Kano", "pitch": "", "types": ["Wizard", "Hero"],
             "functional_text": "", "printings": []}]
    cards = normalize_cards(json.dumps(data).encode("utf-8"))
    assert cards[0]["pitch"] is None
    assert cards[0]["name"] == "Kano"
