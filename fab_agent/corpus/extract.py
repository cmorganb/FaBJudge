"""Normalization of the raw Flesh and Blood corpus into structured markdown.

The three rules documents (Comprehensive Rules, Procedures and Penalty Guide,
Tournament Rules and Policy) are published as plain text. This module turns
them into clean markdown while **preserving the hierarchical rule identifiers**
(e.g. ``1.2.3`` / ``1.2.3a``) that the retrieval layer uses as citation
anchors.

The normalizer is deliberately robust to artifacts that appear in
PDF-extracted text even though the current fabtcg.com text files are already
clean: BOM / latin-1 encodings, hard-wrapped paragraphs, hyphenation at line
breaks, and repeated page headers / footers / page numbers. On already-clean
input these steps are effectively no-ops.

The card database (JSON) is normalized separately by :func:`normalize_cards`.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Optional

# --------------------------------------------------------------------------- #
# Identifier grammar
# --------------------------------------------------------------------------- #
#: A hierarchical rule identifier at the start of a line: at least one dot
#: (Chapter.Section[.Rule ...]) with an optional trailing sub-rule letter and
#: an optional trailing period, e.g. ``1.0``, ``1.2.3``, ``1.2.3a``, ``1.2.3.``.
RULE_ID_RE = re.compile(r"^(?P<id>\d+(?:\.\d+)+[a-z]?)\.?(?=\s|$)")

#: A bare chapter heading: a single integer followed by a short Title Case name
#: (e.g. ``1 Preface``, ``2 Game Concepts``). Prose that merely begins with a
#: number is excluded by the short/Title-case checks in :func:`_match_chapter`.
CHAPTER_RE = re.compile(r"^(?P<id>\d+)\s+(?P<title>\S.*)$")

#: Broad "looks like a rule id" probe used only for the validation pass. Allows
#: leading whitespace and trailing ``.``/``)`` so we can flag identifiers that
#: the strict segmenter did *not* anchor (e.g. accidentally indented ones).
LOOKS_LIKE_ID_RE = re.compile(r"^\s*(?P<id>\d+(?:\.\d+)+[a-z]?)[.)]?(?=\s|$)")

#: Standalone page-number / running-footer lines to always drop.
PAGE_NUMBER_RE = re.compile(r"^(page\s+)?\d+(\s+of\s+\d+)?$", re.IGNORECASE)

#: Chapter titles that are non-rule back-matter (credits / copyright). These
#: sections, and everything under them, are dropped from the extracted corpus
#: so they never become spurious "rules" — e.g. the CR's "Acknowledgments"
#: block reuses rule numbers (2.0/2.1) for its staff and contributor lists.
NON_RULE_SECTIONS = {"acknowledgments", "acknowledgements", "credits", "copyright"}

#: Characters that terminate a sentence; a line NOT ending in one of these was
#: (heuristically) cut mid-sentence by hard wrapping and should be re-joined.
_TERMINAL_CHARS = set('.!?:;"\')”’')


def decode_bytes(raw: bytes) -> str:
    """Decode source bytes, handling a UTF-8 BOM with a latin-1 fallback."""
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def _ends_sentence(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return True
    return stripped[-1] in _TERMINAL_CHARS


def _join_wrapped(prev: str, nxt: str) -> str:
    """Join a wrapped continuation onto ``prev``, repairing hyphenation.

    ``"inevita-" + "ble"`` -> ``"inevitable"``; otherwise the two fragments are
    joined with a single space.
    """
    prev = prev.rstrip()
    nxt = nxt.strip()
    if len(prev) >= 2 and prev.endswith("-") and prev[-2].isalpha() and nxt[:1].islower():
        return prev[:-1] + nxt
    return f"{prev} {nxt}"


def _match_chapter(line: str) -> Optional[re.Match]:
    """Return a chapter-heading match, or None if the line is really prose."""
    m = CHAPTER_RE.match(line)
    if not m:
        return None
    title = m.group("title").strip()
    if not title[:1].isupper():
        return None
    if len(line) > 60 or len(title.split()) > 6:
        return None
    if title[-1] in ".,;":
        return None
    return m


# --------------------------------------------------------------------------- #
# Line-level normalization
# --------------------------------------------------------------------------- #
def _detect_repeated_lines(lines: list[str], threshold: int) -> set[str]:
    """Identify short lines that repeat often enough to be headers/footers.

    Rule-identifier lines are never treated as headers, so real content is
    preserved even if a short rule happens to recur.
    """
    counts = Counter(l.strip() for l in lines if l.strip())
    repeated = set()
    for text, n in counts.items():
        if n < threshold:
            continue
        if len(text) > 60:
            continue
        if RULE_ID_RE.match(text):
            continue
        repeated.add(text)
    return repeated


def normalize_lines(text: str, *, repeat_threshold: int = 5) -> tuple[list[str], list[str]]:
    """Clean raw document text into a list of lines.

    Unifies line endings, strips trailing whitespace, collapses runs of blank
    lines, and removes page-number lines and repeated header/footer lines.

    Returns ``(lines, removed)`` where ``removed`` records the distinct
    header/footer strings that were dropped, for auditing.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = [l.rstrip() for l in text.split("\n")]

    repeated = _detect_repeated_lines(raw_lines, repeat_threshold)
    removed: list[str] = []

    kept: list[str] = []
    blank_run = False
    for line in raw_lines:
        stripped = line.strip()
        if stripped and (stripped in repeated or PAGE_NUMBER_RE.match(stripped)):
            removed.append(stripped)
            continue
        if not stripped:
            if blank_run:
                continue  # collapse consecutive blanks
            blank_run = True
            kept.append("")
            continue
        blank_run = False
        kept.append(line)

    # Trim leading/trailing blank lines.
    while kept and not kept[0]:
        kept.pop(0)
    while kept and not kept[-1]:
        kept.pop()
    return kept, sorted(set(removed))


# --------------------------------------------------------------------------- #
# Segmentation into identifier-anchored records
# --------------------------------------------------------------------------- #
@dataclass
class Record:
    """One structural unit of a document."""

    identifier: Optional[str]
    kind: str  # 'chapter' | 'rule' | 'preamble'
    title: Optional[str] = None
    paragraphs: list[str] = field(default_factory=list)


def segment(lines: Iterable[str]) -> tuple[list[Record], list[str]]:
    """Split normalized lines into identifier-anchored records.

    A new record begins at every rule identifier or chapter heading. Lines with
    no identifier are attached to the current record: joined into the current
    paragraph when they are hard-wrap continuations (previous fragment did not
    end a sentence), otherwise kept as a new paragraph (e.g. ``Example:`` lines
    or preface prose). Continuations never cross an identifier boundary.
    """
    lines = list(lines)
    records: list[Record] = []
    cur: Optional[Record] = None
    pending_blank = True  # so the first prose line is never a "continuation"

    for line in lines:
        stripped = line.strip()
        if not stripped:
            pending_blank = True
            continue

        m = RULE_ID_RE.match(line)
        if m:
            text = line[m.end():].strip()
            cur = Record(identifier=m.group("id"), kind="rule",
                         paragraphs=[text] if text else [])
            records.append(cur)
            pending_blank = False
            continue

        chapter = _match_chapter(line)
        if chapter:
            cur = Record(identifier=chapter.group("id"), kind="chapter",
                         title=chapter.group("title").strip())
            records.append(cur)
            pending_blank = False
            continue

        # Prose line with no identifier.
        mergeable = (
            cur is not None
            and cur.kind in ("rule", "preamble")
            and cur.paragraphs
            and not pending_blank
            and not _ends_sentence(cur.paragraphs[-1])
        )
        if mergeable:
            cur.paragraphs[-1] = _join_wrapped(cur.paragraphs[-1], stripped)
        else:
            if cur is None:
                cur = Record(identifier=None, kind="preamble")
                records.append(cur)
            cur.paragraphs.append(stripped)
        pending_blank = False

    return records, []


def _drop_back_matter(records: list[Record]) -> tuple[list[Record], list[str]]:
    """Drop non-rule back-matter chapters (credits/copyright) and their content.

    A matching chapter heading starts a "dropping" region that continues until
    the next chapter heading. Returns ``(kept_records, dropped_titles)``.
    """
    kept: list[Record] = []
    dropped: list[str] = []
    dropping = False
    for rec in records:
        if rec.kind == "chapter":
            if (rec.title or "").strip().lower() in NON_RULE_SECTIONS:
                dropping = True
                dropped.append(rec.title or "")
                continue
            dropping = False
        if dropping:
            continue
        kept.append(rec)
    return kept, dropped


def _validate_identifiers(lines: list[str], records: list[Record]) -> list[str]:
    """Flag lines that look like rule ids but were not anchored as records."""
    captured = {r.identifier for r in records if r.kind == "rule"}
    warnings: list[str] = []
    for i, line in enumerate(lines, start=1):
        m = LOOKS_LIKE_ID_RE.match(line)
        if not m:
            continue
        candidate = m.group("id")
        # A cleanly anchored line starts at column 0 with the identifier.
        anchored = bool(RULE_ID_RE.match(line)) and candidate in captured
        if not anchored:
            warnings.append(f"line {i}: uncaptured id-like line: {line.strip()!r}")
    return warnings


# --------------------------------------------------------------------------- #
# Markdown assembly
# --------------------------------------------------------------------------- #
def _rule_depth_label(identifier: str) -> str:
    letter = identifier[-1].isalpha()
    ncomp = identifier.rstrip("abcdefghijklmnopqrstuvwxyz").count(".") + 1
    return f"{ncomp}-level" + ("+letter" if letter else "")


def build_markdown(records: list[Record], *, meta: dict[str, str]) -> str:
    """Render records to markdown with a YAML metadata block."""
    out: list[str] = ["---"]
    for key, value in meta.items():
        out.append(f"{key}: {value}")
    out.append("---")
    out.append("")

    for rec in records:
        if rec.kind == "chapter":
            out.append(f"# {rec.identifier} {rec.title}")
            out.append("")
            for para in rec.paragraphs:
                out.append(para)
                out.append("")
        elif rec.kind == "preamble":
            for para in rec.paragraphs:
                out.append(para)
                out.append("")
        else:  # rule
            first = rec.paragraphs[0] if rec.paragraphs else ""
            out.append(f"**{rec.identifier}** {first}".rstrip())
            out.append("")
            for para in rec.paragraphs[1:]:
                out.append(para)
                out.append("")

    return "\n".join(out).rstrip() + "\n"


@dataclass
class ExtractResult:
    markdown: str
    records: list[Record]
    warnings: list[str]
    removed_lines: list[str]
    rule_count: int
    depth_counts: dict[str, int]
    dropped_sections: list[str]


def extract_rules_document(
    raw: bytes,
    *,
    source_file: str,
    document_title: str,
    version: Optional[str] = None,
    processed_at: Optional[datetime] = None,
    repeat_threshold: int = 5,
) -> ExtractResult:
    """Full pipeline: raw bytes -> normalized, structured markdown + stats."""
    processed_at = processed_at or datetime.now(timezone.utc)

    text = decode_bytes(raw)
    lines, removed = normalize_lines(text, repeat_threshold=repeat_threshold)
    records, _ = segment(lines)
    # Validate against the full segmentation, then drop non-rule back-matter so
    # its (intentionally discarded) identifiers are not flagged as warnings.
    warnings = _validate_identifiers(lines, records)
    records, dropped_sections = _drop_back_matter(records)

    rules = [r for r in records if r.kind == "rule"]
    depth_counts: dict[str, int] = {}
    for r in rules:
        label = _rule_depth_label(r.identifier)
        depth_counts[label] = depth_counts.get(label, 0) + 1

    meta = {
        "document": document_title,
        "source_file": source_file,
        "version": version or "unknown",
        "processed_at": processed_at.isoformat().replace("+00:00", "Z"),
        "rules_detected": str(len(rules)),
    }
    markdown = build_markdown(records, meta=meta)

    return ExtractResult(
        markdown=markdown,
        records=records,
        warnings=warnings,
        removed_lines=removed,
        rule_count=len(rules),
        depth_counts=depth_counts,
        dropped_sections=dropped_sections,
    )


# --------------------------------------------------------------------------- #
# Card database normalization
# --------------------------------------------------------------------------- #
def _printing_key(p: dict) -> tuple:
    return (p.get("set_id"), p.get("card_id"), p.get("edition"), p.get("rarity"))


def normalize_cards(raw: bytes) -> list[dict]:
    """Normalize the raw card JSON into a deduplicated list of card records.

    Each record carries name, pitch value, card type(s), full rules text, and
    set/printing identifiers. Reprints are collapsed by ``(name, pitch, text)``,
    merging their printings.
    """
    data = json.loads(decode_bytes(raw))

    by_key: dict[tuple, dict] = {}
    order: list[tuple] = []

    for card in data:
        name = (card.get("name") or "").strip()
        pitch_raw = str(card.get("pitch") or "").strip()
        pitch = int(pitch_raw) if pitch_raw.isdigit() else None
        types = card.get("types") or []
        text = (card.get("functional_text") or "").strip()

        printings = [
            {
                "set_id": p.get("set_id"),
                "card_id": p.get("id"),
                "edition": p.get("edition"),
                "rarity": p.get("rarity"),
            }
            for p in (card.get("printings") or [])
        ]

        key = (name.lower(), pitch, text)
        if key in by_key:
            existing = by_key[key]
            seen = {_printing_key(p) for p in existing["printings"]}
            for p in printings:
                if _printing_key(p) not in seen:
                    existing["printings"].append(p)
                    seen.add(_printing_key(p))
            continue

        by_key[key] = {
            "name": name,
            "pitch": pitch,
            "types": types,
            "type_text": card.get("type_text", ""),
            "text": text,
            "printings": printings,
        }
        order.append(key)

    return [by_key[k] for k in order]
