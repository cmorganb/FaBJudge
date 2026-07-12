"""Split the normalized corpus into citable chunks — one per numbered rule.

Consumes the markdown produced by :mod:`fab_agent.corpus.extract`
(``data/processed/{cr,ppg,trp}.md``) plus ``cards.jsonl`` and emits
``data/processed/chunks.jsonl``. Each chunk is anchored to a rule identifier so
a downstream ruling can cite it precisely (e.g. ``CR-1.2.3a``).

Run directly to (re)build the chunk store and print a report:

    uv run python -m fab_agent.corpus.chunker
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------------------- #
# Tunables
# --------------------------------------------------------------------------- #
MAX_CHARS = 1500  # rules longer than this are split into -p1/-p2 sub-chunks
SHORT_CHARS = 200  # rules shorter than this also carry a parent context window

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

#: processed markdown filename -> document code
DOC_FILES: dict[str, str] = {"cr.md": "CR", "ppg.md": "PPG", "trp.md": "TRP"}

_CHAPTER_RE = re.compile(r"^# (?P<id>\d+(?:\.\d+)*)\s+(?P<title>.+)$")
_RULE_RE = re.compile(r"^\*\*(?P<id>\d+(?:\.\d+)*[a-z]?)\*\*\s?(?P<text>.*)$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


# --------------------------------------------------------------------------- #
# Chunk model
# --------------------------------------------------------------------------- #
class Chunk(BaseModel):
    """A single citable unit of the corpus."""

    chunk_id: str
    doc: Literal["CR", "PPG", "TRP", "CARD"]
    rule_id: str
    title: str = ""
    text: str
    parent_ids: list[str] = Field(default_factory=list)
    char_count: int = 0
    #: For very short rules, the parent rule's text, to aid retrieval. The
    #: citation anchor remains this chunk's own ``rule_id``.
    context_window: Optional[str] = None

    @model_validator(mode="after")
    def _set_char_count(self) -> "Chunk":
        self.char_count = len(self.text)
        return self


# --------------------------------------------------------------------------- #
# Identifier hierarchy
# --------------------------------------------------------------------------- #
def _numeric_base(rule_id: str) -> str:
    """Strip a trailing sub-rule letter: ``1.2.3a`` -> ``1.2.3``."""
    return rule_id[:-1] if rule_id and rule_id[-1].isalpha() else rule_id


def rule_depth(rule_id: str) -> int:
    """Number of numeric components (a sub-rule letter shares its parent's depth)."""
    return _numeric_base(rule_id).count(".") + 1


def ancestor_ids(rule_id: str) -> list[str]:
    """Ancestor rule ids, outermost first.

    ``1.2.3a`` -> ``["1", "1.2", "1.2.3"]``; ``1.2.3`` -> ``["1", "1.2"]``.
    """
    has_letter = bool(rule_id) and rule_id[-1].isalpha()
    base = _numeric_base(rule_id)
    parts = base.split(".")
    anc = [".".join(parts[:i]) for i in range(1, len(parts))]
    if has_letter:
        anc.append(base)  # the numeric rule is the parent of its lettered sub-rule
    return anc


# --------------------------------------------------------------------------- #
# Markdown parsing
# --------------------------------------------------------------------------- #
def _strip_front_matter(md_text: str) -> str:
    if md_text.startswith("---"):
        close = md_text.find("\n---", 3)
        if close != -1:
            nl = md_text.find("\n", close + 1)
            if nl != -1:
                return md_text[nl + 1:]
    return md_text


def parse_markdown(md_text: str) -> list[dict]:
    """Parse processed markdown into an ordered list of chapter / rule nodes."""
    nodes: list[dict] = []
    current_rule: Optional[dict] = None

    for line in _strip_front_matter(md_text).split("\n"):
        chapter = _CHAPTER_RE.match(line)
        if chapter:
            nodes.append({
                "kind": "chapter",
                "id": chapter.group("id"),
                "text": chapter.group("title").strip(),
            })
            current_rule = None
            continue

        rule = _RULE_RE.match(line)
        if rule:
            first = rule.group("text").strip()
            current_rule = {
                "kind": "rule",
                "id": rule.group("id"),
                "paras": [first] if first else [],
            }
            nodes.append(current_rule)
            continue

        if not line.strip():
            continue

        # A non-blank line with no marker is a continuation paragraph.
        if current_rule is not None:
            current_rule["paras"].append(line.strip())

    for node in nodes:
        if node["kind"] == "rule":
            node["text"] = "\n\n".join(node["paras"])
    return nodes


# --------------------------------------------------------------------------- #
# Splitting helpers
# --------------------------------------------------------------------------- #
def _hard_split(text: str, max_chars: int) -> list[str]:
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def _split_sentences(paragraph: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    cur = ""
    for sentence in _SENTENCE_SPLIT_RE.split(paragraph):
        if len(sentence) > max_chars:
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.extend(_hard_split(sentence, max_chars))
            continue
        if cur and len(cur) + 1 + len(sentence) > max_chars:
            pieces.append(cur)
            cur = sentence
        else:
            cur = sentence if not cur else f"{cur} {sentence}"
    if cur:
        pieces.append(cur)
    return pieces


def split_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split overlong text on paragraph, then sentence, then hard boundaries."""
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    cur = ""
    for para in text.split("\n\n"):
        if len(para) > max_chars:
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.extend(_split_sentences(para, max_chars))
            continue
        if cur and len(cur) + 2 + len(para) > max_chars:
            pieces.append(cur)
            cur = para
        else:
            cur = para if not cur else f"{cur}\n\n{para}"
    if cur:
        pieces.append(cur)
    return pieces


def _short_title(text: str) -> str:
    head = text.strip().split("\n", 1)[0]
    if len(head) <= 80:
        return head
    return head[:80].rsplit(" ", 1)[0] + "…"


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def ensure_unique_ids(chunks: list[Chunk]) -> list[Chunk]:
    """Guarantee globally unique ``chunk_id`` values (the citation anchor).

    Some source documents reuse rule numbers across unrelated sections — e.g.
    the CR's back-matter "Acknowledgments" block reuses ``2.0``/``2.1`` already
    used by chapter 2. Collisions are disambiguated deterministically with a
    ``-dupN`` suffix while the ``rule_id`` is left untouched.
    """
    seen: dict[str, int] = {}
    for c in chunks:
        n = seen.get(c.chunk_id, 0) + 1
        seen[c.chunk_id] = n
        if n > 1:
            c.chunk_id = f"{c.chunk_id}-dup{n}"
    return chunks


def chunk_document(md_text: str, doc: str) -> list[Chunk]:
    """Produce one chunk per numbered rule for a single document."""
    nodes = parse_markdown(md_text)
    rule_text = {n["id"]: n["text"] for n in nodes if n["kind"] == "rule"}

    chunks: list[Chunk] = []
    current_chapter = ""
    section_titles: dict[str, str] = {}

    for node in nodes:
        if node["kind"] == "chapter":
            current_chapter = node["text"]
            continue

        rid = node["id"]
        text = node["text"]
        depth = rule_depth(rid)

        if depth == 2:
            section_titles[rid] = _short_title(text)
            title = current_chapter
        elif depth > 2:
            section = ".".join(_numeric_base(rid).split(".")[:2])
            title = section_titles.get(section, current_chapter)
        else:
            title = current_chapter

        anc = ancestor_ids(rid)
        parent_ids = [f"{doc}-{a}" for a in anc]

        context_window = None
        if len(text) < SHORT_CHARS:
            for a in reversed(anc):
                if rule_text.get(a):
                    context_window = rule_text[a]
                    break

        pieces = split_text(text)
        if len(pieces) == 1:
            chunks.append(Chunk(
                chunk_id=f"{doc}-{rid}", doc=doc, rule_id=rid, title=title,
                text=text, parent_ids=parent_ids, context_window=context_window,
            ))
        else:
            for i, piece in enumerate(pieces, start=1):
                chunks.append(Chunk(
                    chunk_id=f"{doc}-{rid}-p{i}", doc=doc, rule_id=rid,
                    title=title, text=piece, parent_ids=parent_ids,
                ))

    return ensure_unique_ids(chunks)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "card"


def chunk_cards(cards: list[dict]) -> list[Chunk]:
    """Produce one chunk per unique card, with collision-free ``CARD-<slug>`` ids."""
    used: set[str] = set()
    chunks: list[Chunk] = []
    for card in cards:
        base = _slugify(card.get("name", ""))
        slug = base
        n = 2
        while slug in used:
            slug = f"{base}-{n}"
            n += 1
        used.add(slug)

        header = card.get("name", "")
        if card.get("type_text"):
            header += f" — {card['type_text']}"
        if card.get("pitch") is not None:
            header += f" (pitch {card['pitch']})"
        body = card.get("text", "")
        text = f"{header}\n\n{body}" if body else header

        chunks.append(Chunk(
            chunk_id=f"CARD-{slug}", doc="CARD", rule_id=card.get("name", ""),
            title=card.get("name", ""), text=text,
        ))
    return chunks


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_all_chunks() -> tuple[list[Chunk], dict[str, int]]:
    """Chunk every processed document and the card database."""
    all_chunks: list[Chunk] = []
    counts: dict[str, int] = {}

    for filename, doc in DOC_FILES.items():
        path = PROCESSED_DIR / filename
        chunks = chunk_document(path.read_text("utf-8"), doc)
        counts[doc] = len(chunks)
        all_chunks.extend(chunks)

    cards_path = PROCESSED_DIR / "cards.jsonl"
    cards = [json.loads(line) for line in cards_path.read_text("utf-8").splitlines() if line.strip()]
    card_chunks = chunk_cards(cards)
    counts["CARD"] = len(card_chunks)
    all_chunks.extend(card_chunks)

    return all_chunks, counts


def write_chunks(chunks: list[Chunk], path: Path = CHUNKS_PATH) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(chunk.model_dump_json() + "\n")


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="Seed for the random sample.")
    parser.add_argument("--sample", type=int, default=5, help="Number of chunks to print.")
    args = parser.parse_args(argv)

    missing = [f for f in list(DOC_FILES) + ["cards.jsonl"] if not (PROCESSED_DIR / f).exists()]
    if missing:
        print(f"Missing processed input(s): {', '.join(missing)}. Run scripts/extract_corpus.py first.")
        return 1

    chunks, counts = build_all_chunks()
    write_chunks(chunks)

    split_chunks = sum(1 for c in chunks if re.search(r"-p\d+$", c.chunk_id))
    with_context = sum(1 for c in chunks if c.context_window)

    print("=" * 68)
    print(f"Chunking complete → {CHUNKS_PATH.relative_to(ROOT)}")
    print("=" * 68)
    print("\nChunk counts:")
    for key, val in counts.items():
        print(f"  • {key}: {val}")
    print(f"  Total: {len(chunks)}")
    print(f"\n  Long rules split into sub-chunks: {split_chunks}")
    print(f"  Short rules given a parent context window: {with_context}")

    print(f"\n{args.sample} random chunks (spot-check):")
    print("-" * 68)
    rng = random.Random(args.seed)
    for c in rng.sample(chunks, min(args.sample, len(chunks))):
        snippet = c.text if len(c.text) <= 300 else c.text[:300] + " …"
        print(f"[{c.chunk_id}]  rule_id={c.rule_id}  chars={c.char_count}")
        print(f"    title:   {c.title}")
        print(f"    parents: {c.parent_ids}")
        if c.context_window:
            ctx = c.context_window if len(c.context_window) <= 120 else c.context_window[:120] + " …"
            print(f"    context: {ctx}")
        print(f"    text:    {snippet}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
