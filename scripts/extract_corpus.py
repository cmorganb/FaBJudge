#!/usr/bin/env python3
"""Normalize the raw fab_agent corpus into ``data/processed/``.

Reads the three rules documents and the card database from ``data/raw/`` and
produces:

* ``cr.md`` / ``ppg.md`` / ``trp.md`` — clean, citation-anchored markdown.
* ``<doc>.extract.log`` — any lines that look like rule ids but were not
  captured, plus the header/footer lines that were stripped.
* ``cards.jsonl`` — one normalized, de-duplicated card per line.

Usage:
    uv run python scripts/extract_corpus.py
    uv run python scripts/extract_corpus.py --seed 0   # reproducible sample
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

from fab_agent.corpus.extract import extract_rules_document, normalize_cards

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MANIFEST_PATH = RAW_DIR / "manifest.json"

#: source filename -> (human title, output markdown filename)
RULE_DOCS = {
    "en-fab-cr.txt": ("Comprehensive Rules (CR)", "cr.md"),
    "en-fab-ppg.txt": ("Procedures and Penalty Guide (PPG)", "ppg.md"),
    "en-fab-trp.txt": ("Tournament Rules and Policy (TRP)", "trp.md"),
}
CARD_FILE = "card.json"


def _load_versions() -> dict[str, str]:
    """Map raw filename -> version label from the download manifest."""
    if not MANIFEST_PATH.exists():
        return {}
    manifest = json.loads(MANIFEST_PATH.read_text("utf-8"))
    return {
        name: entry.get("version_label") or "unknown"
        for name, entry in manifest.get("files", {}).items()
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed for the random rule sample (default: nondeterministic).",
    )
    parser.add_argument(
        "--sample", type=int, default=5,
        help="Number of random extracted rules to print (default: 5).",
    )
    args = parser.parse_args(argv)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    versions = _load_versions()

    missing = [f for f in list(RULE_DOCS) + [CARD_FILE] if not (RAW_DIR / f).exists()]
    if missing:
        print(
            f"Missing raw source(s): {', '.join(missing)}.\n"
            f"Run scripts/download_corpus.py first.",
            file=sys.stderr,
        )
        return 1

    # ------------------------------------------------------------------ #
    # Rules documents
    # ------------------------------------------------------------------ #
    all_rules: list[tuple[str, str, str]] = []  # (doc, id, text)
    doc_stats: list[tuple[str, int, dict[str, int], int, list[str]]] = []

    for filename, (title, out_name) in RULE_DOCS.items():
        raw = (RAW_DIR / filename).read_bytes()
        result = extract_rules_document(
            raw,
            source_file=filename,
            document_title=title,
            version=versions.get(filename),
        )

        (PROCESSED_DIR / out_name).write_text(result.markdown, encoding="utf-8")

        log_path = PROCESSED_DIR / (out_name.rsplit(".", 1)[0] + ".extract.log")
        log_lines = [f"# Extraction log for {filename}", ""]
        log_lines.append(f"## Stripped header/footer lines ({len(result.removed_lines)})")
        log_lines += result.removed_lines or ["(none)"]
        log_lines += ["", f"## Dropped non-rule sections ({len(result.dropped_sections)})"]
        log_lines += result.dropped_sections or ["(none)"]
        log_lines += ["", f"## Uncaptured id-like lines ({len(result.warnings)})"]
        log_lines += result.warnings or ["(none)"]
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

        for rec in result.records:
            if rec.kind == "rule" and rec.paragraphs and rec.paragraphs[0]:
                all_rules.append((title, rec.identifier, rec.paragraphs[0]))

        doc_stats.append((title, result.rule_count, result.depth_counts,
                          len(result.warnings), result.dropped_sections))

    # ------------------------------------------------------------------ #
    # Card database
    # ------------------------------------------------------------------ #
    cards = normalize_cards((RAW_DIR / CARD_FILE).read_bytes())
    cards_path = PROCESSED_DIR / "cards.jsonl"
    with cards_path.open("w", encoding="utf-8") as fh:
        for card in cards:
            fh.write(json.dumps(card, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ #
    # Report
    # ------------------------------------------------------------------ #
    print("=" * 68)
    print("Corpus normalization complete →", PROCESSED_DIR.relative_to(ROOT))
    print("=" * 68)
    print("\nRules detected per document:")
    for title, count, depths, warns, dropped in doc_stats:
        breakdown = ", ".join(f"{k}: {v}" for k, v in sorted(depths.items()))
        print(f"  • {title}: {count} rules   [{breakdown}]")
        if dropped:
            print(f"      ⃠ dropped non-rule section(s): {', '.join(dropped)}")
        if warns:
            print(f"      ⚠ {warns} id-like line(s) flagged for review (see .extract.log)")

    print(f"\nUnique cards: {len(cards)}")

    print(f"\n{args.sample} random extracted rules (spot-check against the raw .txt):")
    print("-" * 68)
    rng = random.Random(args.seed)
    for doc, ident, text in rng.sample(all_rules, min(args.sample, len(all_rules))):
        snippet = text if len(text) <= 400 else text[:400] + " …"
        print(f"[{doc}] {ident}\n    {snippet}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
