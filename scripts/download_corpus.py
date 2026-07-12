#!/usr/bin/env python3
"""Download and version-pin the raw fab_agent corpus.

Fetches the official Flesh and Blood rules documents from Legend Story Studios
(via fabtcg.com) and the open-source English card database from the
``the-fab-cube/flesh-and-blood-cards`` GitHub repository, saving everything
under ``data/raw/`` and recording provenance in ``data/raw/manifest.json``.

The script is **idempotent**: a file whose on-disk SHA-256 already matches the
manifest is skipped. Use ``--force`` to re-download regardless.

Usage:
    uv run python scripts/download_corpus.py
    uv run python scripts/download_corpus.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Sources — edit these if a document moves. Each entry is downloaded verbatim
# to ``data/raw/<filename>``.
# --------------------------------------------------------------------------- #
#: The three official rules documents are published as plain-text at fixed
#: "latest" URLs on rules.fabtcg.com. The card database is a single English
#: ``card.json`` served raw from GitHub.
SOURCES: dict[str, dict] = {
    "comprehensive_rules": {
        "label": "Comprehensive Rules (CR)",
        "filename": "en-fab-cr.txt",
        "url": "https://rules.fabtcg.com/txt/latest/en-fab-cr.txt",
        "kind": "rules_txt",
    },
    "penalty_guide": {
        "label": "Procedures and Penalty Guide (PPG)",
        "filename": "en-fab-ppg.txt",
        "url": "https://rules.fabtcg.com/txt/latest/en-fab-ppg.txt",
        "kind": "rules_txt",
    },
    "tournament_rules": {
        "label": "Tournament Rules and Policy (TRP)",
        "filename": "en-fab-trp.txt",
        "url": "https://rules.fabtcg.com/txt/latest/en-fab-trp.txt",
        "kind": "rules_txt",
    },
    "card_database": {
        "label": "the-fab-cube/flesh-and-blood-cards — English card.json",
        "filename": "card.json",
        "url": (
            "https://raw.githubusercontent.com/the-fab-cube/"
            "flesh-and-blood-cards/develop/json/english/card.json"
        ),
        "kind": "card_json",
        # Used to resolve a human-readable version label (latest commit).
        "repo": "the-fab-cube/flesh-and-blood-cards",
        "repo_path": "json/english/card.json",
        "repo_ref": "develop",
    },
}

#: Repository root (this file lives in ``<root>/scripts/``).
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "manifest.json"

HTTP_TIMEOUT = httpx.Timeout(60.0, connect=15.0)


# --------------------------------------------------------------------------- #
# Manifest schema
# --------------------------------------------------------------------------- #
class ManifestEntry(BaseModel):
    """Provenance record for a single downloaded file."""

    source_url: str
    download_date: datetime
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    version_label: Optional[str] = None
    bytes: int = Field(ge=0)


class Manifest(BaseModel):
    """Top-level manifest: one entry per file, keyed by filename."""

    generated_at: datetime
    files: dict[str, ManifestEntry] = Field(default_factory=dict)


class DownloadError(RuntimeError):
    """Raised when a source cannot be fetched; message guides the user."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str, url: str, client: httpx.Client) -> httpx.Response:
    """Fetch ``url`` or raise a :class:`DownloadError` with actionable text."""
    try:
        resp = client.get(url, follow_redirects=True)
        resp.raise_for_status()
        return resp
    except httpx.HTTPStatusError as exc:
        raise DownloadError(
            f"Failed to download '{name}' from {url}\n"
            f"  Server responded with HTTP {exc.response.status_code}.\n"
            f"  The document may have moved — check the URL in a browser and "
            f"update the SOURCES dict in {Path(__file__).name}."
        ) from exc
    except httpx.RequestError as exc:
        raise DownloadError(
            f"Failed to download '{name}' from {url}\n"
            f"  Network error: {exc}\n"
            f"  Check your connection and that the URL is correct, then update "
            f"the SOURCES dict in {Path(__file__).name} if it has changed."
        ) from exc


def _github_commit_label(source: dict, client: httpx.Client) -> Optional[str]:
    """Return ``<short-sha> (<date>)`` for the file's latest commit, or None."""
    try:
        api = f"https://api.github.com/repos/{source['repo']}/commits"
        resp = client.get(
            api,
            params={
                "path": source["repo_path"],
                "sha": source["repo_ref"],
                "per_page": 1,
            },
            follow_redirects=True,
        )
        resp.raise_for_status()
        commits = resp.json()
        if not commits:
            return None
        sha = commits[0]["sha"][:10]
        date = commits[0]["commit"]["committer"]["date"]
        return f"{source['repo_ref']}@{sha} ({date})"
    except (httpx.HTTPError, KeyError, IndexError, ValueError):
        return None


def derive_version_label(
    source: dict, content: bytes, headers: httpx.Headers, client: httpx.Client
) -> Optional[str]:
    """Best-effort human-readable version for a source.

    Rules docs advertise their version as ``MAJOR.MINOR.PATCH`` in the text
    when present; card data is pinned to the latest commit. In all cases we
    fall back to the HTTP ``Last-Modified`` header.
    """
    kind = source["kind"]

    if kind == "rules_txt":
        text = content.decode("utf-8", errors="replace")
        m = re.search(r"version\s+(\d+\.\d+\.\d+)", text, flags=re.IGNORECASE)
        if m:
            return f"v{m.group(1)}"

    if kind == "card_json":
        label = _github_commit_label(source, client)
        if label:
            return label

    last_modified = headers.get("last-modified")
    if last_modified:
        return f"last-modified {last_modified}"
    return None


def load_manifest() -> Manifest:
    if MANIFEST_PATH.exists():
        try:
            return Manifest.model_validate_json(MANIFEST_PATH.read_text("utf-8"))
        except ValueError:
            print(
                f"! Existing manifest at {MANIFEST_PATH} is invalid; rebuilding.",
                file=sys.stderr,
            )
    return Manifest(generated_at=datetime.now(timezone.utc), files={})


def save_manifest(manifest: Manifest) -> None:
    manifest.generated_at = datetime.now(timezone.utc)
    MANIFEST_PATH.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download every source even if the hash already matches.",
    )
    args = parser.parse_args(argv)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    failures: list[str] = []

    with httpx.Client(timeout=HTTP_TIMEOUT, headers={"User-Agent": "fab_agent-corpus/0.1"}) as client:
        for name, source in SOURCES.items():
            filename = source["filename"]
            dest = RAW_DIR / filename
            existing = manifest.files.get(filename)

            # --- Idempotency: skip when on-disk hash matches the manifest. ---
            if not args.force and dest.exists() and existing is not None:
                if sha256_file(dest) == existing.sha256:
                    print(f"= {filename}: up to date (sha {existing.sha256[:12]}…), skipping.")
                    continue
                print(f"~ {filename}: on-disk hash differs from manifest, re-downloading.")

            print(f"↓ {filename}: downloading {source['label']} …")
            try:
                resp = download(name, source["url"], client)
            except DownloadError as exc:
                print(f"✗ {exc}", file=sys.stderr)
                failures.append(filename)
                continue

            content = resp.content
            digest = sha256_bytes(content)
            dest.write_bytes(content)
            version_label = derive_version_label(source, content, resp.headers, client)

            manifest.files[filename] = ManifestEntry(
                source_url=source["url"],
                download_date=datetime.now(timezone.utc),
                sha256=digest,
                version_label=version_label,
                bytes=len(content),
            )
            print(
                f"✓ {filename}: {len(content):,} bytes, sha {digest[:12]}…"
                + (f", version: {version_label}" if version_label else "")
            )

    save_manifest(manifest)
    print(f"\nManifest written to {MANIFEST_PATH.relative_to(ROOT)}")

    if failures:
        print(
            f"\n{len(failures)} source(s) failed: {', '.join(failures)}.\n"
            f"Fix the URL(s) in the SOURCES dict and re-run.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
