"""Schema tests for the corpus manifest produced by scripts/download_corpus.py.

These tests never touch the network: they load the script module by path and
validate the ``Manifest`` / ``ManifestEntry`` pydantic models against fixtures.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

# Load scripts/download_corpus.py without requiring `scripts` to be a package.
# The module is registered in sys.modules so pydantic can resolve the deferred
# annotations (the script uses `from __future__ import annotations`).
_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "download_corpus.py"
_spec = importlib.util.spec_from_file_location("download_corpus", _MODULE_PATH)
download_corpus = importlib.util.module_from_spec(_spec)
sys.modules["download_corpus"] = download_corpus
_spec.loader.exec_module(download_corpus)

Manifest = download_corpus.Manifest
ManifestEntry = download_corpus.ManifestEntry


@pytest.fixture
def valid_manifest_dict() -> dict:
    """A well-formed manifest matching what the script writes."""
    return {
        "generated_at": "2026-07-11T12:00:00Z",
        "files": {
            "en-fab-cr.txt": {
                "source_url": "https://rules.fabtcg.com/txt/latest/en-fab-cr.txt",
                "download_date": "2026-07-11T12:00:00Z",
                "sha256": "a" * 64,
                "version_label": "v2.6.0",
                "bytes": 322134,
            },
            "card.json": {
                "source_url": (
                    "https://raw.githubusercontent.com/the-fab-cube/"
                    "flesh-and-blood-cards/develop/json/english/card.json"
                ),
                "download_date": "2026-07-11T12:00:00Z",
                "sha256": "0123456789abcdef" * 4,
                "version_label": None,  # version_label is optional
                "bytes": 22756270,
            },
        },
    }


def test_valid_manifest_parses(valid_manifest_dict):
    manifest = Manifest.model_validate(valid_manifest_dict)
    assert set(manifest.files) == {"en-fab-cr.txt", "card.json"}

    cr = manifest.files["en-fab-cr.txt"]
    assert isinstance(cr, ManifestEntry)
    assert cr.sha256 == "a" * 64
    assert cr.version_label == "v2.6.0"
    assert cr.bytes == 322134


def test_version_label_is_optional(valid_manifest_dict):
    entry = valid_manifest_dict["files"]["card.json"]
    del entry["version_label"]
    manifest = Manifest.model_validate(valid_manifest_dict)
    assert manifest.files["card.json"].version_label is None


def test_roundtrip_json_is_stable(valid_manifest_dict):
    manifest = Manifest.model_validate(valid_manifest_dict)
    dumped = manifest.model_dump_json()
    reparsed = Manifest.model_validate_json(dumped)
    assert reparsed == manifest


@pytest.mark.parametrize("bad_hash", ["", "xyz", "A" * 64, "a" * 63, "a" * 65])
def test_rejects_malformed_sha256(valid_manifest_dict, bad_hash):
    valid_manifest_dict["files"]["en-fab-cr.txt"]["sha256"] = bad_hash
    with pytest.raises(ValidationError):
        Manifest.model_validate(valid_manifest_dict)


def test_rejects_missing_required_field(valid_manifest_dict):
    del valid_manifest_dict["files"]["en-fab-cr.txt"]["source_url"]
    with pytest.raises(ValidationError):
        Manifest.model_validate(valid_manifest_dict)


def test_rejects_negative_bytes(valid_manifest_dict):
    valid_manifest_dict["files"]["en-fab-cr.txt"]["bytes"] = -1
    with pytest.raises(ValidationError):
        Manifest.model_validate(valid_manifest_dict)


def test_sources_are_well_formed():
    """Every SOURCES entry has the fields the downloader relies on."""
    for name, source in download_corpus.SOURCES.items():
        assert source["url"].startswith("https://"), name
        assert source["filename"], name
        assert source["kind"] in {"rules_txt", "card_json"}, name
