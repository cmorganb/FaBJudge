"""Tests for the lexical (BM25) and semantic (Chroma) indexes.

A 10-chunk fixture is indexed both ways. BM25 must satisfy exact keyword /
rule-number queries; the vector index must satisfy paraphrase queries that
share little surface vocabulary with the target chunk.

The semantic test loads the real sentence-transformers model, so it is a little
slow the first time (model download) but runs on only 10 short chunks.
"""

import pytest

from fab_agent.config import get_settings
from fab_agent.retrieval.index import BM25Index, ChromaIndex, tokenize

FIXTURE = [
    {"chunk_id": "CR-1.2.3a", "doc": "CR", "rule_id": "1.2.3a", "title": "Intangible",
     "text": "An intangible object cannot be targeted by attacks or effects that "
             "specify a legal target."},
    {"chunk_id": "CR-2.1", "doc": "CR", "rule_id": "2.1", "title": "Life",
     "text": "A hero begins the game with life points equal to the value printed in "
             "the top right of the hero card, usually twenty."},
    {"chunk_id": "CR-3.1", "doc": "CR", "rule_id": "3.1", "title": "Arena",
     "text": "The arena is the shared space where attacks and defenses are resolved "
             "between the two heroes."},
    {"chunk_id": "CR-4.2", "doc": "CR", "rule_id": "4.2", "title": "Pitch",
     "text": "Pitching a card sends it to the pitch zone to generate resource points "
             "used to pay costs."},
    {"chunk_id": "CR-5.3", "doc": "CR", "rule_id": "5.3", "title": "Arsenal",
     "text": "A card placed in the arsenal is set aside face down and may be played "
             "on a later turn."},
    {"chunk_id": "CR-6.4", "doc": "CR", "rule_id": "6.4", "title": "Go Again",
     "text": "An attack with go again grants the player an additional action point "
             "after it resolves."},
    {"chunk_id": "CR-7.5", "doc": "CR", "rule_id": "7.5", "title": "Blocking",
     "text": "During the defend step a hero may block by committing cards from hand "
             "to reduce incoming damage."},
    {"chunk_id": "PPG-2.1", "doc": "PPG", "rule_id": "2.1", "title": "Slow Play",
     "text": "A player who takes too long to make their decisions may receive a "
             "penalty for slow play."},
    {"chunk_id": "TRP-1.2", "doc": "TRP", "rule_id": "1.2", "title": "Rules Enforcement Levels",
     "text": "The three rules enforcement levels are casual, competitive, and "
             "professional, each with different expectations."},
    {"chunk_id": "CARD-command-and-conquer", "doc": "CARD", "rule_id": "Command and Conquer",
     "title": "Command and Conquer", "text": "Command and Conquer is a warrior attack "
             "action with dominate and go again."},
]


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
def test_tokenizer_keeps_rule_numbers_whole():
    toks = tokenize("See rule 1.2.3a and also 10.0.1 for details.")
    assert "1.2.3a" in toks
    assert "10.0.1" in toks
    # A rule number is one token, not shattered on its dots.
    assert "1" not in toks and "2" not in toks


def test_tokenizer_lowercases_and_strips_punctuation():
    assert tokenize("Go Again! (Dominate)") == ["go", "again", "dominate"]


# --------------------------------------------------------------------------- #
# Shared indexes (built once)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def bm25(tmp_path_factory):
    path = tmp_path_factory.mktemp("idx") / "bm25.pkl"
    BM25Index.build(FIXTURE).save(path)
    return BM25Index.load(path)  # exercise the persistence round-trip


@pytest.fixture(scope="module")
def chroma(tmp_path_factory):
    path = tmp_path_factory.mktemp("chroma") / "chroma"
    model = get_settings().embedding_model
    ChromaIndex.build(FIXTURE, model, path=path, batch_size=8, show_progress=False)
    return ChromaIndex.load(model, path=path)  # exercise the persistence round-trip


# --------------------------------------------------------------------------- #
# Lexical (BM25) — exact keyword / rule-number queries
# --------------------------------------------------------------------------- #
def test_bm25_rule_number_query(bm25):
    hits = bm25.query("1.2.3a", k=3)
    assert hits[0].chunk_id == "CR-1.2.3a"


def test_bm25_keyword_query(bm25):
    hits = bm25.query("card face down in arsenal", k=3)
    assert hits[0].chunk_id == "CR-5.3"

    hits = bm25.query("penalty for taking too long", k=3)
    assert hits[0].chunk_id == "PPG-2.1"


# --------------------------------------------------------------------------- #
# Semantic (Chroma) — paraphrase queries with little lexical overlap
# --------------------------------------------------------------------------- #
def test_chroma_paraphrase_starting_health(chroma):
    # "health"/"start of a match" do not appear in the target chunk ("life
    # points", "begins the game"); only a semantic match can find it.
    hits = chroma.query("how much health does a hero have at the start of a match", k=3)
    assert hits[0].chunk_id == "CR-2.1"


def test_chroma_paraphrase_stalling(chroma):
    hits = chroma.query("what is the ruling when someone stalls the match", k=3)
    assert hits[0].chunk_id == "PPG-2.1"


def test_chroma_metadata_and_dim(chroma):
    assert chroma.dimension == 384  # BAAI/bge-small-en-v1.5
    hits = chroma.query("shared space where combat happens", k=1)
    assert hits[0].metadata["doc"] == "CR"
    assert hits[0].metadata["rule_id"] == "3.1"
