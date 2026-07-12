"""Persistent lexical (BM25) and semantic (Chroma) indexes over the chunks.

Two complementary indexes are built over ``data/processed/chunks.jsonl``:

* :class:`BM25Index` — lexical retrieval via ``rank_bm25``. The tokenizer keeps
  hierarchical rule numbers such as ``1.2.3a`` as single tokens so a query can
  match a citation anchor exactly. Persisted with :mod:`pickle`.
* :class:`ChromaIndex` — dense retrieval via a persistent ChromaDB collection,
  embedded with the sentence-transformers model named in
  :class:`fab_agent.config.Settings` (default ``BAAI/bge-small-en-v1.5``).

Heavy dependencies (torch / sentence-transformers / chromadb) are imported
lazily so the tokenizer and BM25 path stay cheap to import and test.
"""

from __future__ import annotations

import json
import pickle
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = ROOT / "data" / "index"
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
BM25_PATH = INDEX_DIR / "bm25.pkl"
CHROMA_PATH = INDEX_DIR / "chroma"

COLLECTION_NAME = "fab_corpus"
#: BAAI/bge models expect this instruction prefixed to *queries* (not passages).
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

#: A rule number (``1.2.3`` / ``1.2.3a``) OR an alphanumeric word (optionally
#: with an internal apostrophe). Rule numbers are matched first so they survive
#: as single tokens instead of being shattered on their dots.
_TOKEN_RE = re.compile(r"\d+(?:\.\d+)+[a-z]?|[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    """Lowercase, punctuation-stripping tokenizer that keeps rule numbers whole."""
    return _TOKEN_RE.findall(text.lower())


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    """Load chunk records from a JSONL file."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


#: How many times the rule id is repeated in the lexical document. Boosting the
#: id's term frequency makes a chunk rank first for *its own* number, above the
#: many chunks that merely reference it (e.g. "[7.5.5]").
RULE_ID_BOOST = 4


def _index_text(chunk: dict) -> str:
    """Text fed to the lexical index: (boosted) rule id + title + body."""
    rule_id = chunk.get("rule_id", "")
    boosted_id = " ".join([rule_id] * RULE_ID_BOOST) if rule_id else ""
    parts = [boosted_id, chunk.get("title", ""), chunk.get("text", "")]
    return " ".join(p for p in parts if p)


@dataclass
class Hit:
    chunk_id: str
    score: float
    metadata: dict


# --------------------------------------------------------------------------- #
# Lexical index
# --------------------------------------------------------------------------- #
class BM25Index:
    """A BM25 index that persists its tokenized corpus and chunk ids."""

    def __init__(self, chunk_ids: list[str], tokenized_corpus: list[list[str]],
                 metadatas: list[dict]):
        from rank_bm25 import BM25Okapi

        self.chunk_ids = chunk_ids
        self.tokenized_corpus = tokenized_corpus
        self.metadatas = metadatas
        self.bm25 = BM25Okapi(tokenized_corpus)

    @classmethod
    def build(cls, chunks: list[dict]) -> "BM25Index":
        chunk_ids = [c["chunk_id"] for c in chunks]
        tokenized = [tokenize(_index_text(c)) for c in chunks]
        metadatas = [{"doc": c.get("doc"), "rule_id": c.get("rule_id")} for c in chunks]
        return cls(chunk_ids, tokenized, metadatas)

    def save(self, path: Path = BM25_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "chunk_ids": self.chunk_ids,
                    "tokenized_corpus": self.tokenized_corpus,
                    "metadatas": self.metadatas,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: Path = BM25_PATH) -> "BM25Index":
        with Path(path).open("rb") as fh:
            data = pickle.load(fh)
        return cls(data["chunk_ids"], data["tokenized_corpus"], data["metadatas"])

    def query(self, text: str, k: int = 5) -> list[Hit]:
        scores = self.bm25.get_scores(tokenize(text))
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        return [Hit(self.chunk_ids[i], float(scores[i]), self.metadatas[i]) for i in order]


# --------------------------------------------------------------------------- #
# Semantic index
# --------------------------------------------------------------------------- #
def _embed(model, texts: list[str], *, batch_size: int, show_progress: bool) -> list[list[float]]:
    """Encode ``texts`` in batches (normalized) with a tqdm progress bar."""
    from tqdm import tqdm

    vectors: list[list[float]] = []
    for start in tqdm(range(0, len(texts), batch_size),
                      disable=not show_progress, desc="embedding", unit="batch"):
        batch = texts[start:start + batch_size]
        embeddings = model.encode(batch, normalize_embeddings=True,
                                  show_progress_bar=False)
        vectors.extend(v.tolist() for v in embeddings)
    return vectors


class ChromaIndex:
    """A persistent Chroma collection of chunk embeddings."""

    def __init__(self, collection, model):
        self.collection = collection
        self.model = model

    @property
    def dimension(self) -> int:
        # Method was renamed across sentence-transformers versions.
        getter = getattr(self.model, "get_embedding_dimension", None) or \
            self.model.get_sentence_embedding_dimension
        return getter()

    @classmethod
    def _load_model(cls, model_name: str):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)

    @classmethod
    def build(cls, chunks: list[dict], model_name: str, *, path: Path = CHROMA_PATH,
              batch_size: int = 64, show_progress: bool = True) -> "ChromaIndex":
        import chromadb

        path = Path(path)
        if path.exists():  # idempotent: delete-and-rebuild for a clean slate
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

        model = cls._load_model(model_name)
        ids = [c["chunk_id"] for c in chunks]
        documents = [c.get("text", "") for c in chunks]
        metadatas = [{"doc": c.get("doc"), "rule_id": c.get("rule_id")} for c in chunks]
        embeddings = _embed(model, documents, batch_size=batch_size, show_progress=show_progress)

        client = chromadb.PersistentClient(path=str(path))
        collection = client.create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        # Chroma caps the number of records per add() call; batch to stay safe.
        add_batch = 1000
        for start in range(0, len(ids), add_batch):
            sl = slice(start, start + add_batch)
            collection.add(ids=ids[sl], embeddings=embeddings[sl],
                           documents=documents[sl], metadatas=metadatas[sl])

        return cls(collection, model)

    @classmethod
    def load(cls, model_name: str, *, path: Path = CHROMA_PATH) -> "ChromaIndex":
        import chromadb

        client = chromadb.PersistentClient(path=str(path))
        collection = client.get_collection(name=COLLECTION_NAME)
        return cls(collection, cls._load_model(model_name))

    def count(self) -> int:
        return self.collection.count()

    def query(self, text: str, k: int = 5) -> list[Hit]:
        vector = self.model.encode(
            [QUERY_INSTRUCTION + text], normalize_embeddings=True, show_progress_bar=False
        )[0].tolist()
        result = self.collection.query(query_embeddings=[vector], n_results=k)
        hits: list[Hit] = []
        for cid, dist, meta in zip(
            result["ids"][0], result["distances"][0], result["metadatas"][0]
        ):
            hits.append(Hit(cid, float(dist), meta or {}))
        return hits
