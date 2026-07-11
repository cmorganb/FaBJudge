# fab_agent — Project & Architecture

An LLM-based **rules-adjudication agent** for the trading card game
*Flesh and Blood* (FAB). Given a natural-language rules question or a described
board state, the agent produces a structured, **citation-bearing verdict**
grounded in the official rules corpus.

## Guiding research question

> **Given identical grounding architecture, how do modest open-weights models
> compare to frontier closed models — does grounding matter more than model
> scale for accurate, faithfully cited rulings?**

Everything below is designed to isolate that variable: the retrieval and
reasoning pipeline is held constant, and only the underlying model is swapped
(via configuration) between frontier closed models and modest open-weights
models. Differences in accuracy and citation faithfulness are then attributable
to model scale rather than to architectural differences.

## Design principle: model-agnostic by construction

The system **must run unchanged** on:

- **Closed / frontier models** — Gemini, GPT, Claude (hosted APIs), and
- **Open-weights models** — served locally via **Ollama** through its
  OpenAI-compatible endpoint.

The backend is selected **purely through configuration** (`.env` →
`fab_agent.config.Settings`). No pipeline, prompt, tool, or retrieval component
branches on the model vendor. All providers are reached through a single
OpenAI-compatible client abstraction, so the only thing that changes between an
experiment on Gemini and one on a local Llama model is a handful of environment
variables. See `.env.example` and `fab_agent/config.py`.

## End-to-end pipeline

```
                          ┌─────────────────────────────────────────┐
   user query ──────────► │ ROUTER                                  │
                          │ classify intent:                        │
                          │   rules | infraction | card | clarify   │
                          └───────────────┬─────────────────────────┘
                                          │ (route + normalized query)
                                          ▼
                          ┌─────────────────────────────────────────┐
                          │ HYBRID RETRIEVAL                        │
                          │   BM25 (rank-bm25, lexical)             │
                          │   +  dense embeddings (bge-small,       │
                          │      Chroma vector store)               │
                          │   → fused, de-duplicated passages       │
                          └───────────────┬─────────────────────────┘
                                          │ (grounding context + source ids)
                                          ▼
                          ┌─────────────────────────────────────────┐
                          │ ReAct AGENT LOOP                        │
                          │   Thought → Action → Observation …      │
                          │   tools: search_rules, lookup_card,     │
                          │          lookup_infraction, ...         │
                          └───────────────┬─────────────────────────┘
                                          │
                                          ▼
                          ┌─────────────────────────────────────────┐
                          │ VERDICT (JSON, IRAC-structured)         │
                          │   Issue / Rule / Application / Conclusion│
                          │   + citations to corpus sources          │
                          └─────────────────────────────────────────┘
```

### 1. Router (`fab_agent/router/`)
Classifies each query into one of four intents and normalizes it for
retrieval:

- **rules** — general game-rules questions (Comprehensive Rules).
- **infraction** — tournament policy / penalties (Infraction Procedure Guide).
- **card** — questions about a specific card's text, keywords, or errata.
- **clarify** — the query is under-specified; the agent should ask a
  follow-up rather than guess.

### 2. Hybrid retrieval (`fab_agent/retrieval/`)
Combines **lexical** recall (BM25 over the tokenized corpus, `rank-bm25`) with
**dense semantic** recall (sentence-transformers `BAAI/bge-small-en-v1.5`
embeddings stored in **Chroma**). Results are fused and de-duplicated so both
exact-term matches (card names, keyword abilities) and paraphrased concepts are
surfaced. Every returned passage carries a stable source id used later for
citation.

### 3. ReAct agent loop (`fab_agent/agent/`, `fab_agent/tools/`)
A tool-using **ReAct** loop (Thought → Action → Observation) lets the model
iteratively gather evidence before committing to a ruling. Tools wrap the
retrieval layer and corpus lookups (e.g. `search_rules`, `lookup_card`,
`lookup_infraction`). The loop is provider-neutral and driven through the
OpenAI-compatible client.

### 4. IRAC verdict (structured output)
The final answer is emitted as **JSON** following the legal **IRAC** structure:

- **Issue** — the precise question being adjudicated.
- **Rule** — the governing rule(s), quoted with citations.
- **Application** — how the rule applies to the specific situation.
- **Conclusion** — the ruling.

Each verdict is **citation-bearing**: every rule invoked references a specific
corpus source, enabling faithfulness evaluation.

## Corpus (`fab_agent/corpus/`)
Ingests and normalizes official sources into the retrieval index:
Comprehensive Rules, the Infraction Procedure Guide, and card data. PDFs are
parsed with `pypdf`; normalized chunks land in `data/processed/`, and the
vector index in `data/index/`.

## Evaluation (`fab_agent/eval/`, `scenarios/`)
`scenarios/` holds curated adjudication scenarios (question → expected ruling +
expected citations). The `eval` package scores model runs on **accuracy** and
**citation faithfulness**, so the same scenario set can be replayed across
providers to answer the research question. `pandas` + `matplotlib` summarize
results.

## Serving (`fab_agent/ui/`)
A **FastAPI** service exposes the adjudication endpoint; a **Streamlit** app
provides an interactive front-end. Both call the same provider-agnostic core.

## Repository layout

```
fab_agent/
  config.py        # pydantic-settings; single source of backend selection
  corpus/          # ingestion & normalization of official sources
  retrieval/       # BM25 + embedding hybrid retrieval
  tools/           # ReAct tool implementations
  agent/           # ReAct adjudication loop + IRAC verdict assembly
  router/          # query intent classification
  eval/            # accuracy & citation-faithfulness evaluation
  ui/              # FastAPI service + Streamlit app
data/
  raw/             # original source documents (git-ignored)
  processed/       # normalized chunks (tracked structure)
  index/           # built vector index (git-ignored)
scenarios/         # curated adjudication test scenarios
scripts/           # ingestion / indexing / evaluation entry points
tests/             # unit & smoke tests
```
