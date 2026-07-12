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

`HybridRetriever.retrieve()` (`fab_agent/retrieval/hybrid.py`) is the single
retrieval entry point shared by the agent and the evaluation harness. In the
default **hybrid** mode it pulls the top 20 from each retriever and merges them
with **Reciprocal Rank Fusion** (RRF, k=60), de-duplicating by `chunk_id`. A
`doc_filter` argument scopes results to specific documents (e.g. `["PPG"]`),
which the router uses to route infraction questions to the penalty guide.

**Ablation modes.** `retrieve(..., mode=...)` also accepts `"bm25"`
(lexical-only) and `"dense"` (dense-only). These exist specifically for the
Stage 5 ablation — *hybrid vs lexical-only vs dense-only* — so the harness can
replay identical queries against each configuration through one code path and
attribute quality differences to the retrieval strategy, keeping the grounding
architecture constant per the research question.

### 3. ReAct agent loop (`fab_agent/agent/`, `fab_agent/tools/`)
A tool-using **ReAct** loop (Thought → Action → Observation) lets the model
iteratively gather evidence before committing to a ruling. Tools wrap the
retrieval layer and corpus lookups (e.g. `search_rules`, `lookup_card`,
`lookup_infraction`). The loop is provider-neutral and driven through the
OpenAI-compatible client.

#### Tools (`fab_agent/tools/`)
The agent reasons over four tools, each a plain function with pydantic
input/output models, exported as OpenAI-compatible schemas by
`fab_agent/tools/registry.py`:

- **`search_rules`** — hybrid retrieval over the corpus (thin wrapper over
  `HybridRetriever`), returning citation-anchored passages.
- **`get_card`** — fuzzy card lookup over `cards.jsonl`; returns close
  suggestions instead of guessing when no confident match exists.
- **`get_precedence_context`** — the **defeasible-reasoning** framing (below).
- **`ask_clarification`** — a sentinel the agent emits to stop and ask the user;
  reserved for cases where the ruling genuinely cannot be determined without a
  missing fact (the evaluation penalizes unnecessary clarifications).

**Defeasible reasoning & precedence.** FAB rulings are *defeasible*: a general
rule holds unless a more specific authority overrides it. The governing order is
CR by default → card text overrides general CR rules where they conflict → in
sanctioned tournaments the TRP and PPG prevail within their scope, at the active
Rules Enforcement Level. This framing is stored as an editable template
(`fab_agent/tools/precedence.md`) and surfaced through `get_precedence_context`
so the agent weighs conflicting authorities correctly before concluding.

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
