# fab_agent

An LLM-based **rules-adjudication agent** for the trading card game
*Flesh and Blood*. It answers rules questions with structured, citation-bearing
verdicts grounded in the official rules corpus, using hybrid retrieval and a
ReAct tool-using agent loop.

The system is **model-agnostic**: the same architecture runs unchanged on
closed models (Gemini, GPT, Claude) and open-weights models (via Ollama),
selected purely through configuration. See [PROJECT.md](PROJECT.md) for the
full architecture and the guiding research question.

## Requirements

- Python **3.11+**
- [uv](https://docs.astral.sh/uv/) for dependency management

## Setup

```bash
# 1. Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies into a managed virtual environment
uv sync

# 3. Configure your model backend
cp .env.example .env
#    then edit .env and set LLM_API_KEY (and LLM_PROVIDER / LLM_MODEL if desired)
```

### Getting a Gemini API key (default, free tier)

The default backend is **Gemini 3.5 Flash**, the current free-tier model.
Create an API key at **Google AI Studio**:
<https://aistudio.google.com/app/apikey> — no billing setup is required for the
free tier. Paste the key into `.env` as `LLM_API_KEY`.

### Using a local open-weights model (Ollama)

```bash
# Install Ollama (https://ollama.com) and pull a model, e.g.:
ollama pull llama3.1:8b
```

Then in `.env`:

```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
# LLM_BASE_URL defaults to http://localhost:11434/v1 automatically
# LLM_API_KEY can be left blank
```

No other change is needed — the pipeline runs identically on the local model.

## Running the tests

```bash
uv run pytest
```

## Configuration reference

All configuration lives in `.env` and is loaded by
`fab_agent.config.Settings` (pydantic-settings). See
[`.env.example`](.env.example) for every variable and its meaning:

| Variable          | Default                   | Purpose                                            |
| ----------------- | ------------------------- | -------------------------------------------------- |
| `LLM_PROVIDER`    | `gemini`                  | Backend: `gemini` \| `openai` \| `anthropic` \| `ollama` |
| `LLM_MODEL`       | `gemini-3.5-flash`        | Model identifier for the provider                  |
| `LLM_API_KEY`     | *(none)*                  | Provider API key (unused for Ollama)               |
| `LLM_BASE_URL`    | *(provider default)*      | OpenAI-compatible base-URL override                |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5`  | Dense-retrieval embedding model                    |
| `REL_LEVEL`       | `competitive`             | Rules Enforcement Level context for rulings        |

## Project layout

```
fab_agent/     corpus/ retrieval/ tools/ agent/ router/ eval/ ui/
data/          raw/ processed/ index/
scenarios/     curated adjudication scenarios
scripts/       ingestion / indexing / evaluation entry points
tests/         unit & smoke tests
```

## License

For academic / research use.
