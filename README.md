# newsrag

Daily news intelligence for US, Europe and India. Works with no API keys (rules mode) and uses an
LLM automatically when one is available, including local models via Ollama.

- What to build: [REQUIREMENTS.md](REQUIREMENTS.md)
- Why it is built this way: [ARCHITECTURE.md](ARCHITECTURE.md)
- Prompt to rebuild it with Claude Code: [PROMPT.md](PROMPT.md)

## Status
Stage 3 of 7: skeleton, sources and pipeline (stages 1-2), plus the two engines. `RuleEngine`
works with no key; `LLMEngine` uses Ollama, a local OpenAI-compatible server, Claude, Gemini or
an OpenAI-compatible API, validates every reply, and falls back to rules per item. Storage,
search, digest, chat and UI arrive in later stages.

## LLM options (all optional)
- **Local, no key:** install [Ollama](https://ollama.com), pull a model, and leave the provider
  set to `ollama` (the default). With no model set, the first installed model is used.
- **Claude:** `uv pip install -e ".[anthropic]"` and set `ANTHROPIC_API_KEY`. Default model is
  `claude-opus-5-5`; choose another per task in Settings.
- **Gemini / OpenAI-compatible:** set the key and choose a model in Settings.

`auto` mode (default) tries the configured provider, then Claude if a key is present, then
falls back to rules and says why.

## Setup
```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

## Try it
```bash
.venv/bin/python -m newsrag --workspace ~/NewsRAG/personal workspace
.venv/bin/python -m newsrag config
.venv/bin/python -m newsrag sources            # list configured sources
.venv/bin/python -m newsrag sources --check    # fetch each one live and report status
.venv/bin/python -m newsrag fetch --show 20    # dry run: fetch, filter, dedupe, rank
.venv/bin/python -m newsrag process --limit 5  # dry run: fetch, then summarise and tag
```

## Sources
Feeds live in [newsrag/defaults/config.yaml](newsrag/defaults/config.yaml). Each was fetched,
parsed and checked against the site's robots.txt before being added. Google News RSS is not used
because its robots.txt disallows `/rss`.

## Checks
```bash
.venv/bin/pytest && .venv/bin/ruff check . && .venv/bin/mypy
```

## API keys
Optional. Set them as environment variables (for example `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`,
`GNEWS_API_KEY`) or type them in the UI. They are kept in memory only and never written to disk
or logs.
