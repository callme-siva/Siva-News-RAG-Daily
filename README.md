# newsrag

Daily news intelligence for US, Europe and India. Works with no API keys (rules mode) and uses an
LLM automatically when one is available, including local models via Ollama.

- What to build: [REQUIREMENTS.md](REQUIREMENTS.md)
- Why it is built this way: [ARCHITECTURE.md](ARCHITECTURE.md)
- Prompt to rebuild it with Claude Code: [PROMPT.md](PROMPT.md)

## Status
Stage 2 of 7: skeleton (stage 1) plus sources and the deterministic pipeline: 35 verified RSS
feeds across US, Europe and India, GDELT (off by default), optional GNews and NewsData adapters,
then filter, dedupe and rank. Storage, search, digest, chat and UI arrive in later stages.

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
