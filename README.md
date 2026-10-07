# newsrag

Daily news intelligence for US, Europe and India. Works with no API keys (rules mode) and uses an
LLM automatically when one is available, including local models via Ollama.

- What to build: [REQUIREMENTS.md](REQUIREMENTS.md)
- Why it is built this way: [ARCHITECTURE.md](ARCHITECTURE.md)
- Prompt to rebuild it with Claude Code: [PROMPT.md](PROMPT.md)

## Status
Stage 1 of 7: skeleton, models, config, settings, workspace, key redaction. Fetching, search,
digest, chat and UI arrive in later stages.

## Setup
```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

## Try it
```bash
.venv/bin/python -m newsrag --workspace ~/NewsRAG/personal workspace
.venv/bin/python -m newsrag config
```

## Checks
```bash
.venv/bin/pytest && .venv/bin/ruff check . && .venv/bin/mypy
```

## API keys
Optional. Set them as environment variables (for example `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`,
`GNEWS_API_KEY`) or type them in the UI. They are kept in memory only and never written to disk
or logs.
