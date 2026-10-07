# newsrag

Daily news intelligence for US, Europe and India. Works with no API keys (rules mode) and uses an
LLM automatically when one is available, including local models via Ollama.

- What to build: [REQUIREMENTS.md](REQUIREMENTS.md)
- Why it is built this way: [ARCHITECTURE.md](ARCHITECTURE.md)
- Prompt to rebuild it with Claude Code: [PROMPT.md](PROMPT.md)

## Status
All 6 build stages done (stage 7 is docs and scheduling). Fetch, process, store, search, digest,
grounded chat, agent-ready tools, and a web UI.

## Web UI
```bash
.venv/bin/python -m newsrag ui                         # http://localhost:8501
.venv/bin/python -m newsrag --workspace ~/NewsRAG/personal ui
```
- Start screen: open a recent workspace or create one in any folder.
- Top right: text size **A / A+ / A++** and theme **Light / Dark / System** (saved per workspace).
- Pages: **Today** (briefing, region filter), **Ask** (grounded chat with sources),
  **Browse** (search or latest), **Fetch** (fetch now, run history), **Sources** (on/off,
  add and check a feed, OPML import/export), **Data** (stats, cleanup with backup, delete by
  source/region/date, verify, rebuild index, backup/restore), **Settings** (LLM, keys in
  memory only, retrieval, sources, briefing).

The first `run` or `search` downloads two small local models (about 90 MB each) from Hugging Face.

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
.venv/bin/python -m newsrag run                # fetch, process and store new articles
.venv/bin/python -m newsrag search "RBI rate hike" --region IN --days 7
.venv/bin/python -m newsrag digest --print     # write (and show) today's briefing
.venv/bin/python -m newsrag chat               # interactive; /region IN, /days 3, /quit
.venv/bin/python -m newsrag chat "What did the RBI decide?" --region IN --days 3
.venv/bin/python -m newsrag tools              # agent-ready tools (--json for schemas)
.venv/bin/python -m newsrag data stats         # also: cleanup [--apply], verify [--repair],
                                               #       reindex, compact, backup, restore PATH
```

## Email (optional)
Set `email_enabled` in Settings and these environment variables (the password stays in memory):
`NEWSRAG_SMTP_HOST`, `NEWSRAG_SMTP_PORT` (default 587), `NEWSRAG_SMTP_USER`,
`NEWSRAG_SMTP_PASSWORD`, `NEWSRAG_SMTP_TO`, optional `NEWSRAG_SMTP_FROM`. For Gmail use an
app password.

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
