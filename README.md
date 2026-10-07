# newsrag: Daily News Intelligence

Collects US, Europe and India news every day, stores it in a searchable local memory, writes a
morning briefing, and answers questions with numbered sources. It works with **no API keys**
(pure Python rules) and uses an **LLM automatically** when one is available: a local model via
Ollama, or Claude, Gemini or an OpenAI-compatible API.

- What it must do: [REQUIREMENTS.md](REQUIREMENTS.md)
- Why it is built this way: [ARCHITECTURE.md](ARCHITECTURE.md)
- Acceptance results with raw output: [ACCEPTANCE.md](ACCEPTANCE.md)
- Prompt to rebuild it with Claude Code: [PROMPT.md](PROMPT.md)

## Contents
1. [Setup](#setup)
2. [First run](#first-run)
3. [Web UI](#web-ui)
4. [Command line](#command-line)
5. [Using an LLM (optional)](#using-an-llm-optional)
6. [Running it every day](#running-it-every-day)
7. [Sources: adding and checking feeds](#sources-adding-and-checking-feeds)
8. [Email (optional)](#email-optional)
9. [Data, cleanup and backups](#data-cleanup-and-backups)
10. [Troubleshooting](#troubleshooting)
11. [Project layout and development](#project-layout-and-development)

## Setup
Needs Python 3.11 or newer (3.12 recommended) and [uv](https://docs.astral.sh/uv/) or pip.
```bash
cd Siva-News-RAG-Daily
uv venv --python 3.12
uv pip install -e ".[dev]"            # add ",anthropic" inside the brackets for Claude
```
The first `run`, `search` or chat downloads two small local models (about 90 MB each) into the
Hugging Face cache (`~/.cache/huggingface`), shared by all workspaces.

## First run
A **workspace** is one folder holding everything for one profile: the database (articles,
keyword index, vectors), digests, logs and settings. API keys are never stored in it.
```bash
.venv/bin/python -m newsrag --workspace ~/NewsRAG/personal workspace   # create it
.venv/bin/python -m newsrag run                                        # fetch, store, write the briefing
.venv/bin/python -m newsrag ui                                         # open the web UI
```
After the first time, commands use the most recently opened workspace.

## Web UI
```bash
.venv/bin/python -m newsrag ui                    # http://localhost:8501
.venv/bin/python -m newsrag --workspace PATH ui   # open a specific workspace
```
- Start screen: open a recent workspace or create one in any folder.
- Top right: text size **A / A+ / A++** and theme **Light / Dark / System**, saved per workspace.
- **Today**: the briefing with a region filter. **Ask**: chat answered only from stored articles,
  with sources. **Browse**: search or latest articles, with scores. **Fetch**: fetch now and run
  history. **Sources**: turn feeds on or off, add and check a feed, OPML import/export.
  **Data**: stats, cleanup, delete by filter, verify, rebuild index, backup and restore.
  **Settings**: LLM, keys (memory only), retrieval, sources, briefing.
- Every summary, answer and briefing shows whether it was written by **rules** or an **LLM**.

## Command line
| Command | What it does |
|---------|--------------|
| `workspace` / `recent` | Create or open a workspace / list recent ones |
| `run [--limit N] [--no-digest]` | Fetch, process and store new articles, then write the briefing |
| `digest [--print] [--mode rules\|llm]` | Write today's briefing from stored articles |
| `chat ["question"] [--region IN] [--category Finance] [--days 3]` | Ask; without a question it is interactive (`/region`, `/days`, `/clear`, `/quit`) |
| `search "query" [--region] [--category] [--days] [--mode] [--no-rerank]` | Hybrid search with scores |
| `sources [--check] [--all]` | List sources, or fetch each one and report its status |
| `fetch` / `process` | Dry runs: show what would be fetched / how items would be summarised |
| `data stats\|cleanup [--days N] [--apply]\|verify [--repair]\|reindex\|compact\|backup\|restore PATH` | Maintenance |
| `tools [--json]` | The typed tool functions a future agent can call |
| `config` | Regions, categories, sources and which keys were found |
| `ui [--port 8501]` | Start the web UI |

All commands accept `--workspace PATH` before the command name.

## Using an LLM (optional)
`auto` mode (default) tries the configured provider, then Claude if a key is set, then falls back
to rules and says why. If an LLM call fails for one article, that article uses rules.

**Local, private, no key: Ollama**
```bash
ollama pull llama3.2      # or any model; choose it in Settings, or the first installed is used
```
Provider `ollama` is the default. For other local servers (LM Studio, llama.cpp, vLLM) choose
"Local: OpenAI-compatible server" and set the base URL, for example `http://localhost:1234/v1`.
Embeddings can also come from Ollama: set the embedding model to `ollama:nomic-embed-text`,
then Data → Rebuild index.

**Hosted, with a key**
| Provider | Key (environment variable or Settings → API keys) | Model |
|----------|---------------------------------------------------|-------|
| Claude | `ANTHROPIC_API_KEY` (needs `uv pip install -e ".[anthropic]"`) | default `claude-opus-5-5`; set per task in Settings (for example `claude-haiku-4-5` for tagging) |
| Gemini | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | choose in Settings |
| OpenAI-compatible | `OPENAI_API_KEY` + base URL | choose in Settings |

Keys typed in the UI live in memory for that session only. They are never written to disk,
logs, settings or URLs (checked in [ACCEPTANCE.md](ACCEPTANCE.md), criterion 7).

## Running it every day
The app does not need to run all the time. Any scheduler that runs `newsrag run` works; each run
looks back to the previous run (up to `max_catchup_days`, default 7), so missed days catch up.

| Where | File | Notes |
|-------|------|-------|
| macOS (recommended on a Mac) | [deploy/com.newsrag.daily.plist](deploy/com.newsrag.daily.plist) | launchd; runs on wake if the Mac slept through 06:30 |
| Linux or macOS cron | [deploy/cron.txt](deploy/cron.txt) | one line in `crontab -e` |
| GitHub Actions (cloud) | [deploy/github-actions-daily.yml](deploy/github-actions-daily.yml) | emails the briefing; the cloud workspace is a best-effort cache, read the notes in the file |

Keys for scheduled runs go in `~/.newsrag.env` (`chmod 600`) as `export NAME=value` lines; the
cron and launchd examples load it.

## Sources: adding and checking feeds
35 RSS feeds ship in [newsrag/defaults/config.yaml](newsrag/defaults/config.yaml), each fetched,
parsed and checked against the site's `robots.txt` before it was added. Google News RSS is not
used because its `robots.txt` disallows `/rss`.

- **In the UI (recommended):** Sources → Add an RSS feed → Check feed. It checks `robots.txt`,
  fetches and parses the feed, and shows its title and three latest headlines before you save.
  Your additions and on/off choices are stored in the workspace (`sources.json`); the shipped
  file is never edited, and "Restore defaults" removes your changes.
- **For everyone using this copy:** add a line to `config.yaml`, then run
  `newsrag sources --check`:
  ```yaml
  - {name: Example Markets, type: rss, region: IN, category: Finance, url: "https://example.com/feed.xml"}
  ```
  Add `timezone: Asia/Kolkata` if the feed's dates have no timezone.
- **New category:** add it under `categories:` with `include` and `exclude` patterns. Use word
  boundaries (`\bterm\b`) so short words do not match inside longer ones.
- **GDELT** topic searches are included but off; GDELT allows one request per 5 seconds.
- **GNews / NewsData** sources run only when `GNEWS_API_KEY` / `NEWSDATA_API_KEY` is set.

## Email (optional)
Turn on "Email the briefing" in Settings and set (password in memory only):
`NEWSRAG_SMTP_HOST`, `NEWSRAG_SMTP_PORT` (default 587), `NEWSRAG_SMTP_USER`,
`NEWSRAG_SMTP_PASSWORD`, `NEWSRAG_SMTP_TO`, optional `NEWSRAG_SMTP_FROM`.
For Gmail, create an app password and use `smtp.gmail.com`.

## Data, cleanup and backups
- **Duplicates** are blocked four ways: URL identity (tracking parameters stripped), seen URLs
  (skipped before any LLM call), cross-day near-duplicate merge (other outlets are listed on the
  same article), and one transaction per article.
- **Retention** (default 90 days) is applied with Data → Clean up now, or automatically after each
  fetch if enabled. Seen links are kept so old articles are not fetched again.
- **Backups** are zip files of the workspace in `<workspace>/backups/`; the UI offers one before
  every destructive action. Restore from Data → Backup and restore, or `newsrag data restore PATH`.
- **Changing chunk size or the embedding model** applies to stored articles after
  Data → Rebuild index (`newsrag data reindex`). No re-fetching or LLM calls are needed.

## Troubleshooting
| Symptom | Cause and fix |
|---------|---------------|
| `Engine: Rules ... ollama unreachable` | Ollama isn't running. Start it, or ignore: rules mode works. |
| `authentication failed: check the Anthropic API key` | The key is wrong or revoked. The run continues with rules. |
| `This workspace was indexed with X, not Y` | The embedding model changed. Data → Rebuild index. |
| A source shows `error` | Run `newsrag sources --check`. Feeds move; disable or replace it in Sources. |
| `robots.txt disallows ...` | The site does not allow automated fetching of that URL; it cannot be added. |
| "I don't have news on that" | Nothing stored matches. Widen the time range or regions, or fetch first. |
| Chat results include weak matches | Without an LLM, the top results by score are listed. Use an LLM, or raise "Relevance floor" in Settings. |
| UI change to code not visible | Streamlit keeps imported modules cached: restart `newsrag ui`. |

## Project layout and development
```
newsrag/
  defaults/config.yaml   shipped sources, regions, categories
  sources/               rss, gdelt, keyed APIs, polite HTTP (robots, retries)
  pipeline/              fetch, filter, dedupe, rank, chunk (deterministic)
  engines/               RuleEngine, LLMEngine, selection
  llm/                   Claude (SDK), Ollama, OpenAI-compatible, Gemini clients
  store/                 SQLite: items, FTS5 keyword index, vectors, runs, series
  ingest.py runner.py    store new items; one run = fetch -> process -> store
  search.py              hybrid retrieval with rerank and relevance floor
  briefing.py chat.py    digest and grounded chat
  tools/                 typed tool functions + registry (agent-ready)
  ui/                    Streamlit app
tests/                   200 tests; fixtures are synthetic
deploy/                  launchd, cron and GitHub Actions examples
```
Checks:
```bash
.venv/bin/pytest && .venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy
```
Optional git hooks: `.venv/bin/pip install pre-commit && .venv/bin/pre-commit install`.
