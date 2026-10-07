# Daily News Intelligence (Hybrid) — Requirements

A small Python app that collects US, Europe and India news every day, stores it in a searchable local memory,
produces a morning digest, and answers questions in a chat with citations.
It works with **no API keys** (pure Python rules) and **upgrades automatically** when an LLM key is supplied.

## 1. Goals
1. Run fully offline-capable and key-free: free sources, local embeddings, rule-based processing.
2. If an LLM key is present, use the LLM for summaries, entities, chat answers and the digest.
3. If the LLM fails mid-run, fall back to rules for that item and keep going.
4. Every setting a user might tune (top-k, chunk size, model, theme, font size) is editable in the UI.
5. Simple enough that a colleague can build it from this document and `PROMPT.md` with Claude Code.

## 2. Non-goals
- No multi-user hosting, accounts or cloud deployment.
- No paywalled scraping. No full-text scraping unless the user turns it on.
- No investment, legal or medical advice.

## 3. Project rules (the contract)
| # | Rule | Why |
|---|------|-----|
| R1 | **Hybrid by default.** One pipeline, two interchangeable engines (`RuleEngine`, `LLMEngine`) behind one interface. Chosen once at startup, with per-item fallback. | Works with or without a key. |
| R2 | **Code decides, LLM explains.** Filtering, dedupe, ranking, dates, counts and region tagging are plain Python. The LLM only writes summaries, entities, answers and digests. | Reproducible, cheap, testable. |
| R3 | **Keys stay in memory.** Read from env var or the UI password field. Never written to disk, DB, logs, URLs or exception text. Redact on error. | BYOK safety. |
| R4 | **Grounded answers only.** The chat answers from retrieved articles. If nothing relevant is found, it says "I don't have news on that for <range>". | No hallucination. |
| R5 | **Every claim is cited** with headline, source, date and URL. | Verifiability. |
| R6 | **Label provenance.** Each stored item and answer shows `engine = rules` or `engine = llm`. | User knows what wrote the text. |
| R7 | **Be a polite client.** Respect robots.txt and each site's terms, set a User-Agent, cache responses, limit request rate, and retry with backoff. | Avoid bans. |
| R8 | **Source adapters share one shape.** `fetch() -> list[Article]`. Adding a source never touches other modules. | Simplicity. |
| R9 | **Config over code.** Sources, regions, keywords, model and retrieval settings live in `config.yaml`, with UI overrides. No magic numbers in code. | Tunable by colleagues. |
| R10 | **Idempotent daily run.** Re-running never duplicates articles (URL is the key). Failed items are retried next run. | Safe scheduling. |
| R11 | **Fail soft, log clearly.** One bad feed or item never stops the run. A run summary lists successes and failures. | Unattended use. |
| R12 | **Test the deterministic parts.** Unit tests for dedupe, filters, chunking, fallback logic and key redaction. LLM calls are mocked in tests. | Confidence. |
| R13 | **Small and typed.** Python 3.11+, type hints, `ruff` and `mypy` clean, no dependency added without a stated reason. | Maintainable. |
| R14 | **Accessible UI.** Keyboard usable, readable contrast in both themes, adjustable font size, no information conveyed by colour alone. | Usability. |
| R15 | **No invented data.** No fake articles, counts or benchmarks in code, docs or demos. Test fixtures are clearly labelled as fixtures. | Honesty. |

## 4. Data sources
**No key (always on):**
- Publisher RSS feeds, listed in `config.yaml` with `region` (US, EU, IN) and `category` (Technology, Finance, Politics, plus user-defined).
  - US examples: NPR, NYT, CNBC, Federal Reserve press releases.
  - Europe examples: BBC, The Guardian, DW, France24, Euronews, ECB press.
  - India examples: The Hindu, Economic Times, Mint, PIB, RBI, SEBI.
- ~~Google News RSS~~: **not used.** `news.google.com/robots.txt` disallows every `/rss` path (checked 2026-10-07), so it conflicts with R7. Topic searches use GDELT instead.
- GDELT DOC 2.0 API (country and language filters).

**Optional free-tier keys (each enables an extra adapter; absent key = adapter skipped silently):**
GNews, NewsData.io, TheNewsAPI, Currents, NewsAPI.

Feed URLs must be verified when first added, and dead feeds are reported in the run summary.

## 5. Functional requirements
### 5.1 Fetch
- FR1: Run all enabled adapters in parallel with timeouts, retry (3 tries, backoff) and per-source error capture.
- FR2: Normalise to `Article{id, headline, body, url, source, region, category, published_at, fetched_via}`.
- FR3: Follow HTTP redirects to the final URL. Never accept consent or cookie walls to reach an article.

### 5.2 Filter, dedupe, rank
- FR4: Drop articles older than `max_age_hours` (default 48) or matching a category `exclude` pattern. Patterns use word boundaries.
- FR5: Normalise URLs (strip tracking parameters, fragments).
- FR6: Merge near-duplicate headlines (token similarity threshold, default 0.5) and record `also_reported_by`.
- FR7: Rank by number of outlets reporting, then recency. Cap per category and region (`max_per_group`, default 15).

### 5.2.1 Duplicate prevention (four layers)
- DD1 **URL identity:** normalise every URL (lowercase host, drop `www.`, tracking parameters such as `utm_*`, `ref`, `fbclid`, `gclid`, fragments and trailing slashes; prefer `https`). `item_id = sha256(normalised_url)`.
- DD2 **Across runs:** `items.item_id` is a `PRIMARY KEY` and `seen_urls.url` is `UNIQUE`. Writes use `INSERT … ON CONFLICT DO NOTHING`. Seen URLs are skipped **before** processing, so no LLM call is spent on them. `seen_urls` survives cleanup (FR40).
- DD3 **Same story, different URL:**
  - Within a fetch: fuzzy headline match (FR6) merges items and records `also_reported_by`.
  - Across days: before storing, compare with items from the last `dedupe_window_days` (default 3) in the same category, using headline similarity (Jaccard overlap of significant headline words, threshold `dedupe_threshold`) and, when headlines are borderline, embedding similarity (default ≥ 0.90). A match is attached to the existing item as an additional source link; it is not stored as a new item.
  - Merged items keep every source URL, so nothing is lost and the user can see which outlets reported it.
- DD4 **Index writes are idempotent:** `chunk_id = f"{item_id}:{chunk_index}"`, used as the ID in the chunks table, FTS5 and the vector store. Writes are upserts. Each item is ingested in one transaction (item row, chunks, FTS rows, vectors), and the item is marked done only after all succeed. If the vector write fails, the SQLite part is rolled back and the item is retried next run.
- DD5 **Updated articles:** store `content_hash = sha256(clean_text)`. If a later fetch finds the same URL with a different hash, apply `on_update` setting: `ignore` (default) or `replace` (re-process and replace chunks and vectors, keeping the same `item_id`).
- DD6 **Natural key per data type:** every data type declares a natural key and the database enforces it with a `UNIQUE` constraint:
  | Data | Natural key |
  |------|-------------|
  | News | normalised URL (+ DD3 near-duplicate check) |
  | Call transcripts (later) | file content hash, or call ID + date |
  | Numeric series such as gold rates (later) | `(series, date, source)`, upsert so a corrected value replaces the old one |
- DD7 The run summary reports: new, skipped as seen, merged as near-duplicate, updated, failed.
- DD8 Settings → Sources shows `dedupe_window_days`, `dedupe_threshold`, embedding similarity threshold and `on_update`.

### 5.3 Process (engine interface)
- FR8: `Engine.process(article) -> Processed{relevant, category, summary, key_facts[], entities{}, engine}`.
- FR9: `RuleEngine`: keyword relevance, extractive summary (lead sentences), key facts (sentences with numbers), entities by capitalisation and keyword rules. Every output string is copied from the article. TextRank or spaCy may be added later behind the same interface.
- FR10: `LLMEngine`: one JSON-schema response per article, validated with pydantic. On invalid JSON, retry once, then fall back to `RuleEngine` for that item.
- FR11: Engine selection: `mode = auto | rules | llm`. `auto` uses the LLM if one is available: a hosted provider with a key, **or a local model server that responds** (see 5.3.1). Otherwise it uses rules.

#### 5.3.1 Local LLMs (Ollama and similar)
- FR11a: Supported local runtimes: **Ollama** (default, `http://localhost:11434`), plus any **OpenAI-compatible local server** (LM Studio, llama.cpp server, vLLM, Jan). No key is needed for these; the "key" field is hidden.
- FR11b: Settings → LLM shows provider "Local (Ollama)" with: base URL, a model dropdown filled from the server's model list, a "Test connection" button, context length, temperature, max tokens and timeout. Default timeouts are longer for local models (120 s).
- FR11c: Embeddings can also come from Ollama (for example `nomic-embed-text`) instead of `sentence-transformers`, so the whole app runs **fully offline with no keys and no data leaving the machine**.
- FR11d: Small local models often produce invalid JSON. `LLMEngine` uses the runtime's JSON or structured-output mode where available, validates with pydantic, retries once, then falls back to rules for that item (FR10).
- FR11e: Processing is done one article at a time for local models (configurable concurrency, default 1), with a progress bar and a "Stop" button. The README lists sensible model sizes for typical laptops and states that speed depends on hardware.
- FR11f: Per-task model choice: users can pick different models for processing, chat and digest (for example a small fast model for per-article tagging and a larger one for chat).

### 5.4 Store and retrieve
- FR12: SQLite holds articles, seen URLs, run history and errors.
- FR13: Chunk text with configurable `chunk_size` (characters, default 1600) and `chunk_overlap` (default 200). Each chunk repeats headline, source, date, region, category and URL.
- FR14: Embed chunks with a local `sentence-transformers` model by default. A hosted embedding model is optional.
- FR15: Store vectors locally with metadata (`region`, `category`, `published_ts`, `source`). v1 keeps them in the workspace SQLite file (float32 BLOBs, searched with numpy), so rows, keyword index and vectors share one transaction. See ARCHITECTURE.md 5.1.
- FR16: Retrieval is **hybrid search with reranking**, in this order:
  1. **Filter first:** region, category and date range are applied inside both searches, not afterwards.
  2. **Keyword search:** SQLite FTS5 (BM25) over chunk text, returning `candidates_k` results (default 30).
  3. **Semantic search:** vector similarity over chunk embeddings, returning `candidates_k` results.
  4. **Fusion:** merge both lists with reciprocal rank fusion (RRF, constant `k = 60`). A `keyword_weight` setting (0–1, default 0.5) balances the two lists.
  5. **Rerank:** a local cross-encoder (default `cross-encoder/ms-marco-MiniLM-L-6-v2`, via `sentence-transformers`) re-scores the fused candidates against the question. On by default; can be turned off. If the model cannot load, skip this step and log it.
  5b. **Relevance floor:** keep a candidate only if the keyword search matched it or its semantic similarity is at least `min_similarity` (default 0.30). Vector search always returns neighbours, so this is what lets chat say "I don't have news on that" from code. The rerank score is not used as the floor: the cross-encoder gives near-zero scores to relevant passages for short keyword-style queries.
  6. **Group and cut:** keep at most `max_chunks_per_article` (default 2) per article, drop results below `min_score`, return `top_k` (default 8).
- FR16a: Search mode setting: **Hybrid** (default), **Semantic only**, or **Keyword only**. All modes work with no API key.
- FR16b: Each result records which search found it (keyword, semantic or both) and its rerank score, shown in the trace and the Browse page for debugging.
- FR16c: The FTS5 index and the vector index are updated in the same ingest transaction, and both are covered by cleanup, rebuild and "Verify integrity" (FR40, FR41).

### 5.5 Briefing
- FR17: Build a digest grouped by region then category with a "Top of the day" block.
- FR18: Code selects and numbers the articles (last 24 hours, per region and category, most-reported first). The writer (template or LLM) returns lines of headline, why-it-matters and article numbers. Code drops lines citing articles outside their section or not in the list, fills empty sections from the template (with a note), and adds every link itself.
- FR19: Output as HTML and Markdown in `out/`. Optional email via SMTP, with credentials supplied the same way as keys (R3).

### 5.6 Chat
- FR20: Answer from retrieved chunks only, with numbered citations and a Sources list (R4, R5). When retrieval finds nothing relevant, the "I don't have news on that" reply is produced by code without calling the LLM. Citations that do not match a retrieved passage are removed; an answer left with none falls back to the cited article list.
- FR21: Rule mode returns ranked matching articles with snippets and links. LLM mode writes a short answer from them.
- FR22: Honour filters set in the UI (regions, categories, date range) and keep short conversation memory (default 6 turns, configurable).

### 5.7 Running: on demand first, scheduling optional
- FR23: The app is **not** expected to run continuously. Ingest is triggered by the user: an **"Fetch new news"** button in the UI, or `python -m newsrag run`.
- FR24: Ingest is incremental: only new URLs are processed and added (R10). The UI shows progress per source and a summary when done (new, duplicate, skipped, failed).
- FR25: "Catch up" option: when the last run was N days ago, fetch back up to `max_catchup_days` (default 7, limited by what each source still offers) and say which sources could not go back that far.
- FR26: `python -m newsrag chat` gives a terminal chat. `python -m newsrag ui` starts the UI. Scheduling (cron, launchd, GitHub Actions) is optional and documented in the README.

### 5.8 Data location (workspace)
- FR27: All data lives in one **workspace folder**: SQLite DB, vector index, digests, run logs and `settings.json`. Nothing is written outside it.
- FR28: On start, the app asks for the workspace: pick a recent one, choose an existing folder, or create a new one. The last choice is remembered (path only) in the user's app-config folder. `--workspace PATH` overrides it on the CLI.
- FR29: Workspaces are portable: copying the folder to another machine or cloud-synced drive works. A workspace records its schema version and the embedding model used, and refuses to mix embeddings from a different model (offers re-index).
- FR30: Multiple workspaces are allowed (for example "work" and "personal"). Switching does not need a restart.
- FR31: No login or accounts. "Opening a workspace" is the start step. A workspace can optionally be protected with a passphrase in a later version (not v1).

### 5.9 Source (feed) management
- FR32: The **Sources** page lists every configured source in a table: name, type (RSS, GDELT, API), region, category, URL, enabled, last fetch time, last status (ok, empty, error + message), items added last run.
- FR33: Users can **add** a source: paste an RSS URL, the app validates it (fetches it, parses it, shows the feed title and 3 latest headlines) before saving. Region and category are chosen from lists or a new value is typed.
- FR34: Users can edit, enable or disable, test, and delete a source. Bulk import and export as OPML or YAML.
- FR35: GDELT sources are added as a query plus region (for example "RBI repo rate", India), not a raw URL. Before saving any new source, the app checks the site's robots.txt and refuses disallowed URLs.
- FR36: The default sources ship in `config.yaml`. User changes are stored in the workspace, so the shipped defaults are never edited and can be restored.
- FR37: A source that fails 5 runs in a row is flagged "needs attention" (not deleted).

### 5.10 Data retention and cleanup
- FR38: Retention policy in Settings: keep articles for `retention_days` (default 90, or "forever"). Optionally keep digests longer than articles.
- FR39: Cleanup is **manual by default**: a "Clean up now" button shows a preview (how many articles, chunks and MB will be removed) and needs confirmation. Optional "clean up automatically after each ingest".
- FR40: Cleanup removes the article, its chunks and its vectors together, so the DB and the vector index never disagree. The seen-URL record is kept (lighter) so old articles are not re-fetched.
- FR41: Other maintenance actions: delete by source, by region or category, or by date range; "Compact database" (VACUUM); "Rebuild vector index" (needed after changing chunk size or embedding model); "Verify integrity" (finds orphan chunks or missing vectors and repairs them).
- FR42: "Back up workspace" creates a dated zip of the workspace. "Restore" imports one. Every destructive action offers a backup first.
- FR43: The Run page shows workspace size, article count by region and category, and oldest and newest article dates.

## 6. UI requirements (Streamlit)
Start screen: **Open workspace** (recent list, choose folder, create new). Then pages: **Today** (digest), **Ask** (chat), **Browse** (search and filter articles), **Fetch** (fetch new news button, progress, run history), **Sources** (feed table, add/test/edit), **Data** (stats, cleanup, backup, re-index), **Settings**.

**Settings panel (all persisted to `settings.json`, never including keys):**
| Group | Controls |
|-------|----------|
| Appearance | Set from the **top-right header on every page** (also mirrored here): font size as three buttons **A / A+ / A++** (regular 16 px, large 18 px, extra large 20 px, applied to all text); theme as icon buttons **sun (Light) / moon (Dark) / monitor (System)**. Each button has a tooltip and an accessible label, and the active one is highlighted. Choices persist in the workspace. Optional high-contrast toggle. |
| Mode | Engine: Auto / Rules only / LLM only. Shows a badge for the active engine. |
| LLM | Provider (Anthropic, Google Gemini, OpenAI-compatible, **Local: Ollama / OpenAI-compatible local server**), base URL (local only), model name or dropdown, **API key (password field, memory only; hidden for local)**, per-task model (processing, chat, digest), temperature, max output tokens, context length, request timeout, concurrency, "Test connection" button. |
| Data | Workspace path (read-only display, "Switch workspace"), `retention_days`, auto-cleanup on or off, backup folder. |
| Retrieval | Search mode (Hybrid / Semantic / Keyword), `top_k`, `candidates_k`, `keyword_weight`, rerank on or off, rerank model, `max_chunks_per_article`, `min_score`, `min_similarity`, chunk size, chunk overlap, embedding model, conversation memory length. Changing chunk settings or the embedding model offers "Re-index". |
| Sources | Toggle regions and categories, enable or disable each feed, add a custom RSS URL, `max_age_hours`, `max_per_group`, dedupe threshold. |
| Briefing | Items per category, language style (brief / detailed), email on or off. |
| Safety | Show/hide provenance badges (default on), "Clear stored data" with confirmation. |

Every control has a one-line help tooltip and a "Reset to defaults" button.

## 7. Tools and libraries
| Purpose | Tool |
|---------|------|
| Language, packaging | Python 3.11+, `uv` or `pip`, `pyproject.toml` |
| HTTP | `httpx` (async), `tenacity` for retries |
| Feeds | `feedparser` |
| Article text (optional) | `trafilatura` |
| Dedupe | Jaccard overlap of significant headline words (standard library, no dependency) |
| Models and validation | `pydantic` v2, `pyyaml` |
| Storage | `sqlite3` (stdlib): rows, FTS5 keyword index and vectors in one file; `numpy` for vector search |
| Local embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Keyword search | SQLite FTS5 (built into Python's `sqlite3`) |
| Reranking | `sentence-transformers` `CrossEncoder` (`ms-marco-MiniLM-L-6-v2`) |
| Rule-based NLP | Standard-library extractive rules (no dependency); TextRank/spaCy optional later |
| LLM (optional) | Claude via the official `anthropic` SDK (`pip install newsrag[anthropic]`; default model `claude-opus-5-5`, structured outputs, refusal fallbacks); Gemini and OpenAI-compatible providers via `httpx` |
| Local LLM (optional) | Ollama via its HTTP API (`httpx`, no extra SDK needed) or any OpenAI-compatible local server |
| Feed import/export | OPML parsing with stdlib `xml.etree` |
| UI | `streamlit` |
| Scheduling | cron, launchd, or GitHub Actions |
| Quality | `pytest`, `ruff`, `mypy`, `pre-commit` |

## 8. Suggested layout
```
newsrag/
  config.yaml            # sources, regions, categories, defaults
  newsrag/
    models.py            # Article, Processed, Settings (pydantic)
    sources/             # rss.py, gdelt.py, keyed.py (gnews, newsdata), http.py, parsing.py
    pipeline/            # fetch.py, filter.py, dedupe.py, chunk.py
    engines/             # base.py (Engine protocol), rules.py, llm.py, select.py
    store/               # db.py (SQLite), vectors.py
    tools/               # tool-shaped functions + registry (section 12)
    briefing.py
    chat.py
    ui/app.py            # Streamlit entry
    __main__.py          # CLI: run | chat | ui
  tests/
  README.md
```

## 9. Acceptance criteria
1. With no keys set, `run` completes, `out/` has a digest, and `Ask` returns cited articles for a test question.
2. With a key set, summaries, answers and the digest are LLM-written and labelled `engine = llm`.
3. With an invalid key, the run completes using rules and the run log shows the fallback reason with no key text.
4. Re-running the same day adds zero duplicate articles.
5. Searching with a region and date filter returns only matching articles.
6. Changing `top_k`, chunk size, theme and font size in the UI takes effect without editing code.
7. `grep` for the key across the repo, database, logs and `settings.json` finds nothing.
8. Tests, `ruff` and `mypy` pass. Adding a new RSS feed requires editing `config.yaml` only.
9. A feed added in the Sources page is validated before saving, and an invalid URL is rejected with a clear message.
10. Choosing a new workspace folder at start puts every file the app writes inside that folder, and nothing elsewhere except the remembered path.
11. "Clean up now" with `retention_days = 30` removes exactly the older articles together with their chunks and vectors, and "Verify integrity" then reports no orphans.
12. With Ollama running and no keys set, `auto` mode uses the local model, and with Ollama stopped it falls back to rules without failing.
13. Every tool in section 12.1 can be called directly from a test with no LLM, and the registry produces a valid JSON schema for each one.
14. A question containing an exact term (for example a ticker or a percentage) that only appears in one stored article returns that article in Hybrid mode. With rerank on, the trace shows rerank scores. With rerank turned off or its model unavailable, search still returns results.
15. **Duplicate tests** (using labelled fixtures):
    - Ingesting the same fixture set twice leaves item, chunk, FTS and vector counts unchanged, and the second run reports everything as "skipped as seen".
    - Two URLs differing only by tracking parameters, `www.`, fragment or trailing slash produce one item.
    - The same story from two outlets on consecutive days produces one item with both sources listed.
    - Killing the process during a vector write and re-running leaves no orphan or duplicate chunks ("Verify integrity" reports clean).
    - Re-ingesting a gold-rate style row for an existing `(series, date, source)` updates the value instead of adding a row (tested on the generic store, even though the gold-rate app is not built in v1).

## 10. Known limits
- RSS often gives only a snippet. Rule-based summaries are weaker than LLM ones.
- Free APIs have daily quotas and may change terms. GDELT throttles to one request per 5 seconds and may reject bursts.
- Feed URLs change over time and need occasional maintenance.
- Local models are slower and less reliable at structured output than hosted ones. Quality depends on the model and hardware.
- `region` is the region of the **outlet's feed**, not of the story: a US outlet's article about India is tagged US. Region filters therefore narrow by outlet. Story-level region tagging is a later improvement.
- Changing URL-normalisation rules (for example adding a tracking parameter) can make an already-stored article look new once; the cross-day near-duplicate check (DD3) merges it.

## 11. Reusing this as a base for other daily-data apps
The design is meant to be reused for other "collect daily, store, ask questions" tools, such as call transcripts or gold rates. To make that easy, build v1 with these seams:

| Seam | What stays generic | What a new use case supplies |
|------|-------------------|------------------------------|
| `Item` model | `id, title, body, url_or_ref, source, tags{}, occurred_at, ingested_at, engine` | Domain fields go in `tags` or a typed extension |
| Source adapter | `fetch(since) -> list[Item]` | A new adapter: RSS, folder watcher, file upload, API |
| Processor | `Engine.process(item)` with rules and LLM versions | Domain prompt and rule set (for example "extract action items" for calls) |
| Store | SQLite + vectors + retention + backup | Nothing |
| Views | Digest, chat, browse, data, settings | Optional domain page (for example a chart) |

Two kinds of data need different storage, and v1 should support both:
- **Text data** (news, call transcripts, meeting notes, emails you export): chunk, embed, retrieve. This is RAG.
- **Numeric time series** (gold rates, FX, fuel prices, stock closes): store as rows in a SQLite table (`series, date, value, unit, source`). Answer questions with **SQL and charts, not embeddings**. The LLM may only explain numbers that code has already computed (R2), for example "gold rose 2.1% this week" is calculated in code, and the LLM writes the sentence around it.

Example adaptations (not part of v1):
- **Call transcripts:** source = a watched folder or file upload (TXT, VTT, DOCX). Processing = summary, decisions, action items, people mentioned. Privacy: transcripts stay in the workspace, local LLM recommended, optional redaction of phone numbers and emails before any hosted LLM call.
- **Gold rates:** source = a free rates API or a published page, fetched once a day. Storage = time series table. Views = chart with 7, 30 and 90-day change, plus a daily note. Rule mode alone covers this fully.

## 12. Agent-ready design
v1 is a **pipeline**: code decides every step and the LLM only writes text. A later version may add an **agent** for the Ask page, where the LLM chooses which tools to call in a loop. v1 must make that addition cheap.

### 12.1 Required in v1
- AG1: Core operations are **tool-shaped functions** in `newsrag/tools/`: typed parameters, a pydantic result model, a one-line docstring, and no UI or printing inside. The UI, CLI and pipeline call these functions; they never duplicate their logic.
- AG2: Minimum tool set:
  | Tool | Purpose | Changes data? |
  |------|---------|---------------|
  | `search_news(query, regions, categories, date_from, date_to, top_k)` | Semantic search with metadata filters | No |
  | `get_article(id)` | Full stored article and its metadata | No |
  | `list_sources(region?, category?)` | Configured sources and their status | No |
  | `stats(date_from?, date_to?)` | Counts by region, category and source | No |
  | `compare_periods(query, period_a, period_b)` | Run the same search over two date ranges and return both result sets | No |
  | `fetch_now(regions?)` | Trigger an incremental fetch | **Yes** |
  | `cleanup(retention_days, dry_run=True)` | Preview or apply retention cleanup | **Yes** |
- AG3: Each tool declares `changes_data: bool`. Tools that change data default to a dry run or preview.
- AG4: A tool registry (`tools/registry.py`) lists all tools with their JSON schema, generated from the pydantic models, so any LLM provider's tool-calling format can be produced from it.
- AG5: Unit tests call every tool directly, without any LLM.

### 12.2 Later version: agent mode
- AG6: Ask page mode selector: **Simple** (today's retrieve-then-answer) or **Agent** (LLM chooses tools in a loop).
- AG7: The agent loop has a step limit (default 6), a time limit (default 90 s), and stops with a partial answer and an explanation when a limit is hit.
- AG8: Read-only tools run without asking. Tools with `changes_data = true` require user confirmation in the UI before running. The agent cannot call `cleanup` with `dry_run = False` itself.
- AG9: A collapsible **trace** under each answer lists every step: tool, arguments, result count, and time taken. Traces are stored in the workspace with the answer and never contain API keys.
- AG10: Answers follow the same rules as Simple mode: grounded, cited (R4, R5), labelled with provenance (R6). Numbers are computed by tools, not by the LLM (R2).
- AG11: **Fallback:** if no LLM is available, or the selected model does not support tool calling (common with small local models), Ask uses Simple mode and says why.
- AG12: Settings: agent on or off, max steps, time limit, which tools are enabled, and "always confirm" for data-changing tools.
- AG13: Evaluation: the evaluation set from section 13 is run in both modes, and Agent mode must cite the expected sources at least as often as Simple mode before it is made the default.

## 13. Candidate features for later versions
Not required for v1. Listed so the design leaves room for them:
1. **Watchlists and alerts:** save topics or entities ("RBI", "Nvidia", "ECB rates") and highlight matching new items after each fetch.
2. **Saved questions:** re-run a saved question after each fetch and show what changed since last time.
3. **Story timeline:** group articles about the same event over days and show how the story developed.
4. **Weekly digest:** roll up 7 daily digests into one.
5. **Export:** digest or search results to Markdown, PDF or CSV.
6. **Multilingual sources:** fetch non-English European or Indian sources, translate with the LLM (labelled `engine = llm`), keep the original link.
7. **Source quality score:** track per-source duplicate rate, error rate and how often its articles are cited.
8. **Cost and usage meter:** tokens and estimated cost per run for hosted LLMs, time per item for local ones.
9. **Evaluation set:** a small list of questions with expected sources, used to compare `top_k`, chunk size and models objectively.
10. **Workspace passphrase:** optional encryption of the workspace at rest.
