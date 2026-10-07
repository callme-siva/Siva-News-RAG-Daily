# Prompt for Claude Code

Copy everything below the line into Claude Code in an empty folder that also contains `REQUIREMENTS.md`.

---

You are building a small Python application called **newsrag**: a daily news intelligence tool for US, Europe and India news.

## Before you write any code
1. Read `REQUIREMENTS.md` and `ARCHITECTURE.md` completely. If they disagree, `REQUIREMENTS.md` wins; tell me about the conflict. It is the contract. Do not edit, skip or loosen its acceptance criteria (section 9) or rules (section 3).
2. Reply first with: (a) any questions or ambiguities, (b) your assumptions, (c) the build plan as numbered stages. Wait for me to confirm before coding.

## How to build
- Build in these stages, committing after each one on its own branch and keeping the tests green:
  1. **Skeleton:** `pyproject.toml`, package layout, pydantic models, `config.yaml` loader, settings persistence (no keys), logging with key redaction.
  2. **Sources and pipeline:** RSS and GDELT adapters (not Google News RSS: its robots.txt disallows /rss); verify every feed URL and its robots.txt before adding it; optional key-based adapters that are skipped silently when no key exists; filter, dedupe, rank. Unit tests with recorded fixtures (label them as fixtures).
  3. **Engines:** `Engine` protocol, `RuleEngine`, `LLMEngine`, selector (`auto | rules | llm`), per-item fallback, pydantic validation of LLM JSON. Mock the LLM in tests.
  4. **Store and retrieval:** duplicate prevention exactly as in section 5.2.1 (DD1–DD8), enforced by database constraints and deterministic IDs, with the duplicate tests from acceptance criterion 15. SQLite, chunking with configurable size and overlap, local embeddings, vector store, and hybrid retrieval as in FR16: filters first, FTS5 keyword search plus semantic search, reciprocal rank fusion, local cross-encoder rerank (toggle, with graceful skip), grouping by article, `top_k` and `min_score`.
  5. **Briefing, chat and tools:** digest where code picks and numbers the articles and adds all links (template or LLM writes only the lines); grounded chat with code-validated citations and a code-produced "I don't have news on that" reply; the section 12.1 tool functions and registry.
  6. **UI:** Streamlit with an "Open workspace" start screen, then pages Today, Ask, Browse, Fetch (on-demand "Fetch new news" with progress), Sources (feed table with add, validate, test, edit, disable, OPML import/export), Data (stats, retention cleanup with preview, backup/restore, re-index, verify integrity), and Settings. Local LLMs (Ollama and OpenAI-compatible local servers) must be selectable with a model dropdown and no key. The Settings page must expose: theme as sun / moon / monitor icon buttons and font size as A / A+ / A++ buttons (16, 18, 20 px), both placed in the top-right header of every page, engine mode, LLM provider, model, API key (password field, memory only), temperature, max tokens, timeout, "Test connection", `top_k`, `min_score`, chunk size, chunk overlap, embedding model, memory length, source toggles, `max_age_hours`, `max_per_group`, dedupe threshold, and Reset to defaults.
  7. **Docs, scheduling and acceptance:** README with setup, run, scheduling examples (cron, launchd, GitHub Actions), how to add a feed, and troubleshooting. Then run every acceptance criterion in section 9 live where possible and write `ACCEPTANCE.md`: pass, partial, or verified-with-mocks-only for each, with raw output. Never mark something as passed that was not actually run.
- Implement section 12.1 of `REQUIREMENTS.md` (tool-shaped functions and the tool registry) in v1, and have the UI, CLI and pipeline call those functions. Do **not** build the agent loop (section 12.2) now.
- Build the generic seams from section 11 of `REQUIREMENTS.md` (`Item` model, source adapter, processor, store) so the same base can later hold other daily data such as call transcripts or price series. Do not build those other use cases now.
- Keep changes small and typed. Do not add a dependency unless you state why. Prefer the standard library when it is enough.

## Non-negotiables
- The app must work with **no API keys at all**, and use the LLM automatically when a key is supplied. If an LLM call fails, fall back to rules for that item and continue.
- Code does filtering, dedupe, ranking, dates and counts. The LLM only writes summaries, entities, answers and digests.
- API keys live in memory only: never in files, the database, logs, URLs or exception messages.
- Chat answers come only from retrieved articles, with numbered citations, and say so when nothing is found.
- Label each summary and answer as `engine = rules` or `engine = llm`.
- Never invent articles, numbers or sources. Do not guess feed URLs: verify each one before adding it, and report dead ones.

## Definition of done
- Every acceptance criterion in section 9 of `REQUIREMENTS.md` passes, and you show the evidence (raw command output for `pytest`, `ruff`, `mypy`, a no-key run, a run with an invalid key, and a repeat run showing zero duplicates).
- Finish with a short report: what was built, what was skipped, known limits, and how to run it.
