# Acceptance Report (REQUIREMENTS.md section 9)

Run on 2026-10-07, branch `stage-7/docs-acceptance`, macOS, Python 3.12, against live feeds and
the real local embedding and rerank models. **No LLM was available on this machine** (no Ollama,
no API key), so every live run used the rules engine; the LLM path is verified with mocked
providers only. Paths in the raw output are shortened to `ws/` (the fresh workspace) and
`appcfg/` (the app-config folder, set with `NEWSRAG_CONFIG_DIR`).

## Summary

| # | Criterion (short) | Result | Evidence |
|---|-------------------|--------|----------|
| 1 | No keys: run completes, digest written, Ask returns cited articles | **Pass** | Live run A1. Digest is in `<workspace>/digests/`, not `out/`: FR27 keeps all output in the workspace |
| 2 | With a key: summaries, answers, digest LLM-written and labelled `llm` | **Pass with mocks only; not verified live** | `test_llm_engine_happy_path`, `test_llm_digest_happy_path`, `test_llm_answer_grounded_and_sources_built_by_code` |
| 3 | Invalid key: run completes on rules, reason in run log, no key text | **Pass** | Live run A3 |
| 4 | Re-running adds zero duplicates | **Pass** | Live run A4 (the 1 new item was a newly published article, checked) + `test_ingest_twice_changes_nothing` |
| 5 | Region and date filters return only matching articles | **Pass** | `test_region_and_date_filters_are_applied_in_the_query`, `test_read_only_tools_run_without_llm`; live CLI search with `--region EU` (stage 4) |
| 6 | `top_k`, chunk size, theme, font size change from the UI without code edits | **Pass** | `test_settings_save_changes` (top_k); chunk size via Settings + Data → Rebuild index (`test_reindex_applies_new_chunk_size_without_refetch`); theme and font size checked live in the browser (stage 6) |
| 7 | Key not found in repo, database, logs, `settings.json` | **Pass** | Live run A7 + `test_keys_are_memory_only` (UI) + `test_logs_on_disk_never_contain_keys` |
| 8 | Tests, ruff, mypy pass; new feed needs `config.yaml` only | **Pass** | A8 output; feeds are data in `config.yaml` (or added in the UI) |
| 9 | Sources page validates a feed; invalid URL rejected with a clear message | **Pass** | `test_check_feed_ok_blocked_and_not_a_feed` (robots-blocked and non-feed URLs rejected with messages), `test_sources_add_feed_after_check` |
| 10 | Everything written stays in the workspace, except the remembered path | **Pass (documented exception)** | Live A10: all app data is in the workspace; only `recent.json` is outside. The embedding and rerank models live in the shared Hugging Face cache (`~/.cache/huggingface`), by decision (see below) |
| 11 | Cleanup with retention removes exactly older articles + chunks + vectors; verify clean | **Pass** | `test_cleanup_removes_old_items_and_their_rows`; live preview in stage 4 and the UI |
| 12 | Ollama running: `auto` uses it; Ollama stopped: falls back to rules | **Stopped case live; running case with mocks only** | Every live run here had Ollama stopped and fell back to rules. `test_select_auto_with_working_ollama`, `test_select_check_failure_closes_client`, `test_ollama_missing_model_and_down_server` |
| 13 | Every tool callable with no LLM; valid JSON schema for each | **Pass** | `test_read_only_tools_run_without_llm`, `test_schemas_are_valid_json_objects`, `test_registry_has_the_required_tools_and_flags` |
| 14 | Exact term found in Hybrid; rerank scores shown; works without rerank | **Pass** | `test_exact_term_found_in_hybrid`, `test_rerank_scores_shown_and_order_follows_rerank`, `test_search_works_without_or_with_broken_reranker[None/reranker1]` |
| 15 | Duplicate tests (five cases) | **Pass** | `test_ingest_twice_changes_nothing`, `test_url_variants_are_one_item`, `test_same_story_next_day_from_another_outlet_is_merged`, `test_crash_during_vector_write_leaves_nothing_then_rerun_is_clean`, `test_series_upsert_replaces_correction` |

**13 pass (10 with a documented exception), 2 verified with mocks only (2, and the "Ollama running" half of 12).**

## Decision (criterion 10): shared model cache
Decided by Siva on 2026-10-07: keep the downloaded models in the shared Hugging Face cache
(`~/.cache/huggingface`). Models are tools the app uses, like Python itself, not user data:
one ~180 MB download serves every workspace, later runs work offline, and workspace backups
stay small. Rejected: a per-workspace `models/` folder (~180 MB per workspace).

## To close the mock-only items
Install Ollama and a model, then repeat A1 and A3:
```bash
ollama pull llama3.2
.venv/bin/python -m newsrag --workspace ~/NewsRAG/test run --limit 10
.venv/bin/python -m newsrag --workspace ~/NewsRAG/test chat "What did the RBI decide?"
```
Expect `Engine: LLM · ollama (llama3.2...)`, `Written by: llm N`, a digest labelled LLM, and an
answer with a `Sources:` list. Then stop Ollama and run again: it must fall back to rules.

## Raw output

### A1 - fresh workspace, no keys
```
Workspace: ws
Schema:    1
Embedding: not indexed yet

Engine: Rules | no LLM available, using rules (ollama: ollama unreachable at http://localhost:11434)
New 30 | skipped as seen 0 | merged duplicates 0 | updated 0 | not relevant 0 | failed 0
Written by: rules 30
Digest (Template, 10 articles): ws/digests/2026-10-07.html
2026-10-07.html
2026-10-07.json
2026-10-07.md
Answering with: Rules | no LLM available, using rules (ollama: ollama unreachable at http://localhost:11434)

Here is what I found for the last 3 days (no LLM; article summaries):

[1] Results of the September 2026 survey on credit terms and conditions in euro-denominated securities financing and OTC derivatives markets (SESFOD) (ECB press, 2026-10-07)
```

### A3 - invalid Anthropic key (`ANTHROPIC_API_KEY=<FAKE-KEY>`)
```
Engine: Rules | no LLM available, using rules (ollama: ollama unreachable at http://localhost:11434; anthropic: authentication failed: check the Anthropic API key)
New 10 | skipped as seen 30 | merged duplicates 0 | updated 0 | not relevant 0 | failed 0
Written by: rules 10
Digest (Template, 15 articles): ws/digests/2026-10-07.html
-- last run record (engine_reason):
Rules | no LLM available, using rules (ollama: ollama unreachable at http://localhost:11434; anthropic: authentication failed: check the Anthropic API key)
```

### A4 - the same run again
```
New 1 | skipped as seen 39 | merged duplicates 0 | updated 0 | not relevant 0 | failed 0
orphan keyword rows 0 | orphan chunks 0 | chunks missing vectors 0 | chunks missing keyword rows 0 | clean
```
Checking the one new item:
```
items: 41 | distinct urls: 41 | distinct ids: 41
newest stored item: ‘Now is the moment’: housing activists at Madrid encampment pin wary hopes on Spain’s earl | Guardian Europe
closest other headline: 0.00 | İzmir's centre-left mayor joins Erdoğan's AK Party, shocking Turkish opposition
```
A new article published between the two runs, not a duplicate.

### A7 - searching for the key
```
files containing the key under ws: 0
files containing the key under appcfg: 0
files containing the key under Siva-News-RAG-Daily: 0
```
(`grep -r -a`, so the binary database file is included; `.venv` excluded.)

### A8 - checks
```
200 passed in 4.99s
All checks passed!
78 files already formatted
Success: no issues found in 73 source files
```

### A10 - every file the app wrote
```
-- workspace:
  digests/2026-10-07.html
  digests/2026-10-07.json
  digests/2026-10-07.md
  logs/newsrag.log
  newsrag.db
  settings.json
  workspace.json
-- app config folder:
  recent.json
-- repo working tree changes from the runs:
  none
-- model cache outside the workspace:
  /Users/siva/.cache/huggingface (599M)
-- written there since this acceptance run started:
  files:        0
```

### Mapped tests
```
tests/test_engines.py::test_llm_engine_happy_path PASSED
tests/test_briefing.py::test_llm_digest_happy_path PASSED
tests/test_chat.py::test_llm_answer_grounded_and_sources_built_by_code PASSED
tests/test_search.py::test_region_and_date_filters_are_applied_in_the_query PASSED
tests/test_tools.py::test_read_only_tools_run_without_llm PASSED
tests/test_ui.py::test_settings_save_changes PASSED
tests/test_ui.py::test_keys_are_memory_only PASSED
tests/test_store.py::test_reindex_applies_new_chunk_size_without_refetch PASSED
tests/test_sources_config.py::test_check_feed_ok_blocked_and_not_a_feed PASSED
tests/test_ui.py::test_sources_add_feed_after_check PASSED
tests/test_workspace.py::test_cli_writes_only_inside_workspace PASSED
tests/test_store.py::test_cleanup_removes_old_items_and_their_rows PASSED
tests/test_engines.py::test_select_auto_with_working_ollama PASSED
tests/test_engines.py::test_select_check_failure_closes_client PASSED
tests/test_llm_clients.py::test_ollama_missing_model_and_down_server PASSED
tests/test_tools.py::test_schemas_are_valid_json_objects PASSED
tests/test_tools.py::test_registry_has_the_required_tools_and_flags PASSED
tests/test_search.py::test_exact_term_found_in_hybrid PASSED
tests/test_search.py::test_rerank_scores_shown_and_order_follows_rerank PASSED
tests/test_search.py::test_search_works_without_or_with_broken_reranker[None] PASSED
tests/test_search.py::test_search_works_without_or_with_broken_reranker[reranker1] PASSED
tests/test_store.py::test_ingest_twice_changes_nothing PASSED
tests/test_store.py::test_url_variants_are_one_item PASSED
tests/test_store.py::test_same_story_next_day_from_another_outlet_is_merged PASSED
tests/test_store.py::test_crash_during_vector_write_leaves_nothing_then_rerun_is_clean PASSED
tests/test_store.py::test_series_upsert_replaces_correction PASSED
```
