# Operations Guide

## Config Model
Config files are loaded from `config/<env>.json` (`dev` or `prod`) and merged with env/CLI overrides.
Startup also loads `.env` in repo root (if present) before config snapshot read.

Validation returns a `ConfigSnapshot`:
- `path`
- `exists`
- `valid`
- `issues`
- `warnings`
- `effective_config`

If invalid, runtime uses a bootstrap-safe config and CLI command gating is enforced.

## Identity and Prompt Templates
System prompt assembly now uses templates in `docs/reference/templates/`:
- `soul_rules.md`
- `evolving_soul.md`
- `identity_card.md`
- `user_preferences_card.md`
- `project_cards.md`
- `todo_cards.md`
- `long_term_snippets.md`

Composer behavior:
- includes evolving soul living-layer notes each turn,
- includes profile + preferences cards every turn,
- includes up to 2 project cards,
- includes top open todos (up to 5),
- includes long-term facts/lessons snippets (up to 10 total),
- appends current session summary.

## Health and Diagnostics
Quick health:
```text
/health
```
Behavior:
- lightweight checks,
- one-line summary of config/data/LLM/Brave/heartbeat,
- LLM reachability timeout target: 3s.

Deep diagnostics:
```text
/admin diagnostics --deep
```
Behavior:
- richer component detail,
- stale lock scan under data root,
- optional Brave live probe when API key exists,
- LLM timeout target: 10s.

## Locking and Atomic Writes
Session and long-term writes use:
- atomic file replace (`tempfile` + `os.replace`),
- per-target lock file `<target>.lock`,
- stale lock reclaim via PID liveness and lock age.

Lock payload shape:
```json
{"pid": 1234, "created_at": "2026-02-24T21:00:00+00:00"}
```

Defaults:
- lock timeout: 10s
- stale age: 1800s (30m)

Timeout behavior:
- raises `FileLockTimeoutError` with structured `error` payload:
  - `code=lock_timeout`
  - `lock_path`
  - `timeout_seconds`
  - `attempts`

Stale lock cleanup API:
- `clean_stale_locks(root, stale_after_seconds=1800)`

## Session Maintenance
Config path:
- `session.maintenance.mode` (`warn` or `enforce`)
- `session.maintenance.prune_after` (default `30d`)
- `session.maintenance.max_entries` (default `500`)
- `session.maintenance.max_disk_bytes` (default `null`)
- `session.maintenance.high_water_ratio` (default `0.8`)

CLI:
- `/session cleanup --dry-run`
- `/session cleanup --enforce`
- optional `--active-key <session_id>`

Execution order:
1. stale prune
2. entry-cap pruning
3. optional disk budget eviction (oldest-first until high-water target)

`warn` mode previews only. `enforce` mode applies deletions.

## Learning Routine
After each user turn, a conservative learning routine can update memory only on explicit statements:
- `my name is ...` updates `profile.name` and appends a `user_fact`,
- `my goal is ...` appends to `profile.long_term_goals` and appends a `user_fact`,
- `please be detailed/concise` updates `preferences.response_style`.

Applied updates are logged to the daily episodic file.

Evolving soul growth:
- persisted at `data/identity/evolving_soul.json`,
- updated with bounded, deduplicated notes from explicit user feedback signals,
- inspect/reset via `/admin soul` and `/admin soul --reset`.

## Web Search Hardening
Tool: `web_search` (Brave)

Reliability behavior:
- in-memory TTL cache keyed by normalized request parameters,
- structured missing-key failure (`missing_brave_api_key`) with action guidance,
- normalized result payload with latency/provider metadata,
- snippet wrapping in explicit untrusted-content boundaries,
- per-result external content metadata:
  - `external_content.untrusted=true`
  - `source=web_search`
  - `wrapped=true`

Automatic usage behavior:
- enabled by `tools.web_search.auto_use=true`,
- trigger-based for time-sensitive/external-knowledge user prompts,
- injects summarized search context into the LLM message stack,
- bounded by `tools.web_search.auto_count` results.

## Tool Inventory
Registered tools:
- `local_search`
- `web_search`
- `check_disk_space`
- `check_docker_containers` (stub)

## Test Coverage (Current)
Primary reliability tests include:
- restricted-mode command gating + invalid config snapshot handling,
- lock timeout + stale lock reclaim + stale lock cleaning,
- session cleanup preview/enforce + maintenance modes,
- quick/deep diagnostics path,
- web search missing-key contract,
- web search cache hit behavior + untrusted wrapping metadata.

Run:
```powershell
pytest -q
```
