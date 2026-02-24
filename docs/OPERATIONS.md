# Operations Guide

## Config Model
Config files are loaded from `config/<env>.json` (`dev` or `prod`) and merged with env/CLI overrides.

Validation returns a `ConfigSnapshot`:
- `path`
- `exists`
- `valid`
- `issues`
- `warnings`
- `effective_config`

If invalid, runtime uses a bootstrap-safe config and CLI command gating is enforced.

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
