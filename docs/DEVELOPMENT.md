# Development Instructions

## Local Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

## Run
```powershell
python -m blaire_core.main --env dev
```

## Test
```powershell
pytest -q
```

## Config Workflow
1. Edit `config/dev.json` (or `config/prod.json`).
2. Optional: add `config/local.json` for machine-local non-secret overrides (gitignored).
3. Keep required sections present: `app`, `paths`, `llm`, `heartbeat`, `tools`, `prompt`, `session`, `logging`.
4. If startup reports invalid config, use `/admin config --effective` and `/admin diagnostics --deep` to inspect issues.

Config precedence:
1. `config/<env>.json`
2. `config/local.json` (optional)
3. env vars / `.env`
4. CLI overrides

## New Runtime Env Flags
- `BLAIRE_EMBEDDING_PROVIDER`: `local` (default) or `hash`.
- `BLAIRE_EMBEDDING_MODEL`: sentence-transformers model id for local embeddings.
- `BLAIRE_EVENT_RETENTION_DAYS`: heartbeat retention window for structured `events`.
- `BLAIRE_DEEP_CUT_ENABLED`: enable/disable daily deep-cut journal artifact generation.

## Admin Memory Commands
- `/admin memory stats`
- `/admin memory recent --limit <N>`
- `/admin memory patterns --limit <N>`
- `/admin memory search <query>`

## Reliability Paths to Recheck After Changes
- restricted-mode command allowlist
- lock stale-reclaim and timeout payload
- session cleanup dry-run/enforce behavior
- `web_search` missing-key and cache behavior
- quick vs deep diagnostics responses
- prompt composer section ordering and template substitution
- learning routine updates to `profile.json` / `preferences.json`
- evolving soul persistence updates in `data/identity/evolving_soul.json`
- `/admin soul` inspect/reset behavior
- structured DB migrations for `data/memory/blaire_memory.db`
- daily summariser cadence and dedupe behavior
- heartbeat event retention pruning behavior

## Commit Guidance
Use focused commits by concern:
1. feature implementation
2. tests
3. docs

Suggested pre-push check:
```powershell
pytest -q
git status --short
```
