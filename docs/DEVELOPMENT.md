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
2. Keep required sections present: `app`, `paths`, `llm`, `heartbeat`, `tools`, `prompt`, `session`, `logging`.
3. If startup reports invalid config, use `/admin config` and `/admin diagnostics --deep` to inspect issues.

## Reliability Paths to Recheck After Changes
- restricted-mode command allowlist
- lock stale-reclaim and timeout payload
- session cleanup dry-run/enforce behavior
- `web_search` missing-key and cache behavior
- quick vs deep diagnostics responses

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
