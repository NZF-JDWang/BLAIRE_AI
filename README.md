# BLAIRE Core v0.1

BLAIRE Core is a local-first assistant runtime with:
- config snapshot validation and diagnostics-only startup mode when config is invalid,
- file-backed session/memory persistence with stale-reclaim lock files,
- Ollama chat orchestration,
- template-driven prompt composition for soul/identity/user context,
- conservative learning updates from explicit user statements,
- hardened Brave web search tool behavior,
- quick and deep health diagnostics,
- session maintenance preview/enforce controls.

## Requirements
- Python 3.11+
- Optional: local Ollama server for chat (`/api/chat`, `/api/tags`)
- Optional: Brave Search API key for `web_search`

## Install
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

## Run CLI
```powershell
python -m blaire_core.main --env dev
```

Optional overrides:
```powershell
python -m blaire_core.main --env dev --llm-model llama3.1 --heartbeat-interval 0 --data-path ./data
```

## Environment Overrides
- `BLAIRE_LLM_BASE_URL`
- `BLAIRE_LLM_MODEL`
- `BLAIRE_DATA_PATH`
- `BLAIRE_HEARTBEAT_INTERVAL`
- `BLAIRE_BRAVE_API_KEY`
- `BLAIRE_LOG_LEVEL`

## Core CLI Commands
- `/help`
- `/exit`
- `/health`
- `/admin status`
- `/admin config`
- `/admin diagnostics [--deep]`
- `/admin memory`
- `/admin soul [--reset]`
- `/heartbeat tick|start|stop|status`
- `/tool <name> <json_args>`
- `/session new|list|use|current`
- `/session cleanup --dry-run|--enforce [--active-key <session_id>]`

## Restricted Mode (Invalid Config)
If config validation fails at startup, BLAIRE enters diagnostics-only mode.

Allowed commands:
- `/health`
- `/admin status`
- `/admin config`
- `/admin diagnostics [--deep]`
- `/help`
- `/exit`

All other slash commands are blocked until config is fixed.

## Tests
```powershell
pytest -q
```

## v0.1 Scope Notes
- `/doctor` is intentionally not included.
- Config repair is manual (edit config files), not automatic.
- Docker checks are currently a stub tool.

## Prompt Templates
Template files are in `docs/reference/templates/` and are used to compose the system prompt each turn:
- soul rules
- evolving soul (living layer)
- identity card (profile)
- user preferences card
- project cards
- todo focus
- long-term snippets

See [docs/OPERATIONS.md](docs/OPERATIONS.md) for runtime behavior and reliability details.
