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

## .env Support
You can place local secrets and overrides in `.env` (auto-loaded on startup).

1. Copy template:
```powershell
Copy-Item .env.example .env
```
2. Set values (especially `BLAIRE_BRAVE_API_KEY`).
3. Run CLI normally.

Existing shell environment variables still win over `.env` values.

## Config Precedence (Single Place Strategy)
Recommended setup:
- Keep shared defaults in `config/dev.json` or `config/prod.json`.
- Keep machine-local, non-secret overrides in `config/local.json` (gitignored).
- Keep secrets in `.env`.

Precedence order (lowest to highest):
1. `config/<env>.json`
2. `config/local.json` (optional)
3. environment variables / `.env`
4. CLI overrides (`--llm-model`, `--heartbeat-interval`, etc.)

## Environment Overrides
- `BLAIRE_LLM_BASE_URL`
- `BLAIRE_LLM_MODEL`
- `BLAIRE_LLM_TEMPERATURE`
- `BLAIRE_LLM_TOP_P`
- `BLAIRE_LLM_REPEAT_PENALTY`
- `BLAIRE_LLM_NUM_CTX`
- `BLAIRE_DATA_PATH`
- `BLAIRE_HEARTBEAT_INTERVAL`
- `BLAIRE_BRAVE_API_KEY`
- `BLAIRE_LOG_LEVEL`
- `BLAIRE_TELEGRAM_ENABLED`
- `BLAIRE_TELEGRAM_BOT_TOKEN`
- `BLAIRE_TELEGRAM_CHAT_ID`
- `BLAIRE_TELEGRAM_POLLING_ENABLED`
- `BLAIRE_TELEGRAM_POLLING_TIMEOUT_SECONDS`
- `BLAIRE_EMBEDDING_PROVIDER` (`local` or `hash`; default `local` with fallback)
- `BLAIRE_EMBEDDING_MODEL` (default `sentence-transformers/all-MiniLM-L6-v2`)
- `BLAIRE_EVENT_RETENTION_DAYS` (default `30`)
- `BLAIRE_DEEP_CUT_ENABLED` (`true|false`)

## Core CLI Commands
- `/help`
- `/exit`
- `/health`
- `/admin status`
- `/admin config`
- `/admin config --effective`
- `/admin diagnostics [--deep]`
- `/admin memory stats`
- `/admin memory recent --limit <N>`
- `/admin memory patterns --limit <N>`
- `/admin memory search <query>`
- `/admin soul [--reset]`
- `/heartbeat tick|start|stop|status`
- `/telegram test "<message>"`
- `/telegram listen`
- `/telegram start|stop|status`
- `/tool <name> <json_args>`
- `/session new|list|use|current`
- `/session cleanup --dry-run|--enforce [--active-key <session_id>]`

## Telegram Two-Way Text
Set these env vars to enable text send/receive:

```powershell
$env:BLAIRE_TELEGRAM_ENABLED="true"
$env:BLAIRE_TELEGRAM_BOT_TOKEN="<your bot token>"
$env:BLAIRE_TELEGRAM_CHAT_ID="<your numeric chat id>"
$env:BLAIRE_TELEGRAM_POLLING_ENABLED="true"
$env:BLAIRE_TELEGRAM_POLLING_TIMEOUT_SECONDS="20"
```

Then run:

```powershell
python -m blaire_core.main --env dev
```

Behavior:
- outbound text from the model is sent to Telegram via `notify_user(...)`
- inbound text from the configured chat is polled and routed into the model
- polling starts automatically when `BLAIRE_TELEGRAM_POLLING_ENABLED=true`

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

## Auto Web Search
`web_search` can run automatically during normal chat when the message appears time-sensitive or external-knowledge dependent.
Control this in `tools.web_search`:
- `auto_use` (default `true`)
- `auto_count` (default `3`)

See [docs/OPERATIONS.md](docs/OPERATIONS.md) for runtime behavior and reliability details.

## Structured Memory and Heartbeat Notes
- Structured memory DB is created at `data/memory/blaire_memory.db` (or under `BLAIRE_DATA_PATH`).
- Heartbeat runs daily summarisation at most once every 24 hours and writes journal artifacts under `data/journal/`.
- Event retention pruning runs on heartbeat using `BLAIRE_EVENT_RETENTION_DAYS`.
- Deep-cut journal output is optional and controlled by `BLAIRE_DEEP_CUT_ENABLED=true`.
