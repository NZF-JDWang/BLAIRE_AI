# Configuration Reference

BLAIRE reads settings from `.env` (see `.env.example`).

## Config layers
- System config (`.env`): loaded at process start.
- Runtime overrides (DB-backed): editable via `GET/PUT /runtime/config` (admin), applied live for selected policy fields.
- User preferences (DB-backed): editable via `GET/PUT /preferences/me`, applied per user.

Runtime overrides currently cover:
- `SEARCH_MODE_DEFAULT`
- `SENSITIVE_ACTIONS_ENABLED`
- `APPROVAL_TOKEN_TTL_MINUTES`
- `ALLOWED_NETWORK_HOSTS`
- `ALLOWED_NETWORK_TOOLS`
- `ALLOWED_OBSIDIAN_PATHS`
- `ALLOWED_HA_OPERATIONS`
- `ALLOWED_HOMELAB_OPERATIONS`

## Core
- `APP_ENV`, `LOG_LEVEL`
- `API_HOST`, `API_PORT`, `API_ALLOWED_HOSTS`, `API_DOCS_ENABLED`
- `REQUIRE_AUTH`, `ADMIN_API_KEYS`, `USER_API_KEYS`
- `FRONTEND_HOST_PORT`, `BACKEND_HOST_PORT`

`API_ALLOWED_HOSTS` common patterns:
- Local testing: `localhost,127.0.0.1,backend`
- Reverse proxy + local: `your-domain.com,localhost,127.0.0.1,backend`
- Internal Docker-only backend: include `backend`

## Data and services
- `DATABASE_URL`
- `QDRANT_URL`, `QDRANT_COLLECTION_NAME`, `QDRANT_EMBEDDING_DIM`
- `INFERENCE_BASE_URL` (LocalAI OpenAI-compatible endpoint)
- `VLLM_BASE_URL` (used for dependency checks and direct vLLM ops)
- `VLLM_MODEL` (default `Qwen/Qwen3-VL-14B` in `.env.example`)
- `VLLM_QUANTIZATION` (default `nvfp4` in `.env.example`)
- `SEARCH_MODE_DEFAULT`, `BRAVE_API_KEY`, `SEARXNG_URL`
- `ENABLE_MCP_SERVICES` (`true` when running `--profile mcp`)
- `ENABLE_VLLM` (`true` when running `--profile gpu`)

## Knowledge and storage
- `DROP_FOLDER`
- `OBSIDIAN_VAULT_PATH`
- `BACKUP_PATH`
- `MAX_UPLOAD_MB`

Portable defaults in `.env.example` use `./data/...` paths so installs work on most hosts without `/srv` bind mounts.

## Model routing
- `MODEL_GENERAL_DEFAULT`
- `MODEL_VISION_DEFAULT`
- `MODEL_EMBEDDING_DEFAULT`
- `MODEL_CODE_DEFAULT`
- `MODEL_ALLOW_ANY_INFERENCE`
- `MODEL_ALLOWLIST_EXTRA_GENERAL`
- `MODEL_ALLOWLIST_EXTRA_VISION`
- `MODEL_ALLOWLIST_EXTRA_EMBEDDING`
- `MODEL_ALLOWLIST_EXTRA_CODE`
- `MODEL_DISALLOWLIST`

## Safety and policy
- `SENSITIVE_ACTIONS_ENABLED`
- `APPROVAL_TOKEN_TTL_MINUTES`
- `ALLOWED_NETWORK_HOSTS`
- `ALLOWED_NETWORK_TOOLS`
- `ALLOWED_WRITE_PATHS`
- `ALLOWED_OBSIDIAN_PATHS`
- `ALLOWED_HA_OPERATIONS`
- `ALLOWED_HOMELAB_OPERATIONS`

## Agent guardrails
- `AGENT_MAX_TOOL_CALLS`
- `AGENT_MAX_RECURSION_DEPTH`
- `AGENT_WORKER_TIMEOUT_SECONDS`
- `AGENT_OVERALL_TIMEOUT_SECONDS`

## Sandbox
- `SANDBOX_ALLOWED_COMMANDS`
- `CLI_SANDBOX_ENABLED`
- `CLI_SANDBOX_BACKEND` (`firejail` or `bubblewrap`)

## Voice
- `PIPER_BIN`, `PIPER_VOICE_MODEL`
- `FASTER_WHISPER_BIN`, `FASTER_WHISPER_MODEL`

## Telegram
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_WEBHOOK_SECRET_TOKEN`
- `TELEGRAM_DEFAULT_CHAT_ID`

## Integrations
- `GOOGLE_API_BASE`
- `GOOGLE_OAUTH_TOKEN`
- `IMAP_HOST`, `IMAP_USER`, `IMAP_PASSWORD`
- `HOME_ASSISTANT_URL`, `HOME_ASSISTANT_TOKEN`, `HOME_ASSISTANT_VERIFY_TLS`
- `HOMELAB_ALLOWED_HTTP_HOSTS` (comma-separated allowlist for `http.check` operation)

## Frontend proxy
- `INTERNAL_API_BASE_URL`
- `NEXT_PUBLIC_API_BASE_URL`
- `FRONTEND_PROXY_API_KEY` (server-side fallback only; browser should use user API key in settings)

## User preference controls
`PUT /preferences/me` also supports:
- `temperature` (`0.0` - `2.0`)
- `top_p` (`0.0` - `1.0`)
- `max_tokens` (`null` or `1` - `8192`)
- `context_window_tokens` (`null` or `256` - `262144`)
- `use_rag` (`true` / `false`)
- `retrieval_k` (`1` - `12`)
