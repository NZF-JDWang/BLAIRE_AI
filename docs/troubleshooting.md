# Troubleshooting

## Backend unreachable
- Check `docker compose ps`
- Check backend logs
- Validate `API_ALLOWED_HOSTS`
- If testing from host, run with `--profile dev` or publish backend port via override

## Invalid host header / TrustedHost errors
- Set `API_ALLOWED_HOSTS` to include the hostname you use.
- Common examples:
  - local: `localhost,127.0.0.1,backend`
  - reverse proxy: `your-domain.com,backend`
  - both: `your-domain.com,localhost,127.0.0.1,backend`

## Auth failures (`401`/`403`)
- Ensure `X-API-Key` is set
- Confirm key belongs to correct role (user/admin)
- In frontend, set API key on `/settings`

## Startup init failures
- Call `POST /ops/init` after stack startup
- Verify Postgres reachable and `DATABASE_URL` valid
- Verify Qdrant reachable and `QDRANT_URL` valid
- Check `GET /ops/status` (admin key) for one-shot readiness state

## Knowledge ingestion issues
- Confirm `DROP_FOLDER` path mounted and readable
- Verify file extension support
- Check inference embedding model availability (`GET /v1/models` on `INFERENCE_BASE_URL`)

## MCP call failures
- Verify MCP sidecars are running (`--profile mcp`)
- Check endpoint env vars (`MCP_*_URL`)
- Check allowlist config and approval state
- Set `ENABLE_MCP_SERVICES=true` when MCP profile is enabled

## CLI sandbox failures
- Ensure `CLI_SANDBOX_ENABLED=true`
- Ensure backend has required binary:
  - `firejail` or `bubblewrap`
- Ensure command is in `SANDBOX_ALLOWED_COMMANDS`

## Voice route failures
- TTS:
  - `PIPER_BIN` present
  - `PIPER_VOICE_MODEL` configured
- STT:
  - `FASTER_WHISPER_BIN` present

## Telegram webhook ignored/rejected
- Verify `TELEGRAM_BOT_TOKEN`
- Verify `X-Telegram-Bot-Api-Secret-Token` matches `TELEGRAM_WEBHOOK_SECRET_TOKEN`
- Check webhook rate limit bursts

## Integrations errors
- Google:
  - valid `GOOGLE_OAUTH_TOKEN`
- IMAP:
  - valid `IMAP_HOST`, `IMAP_USER`, `IMAP_PASSWORD`

## Dependency endpoint shows optional failures
- `GET /health/dependencies` now reports `required` and `enabled`
- Optional dependencies are marked `enabled=false` when not configured
- Common toggles:
  - `ENABLE_MCP_SERVICES` for `--profile mcp`
  - `ENABLE_VLLM` for `--profile gpu`
