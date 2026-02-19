# Admin Runbook

## Daily checks
- `GET /health`
- `GET /health/dependencies`
- `GET /ops/status` (single readiness snapshot)
- Review recent approvals and audit events

## Initialization and migrations
- `POST /ops/init` (admin)
- Re-runnable and safe for idempotent startup bootstrap

## Backup operations
- `POST /ops/backup`
- Output includes:
  - manifest
  - Postgres dump marker or SQL file
  - Qdrant metadata hook
  - app state hook (critical table counts)

## Sensitive operation governance
- Keep `SENSITIVE_ACTIONS_ENABLED=true` in normal mode.
- Temporarily set `false` for emergency freeze.
- Limit operations with allowlists:
  - network hosts/tools
  - Obsidian paths
  - HA operations
  - Homelab operations
- Prefer live runtime overrides for fast response:
  - `GET /runtime/config`
  - `PUT /runtime/config`
  - requires admin key
  - writes are audited with `updated_by` and `updated_at`

## Runtime policy override flow
1. Verify admin API key in Settings.
2. Open Settings or call `PUT /runtime/config`.
3. Set temporary overrides (for example disable sensitive actions or tighten allowlists).
4. Validate with `GET /runtime/options` and `GET /health/dependencies`.
5. Revert overrides back to `null` to return control to `.env`.

## Sandboxed execution
- Local sandbox: `POST /ops/sandbox/execute`
- CLI sandbox: `POST /ops/cli/execute` (requires `CLI_SANDBOX_ENABLED=true`)
- Keep `SANDBOX_ALLOWED_COMMANDS` minimal.

## Telegram webhook
- Configure:
  - `TELEGRAM_BOT_TOKEN`
  - `TELEGRAM_WEBHOOK_SECRET_TOKEN`
- Route:
  - `POST /telegram/webhook`
- Security:
  - secret token validation
  - rate limiting

## Voice integrations
- TTS: `POST /voice/tts` (requires Piper model configured)
- STT: `POST /voice/stt` (requires faster-whisper CLI available)

## Google + IMAP integrations
- Calendar read: `GET /integrations/google/calendar/events`
- Gmail send: `POST /integrations/google/gmail/send` (approval-required)
- IMAP read: `GET /integrations/imap/messages`

## Incident response
1. Disable sensitive actions.
2. Rotate API keys.
3. Review approval audit events.
4. Validate route access and proxy auth configuration.
