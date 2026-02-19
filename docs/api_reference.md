# API Reference

All routes are served by backend FastAPI.
Auth: `X-API-Key` when `REQUIRE_AUTH=true`.

## Health
- `GET /health`
- `GET /health/dependencies`

## Chat and runtime
- `POST /chat`
- `GET /runtime/options`
- `GET /runtime/diagnostics`
- `GET /runtime/system-summary` (admin)
- `GET /runtime/config` (admin)
- `PUT /runtime/config` (admin)
- `GET /runtime/config/audit` (admin)

## Search
- `POST /search`

## Preferences
- `GET /preferences/me`
- `PUT /preferences/me`
- Preference payload includes model/search defaults plus generation controls:
  - `temperature`, `top_p`, `max_tokens`, `context_window_tokens`
  - `use_rag`, `retrieval_k`

## Agents
- `POST /agents/research`
- `GET /agents/swarm/live`

## Knowledge
- `GET /knowledge/status`
- `POST /knowledge/ingest`
- `POST /knowledge/retrieve`
- `POST /knowledge/upload`
- `POST /knowledge/obsidian/reindex`

## Tools
- `GET /tools`
- `POST /tools/execute`

## Approvals
- `GET /approvals/pending`
- `GET /approvals/recent`
- `POST /approvals`
- `GET /approvals/{approval_id}`
- `GET /approvals/{approval_id}/audit`
- `POST /approvals/{approval_id}/approve`
- `POST /approvals/{approval_id}/reject`
- `POST /approvals/{approval_id}/execute`

## MCP
- `POST /mcp/obsidian/read`
- `POST /mcp/obsidian/write`
- `POST /mcp/ha/call`
- `POST /mcp/homelab/call`

## Ops
- `POST /ops/init`
- `GET /ops/status`
- `POST /ops/backup`
- `POST /ops/sandbox/execute`
- `POST /ops/cli/execute`

## Voice
- `POST /voice/tts`
- `POST /voice/stt`

## Integrations
- `GET /integrations/google/calendar/events`
- `POST /integrations/google/gmail/send`
- `GET /integrations/imap/messages`

## Telegram
- `POST /telegram/webhook`
