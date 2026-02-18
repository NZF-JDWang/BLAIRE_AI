# Security Model

## Authentication
- Backend API-key based auth with role split:
  - admin
  - user
- Frontend proxy requires explicit user key:
  - `X-API-Key` header or `blaire_api_key` cookie

## Approval/HITL model
- Sensitive actions classified as:
  - `local_safe`
  - `local_sensitive`
  - `network_sensitive`
- `network_sensitive` actions require approval token flow:
  1. create pending approval
  2. admin approve/reject
  3. execute with one-time token + payload hash

## Policy enforcement
- Host/tool/path/operation allowlists:
  - network
  - Obsidian path scope
  - Home Assistant operations
  - Homelab operations
- Global kill switch:
  - `SENSITIVE_ACTIONS_ENABLED=false`

## Auditing
- Immutable approval audit events with actor and details
- Machine-readable sandbox execution records

## Sandboxing
- Filesystem write sandbox with traversal/symlink defenses
- CLI sandbox with allowlist and backend isolation (`firejail`/`bubblewrap`)
- CLI command approval modes: `allow_once` and per-user `allow_always` for approved commands
- Optional per-user unrestricted CLI mode requires explicit dangerous-mode acknowledgement and confirmation text

## Webhook protection
- Telegram webhook supports secret token verification
- Telegram webhook endpoint rate-limited
