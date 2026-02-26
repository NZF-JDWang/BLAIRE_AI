# Wave 1 Tools

Wave 1 introduces 12 tools:

1. `host_health_snapshot`
2. `docker_container_list`
3. `docker_container_logs`
4. `service_http_probe`
5. `docker_container_restart` (approval-gated write)
6. `obsidian_search`
7. `obsidian_get_note`
8. `qdrant_semantic_search`
9. `whisper_transcribe`
10. `chatterbox_tts_preview`
11. `media_pipeline_status` (Sonarr/Radarr/qBittorrent only)
12. `uptime_kuma_summary`

## Approval Flow

- First call to a gated mutating tool returns:
  - `error.code=approval_required`
  - `metadata.approval_token`
- Execute with CLI:
  - `/approvals list`
  - `/approve <token> <tool> <json_args>`

## Secrets Resolution

Secrets are resolved in this order:

1. Environment variable
2. `.secrets.local.json` in repository root
3. Fallback config value

`.secrets.local.json` is ignored by git.

