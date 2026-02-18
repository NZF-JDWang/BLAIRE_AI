# BLAIRE Deployment Notes

## Compose profiles
- Default: `backend`, `frontend`, `postgres`, `qdrant`
- Search sidecar: `docker compose --profile search up -d`
- MCP sidecars: `docker compose --profile mcp up -d`
- Watchtower ops automation: `docker compose --profile ops up -d watchtower`

## Required bind mounts
- `${DROP_FOLDER:-/srv/ai-knowledge/drop}:/app/knowledge/drop`
- `${OBSIDIAN_VAULT_PATH:-/srv/obsidian/BlacksiteLabVault}:/vault:rw`
- `${QDRANT_STORAGE_PATH:-/srv/blacksitelab/blaire/qdrant}:/qdrant/storage`
- `${BACKUP_PATH:-/mnt/backups/blaire}:/backups`

## Safe Watchtower rollout strategy
1. Keep Watchtower in `ops` profile only; do not enable by default.
2. Limit scope to `blaire-backend` and `blaire-frontend` containers only.
3. Use `--rolling-restart` to avoid simultaneous backend/frontend restarts.
4. Keep interval at 24h and monitor startup checks after each rollout.
5. Use `/ops/init` post-deploy if schema/vector bootstrap is required.
