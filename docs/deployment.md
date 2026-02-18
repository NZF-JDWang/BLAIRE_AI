# Deployment Guide

## Compose topology
Default services:
- `backend`
- `frontend`
- `postgres`
- `qdrant`
- `localai`

Optional profiles:
- `search`: `searxng`
- `mcp`: `obsidian-mcp-server`, `ha-mcp-server`, `homelab-mcp`
- `gpu`: `vllm`
- `ops`: `watchtower`

All services share `containers_core` network.

## Bind mounts
- `${DROP_FOLDER}:/app/knowledge/drop`
- `${OBSIDIAN_VAULT_PATH}:/vault:rw`
- `${QDRANT_STORAGE_PATH}:/qdrant/storage`
- `${BACKUP_PATH}:/backups`
- `${POSTGRES_DATA_PATH}:/var/lib/postgresql/data`

## First deploy
1. `docker compose up -d`
   - with accelerated inference: `docker compose --profile gpu up -d`
   - configure LocalAI model definitions under `${LOCALAI_MODELS_PATH}` before first chat request
   - for hybrid mode, define LocalAI models that proxy to `http://vllm:8000`
2. Run `POST /ops/init`
3. Validate routes and approvals flow
4. Upload test knowledge file and run ingestion

## Updating
- Pull latest image/code and redeploy.
- For safer automatic updates, use `watchtower` in `ops` profile only.
- Keep rolling restart enabled and monitor startup dependency checks.

## Building Docker images via GitHub
- Workflow: `.github/workflows/docker-images.yml`
- Trigger: push to `main` (backend/frontend changes) or manual `workflow_dispatch`
- Publishes to GHCR:
  - `ghcr.io/<owner>/<repo>-backend:latest`
  - `ghcr.io/<owner>/<repo>-frontend:latest`

## Rollback
1. Stop current stack.
2. Deploy previous image/tag.
3. Keep existing Postgres/Qdrant volumes.
4. Re-run `/ops/init` if schema bootstrap is required.
