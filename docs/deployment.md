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
- `dev`: `backend-dev` port proxy for host access to backend
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
   - for local development with backend exposed: `docker compose --profile dev up -d`
   - with accelerated inference: `docker compose --profile gpu up -d`
   - configure LocalAI model definitions under `${LOCALAI_MODELS_PATH}` before first chat request
   - for hybrid mode, define LocalAI models that proxy to `http://vllm:8000`
2. Run `POST /ops/init`
3. Validate routes and approvals flow
4. Upload test knowledge file and run ingestion

## Bootstrap helper
- `ops/bootstrap.sh` can perform first-run setup automatically:
  - creates `.env` if missing
  - generates missing secrets
  - starts stack
  - runs `/ops/init`

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

## Automated Compose deploy (GitHub Actions)
- Workflow: `.github/workflows/deploy-compose.yml`
- Script executed on target host: `ops/deploy-compose.sh`
- Trigger: push to `main` or manual `workflow_dispatch`
- Deploy behavior:
  1. fetch and reset target repo to `origin/main`
  2. `docker compose pull --ignore-pull-failures`
  3. `docker compose up -d --remove-orphans` (and `--build` when enabled)
  4. health-check endpoints (defaults: `/health`, `/health/dependencies`)
  5. rollback to previous git SHA + compose up if any step fails
- Required GitHub secrets:
  - `DEPLOY_HOST`
  - `DEPLOY_USER`
  - `DEPLOY_SSH_KEY`
  - `DEPLOY_PATH` (absolute path to repo on target host)
- Optional GitHub secrets:
  - `DEPLOY_PORT` (default SSH port `22`)
  - `DEPLOY_COMPOSE_FILE` (default `docker-compose.yml`)
  - `DEPLOY_COMPOSE_PROFILES` (comma-separated, e.g. `search,mcp,gpu`)
  - `DEPLOY_BUILD_IMAGES` (`true` or `false`, default `true`)
  - `DEPLOY_HEALTH_URL` (optional; if unset, health checks run inside backend container)
  - `DEPLOY_HEALTH_PATHS` (default `/health,/health/dependencies`)
  - `DEPLOY_HEALTH_RETRIES` (default `30`)
  - `DEPLOY_HEALTH_INTERVAL_SECONDS` (default `5`)
  - `DEPLOY_API_KEY` (used as `X-API-Key` for health requests)

## Rollback
1. Stop current stack.
2. Deploy previous image/tag.
3. Keep existing Postgres/Qdrant volumes.
4. Re-run `/ops/init` if schema bootstrap is required.
