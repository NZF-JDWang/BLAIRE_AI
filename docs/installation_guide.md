# BLAIRE Installation Guide

This guide is the full install path for a first-time setup.

If you want a shorter version, see `docs/quickstart.md`.

## 1) What You Are Installing

BLAIRE runs as a multi-container stack:
- `backend` (FastAPI)
- `frontend` (Next.js)
- `postgres`
- `qdrant`
- `localai`

Optional profiles:
- `search` -> `searxng`
- `mcp` -> `obsidian-mcp-server`, `ha-mcp-server`, `homelab-mcp`
- `gpu` -> `vllm`
- `ops` -> `watchtower`

## 2) Prerequisites

Required:
- Docker Engine + Docker Compose plugin
- Git

Recommended for local validation:
- Python 3.12
- Node.js 22 + npm

Check tools:

```bash
docker --version
docker compose version
git --version
```

## 3) Clone the Repository

```bash
git clone https://github.com/NZF-JDWang/BLAIRE_AI.git
cd BLAIRE_AI
```

## 4) Create Your Environment File

Copy the example file:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

## 5) Fill Required `.env` Values

At minimum, set:
- `POSTGRES_PASSWORD`
- `DATABASE_URL`
- `ADMIN_API_KEYS`
- `USER_API_KEYS`

Important:
- `DATABASE_URL` password must match `POSTGRES_PASSWORD`.
- Current inference default is `INFERENCE_BASE_URL=http://localai:8080`.

Useful hardening values before production:
- `TELEGRAM_WEBHOOK_SECRET_TOKEN`
- `ALLOWED_NETWORK_HOSTS`
- `ALLOWED_NETWORK_TOOLS`
- `ALLOWED_OBSIDIAN_PATHS`
- `ALLOWED_HA_OPERATIONS`
- `ALLOWED_HOMELAB_OPERATIONS`

Helpful install toggles:
- `API_DOCS_ENABLED=true` (temporary for setup/discovery)
- `FRONTEND_HOST_PORT` / `BACKEND_HOST_PORT` (avoid host collisions)
- `ENABLE_MCP_SERVICES=true` when using `--profile mcp`
- `ENABLE_VLLM=true` when using `--profile gpu`

`API_ALLOWED_HOSTS` common setups:
- Local-only: `localhost,127.0.0.1,backend`
- Reverse proxy: `your-domain.com,backend`
- Hybrid (proxy + local test): `your-domain.com,localhost,127.0.0.1,backend`

## 6) Start the Stack

Base stack:

```bash
docker compose up -d
```

Dev stack (publishes backend through `backend-dev` proxy):

```bash
docker compose --profile dev up -d
```

With optional services:

```bash
docker compose --profile search --profile mcp --profile gpu up -d
```

Notes:
- `gpu` profile requires a host/GPU runtime that supports your `vllm` container.
- Keep `mcp` disabled if you have not configured MCP-related env values yet.

## 7) Host Access Defaults

- Frontend is published by default on `FRONTEND_HOST_PORT` (default `3000`).
- Backend is internal by default for safety.
- Backend is published only when you run `--profile dev` on `BACKEND_HOST_PORT` (default `8001`).

## 8) Initialize BLAIRE Metadata

Run one-time init after startup.

If backend is exposed (for example with `--profile dev`):

```bash
curl -X POST http://localhost:${BACKEND_HOST_PORT:-8001}/ops/init -H "X-API-Key: <ADMIN_API_KEY>"
```

If backend is not exposed, run from inside backend container:

```bash
docker compose exec backend sh -lc 'curl -X POST http://localhost:8000/ops/init -H "X-API-Key: <ADMIN_API_KEY>"'
```

## 9) Verify Health

If backend is exposed (for example with `--profile dev`):

```bash
curl http://localhost:${BACKEND_HOST_PORT:-8001}/health
curl http://localhost:${BACKEND_HOST_PORT:-8001}/health/dependencies -H "X-API-Key: <USER_API_KEY>"
curl http://localhost:${BACKEND_HOST_PORT:-8001}/runtime/options -H "X-API-Key: <USER_API_KEY>"
```

If not exposed:

```bash
docker compose exec backend sh -lc 'curl -s http://localhost:8000/health'
docker compose exec backend sh -lc 'curl -s http://localhost:8000/health/dependencies -H "X-API-Key: <USER_API_KEY>"'
docker compose exec backend sh -lc 'curl -s http://localhost:8000/runtime/options -H "X-API-Key: <USER_API_KEY>"'
```

## 10) First-Use Checklist

1. Open frontend (`http://localhost:${FRONTEND_HOST_PORT:-3000}`).
2. Go to Settings and set your user API key.
3. Send a test chat message.
4. Upload one knowledge file in Knowledge page.
5. Trigger one approval-required action and verify approval flow.

## 11) Logs and Basic Debugging

See running services:

```bash
docker compose ps
```

Follow logs:

```bash
docker compose logs -f backend
docker compose logs -f frontend
```

Restart service:

```bash
docker compose restart backend
```

## 12) Update Procedure

```bash
git pull
docker compose pull
docker compose up -d --remove-orphans
```

If local Dockerfiles changed and you build locally:

```bash
docker compose up -d --build --remove-orphans
```

## Optional one-command bootstrap

```bash
bash ops/bootstrap.sh
```

This script:
- creates `.env` from `.env.example` if missing
- generates required secrets when empty
- warns if configured host ports are already in use
- starts compose
- runs `/ops/init` automatically

## 13) Optional Automated Deployment

For SSH-based automated deploys:
- workflow: `.github/workflows/deploy-compose.yml`
- script: `ops/deploy-compose.sh`
- setup details: `docs/deployment.md`

## 14) Known Common Mistakes

1. `DATABASE_URL` password does not match `POSTGRES_PASSWORD`.
2. Forgetting `POST /ops/init` after first startup.
3. Missing API key in frontend settings.
4. Starting `mcp` profile without configuring related env vars.
5. Assuming a specific inference provider is required. Current stack uses LocalAI (`INFERENCE_BASE_URL`) and optional vLLM.
