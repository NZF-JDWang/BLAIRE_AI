# Quickstart
Fastest path to get BLAIRE running locally.

For full setup detail and production-oriented notes, use `docs/installation_guide.md`.

## 1) Prerequisites
- Docker + Docker Compose
- Python 3.12 (for local backend tests)
- Node.js 22 + npm (for local frontend build)

Optional runtime binaries:
- `firejail` or `bubblewrap` for CLI sandbox route
- `piper` for local TTS route
- `faster-whisper` for local STT route

## 2) Configure environment
1. Copy `.env.example` to `.env`.
2. Fill required values:
   - `POSTGRES_PASSWORD`
   - `DATABASE_URL`
   - `ADMIN_API_KEYS`
   - `USER_API_KEYS`

Recommended security values:
- `TELEGRAM_WEBHOOK_SECRET_TOKEN`
- `ALLOWED_NETWORK_HOSTS`
- `ALLOWED_NETWORK_TOOLS`
- `ALLOWED_OBSIDIAN_PATHS`
- `ALLOWED_HA_OPERATIONS`
- `ALLOWED_HOMELAB_OPERATIONS`

Developer convenience:
- set `API_DOCS_ENABLED=true` during first setup if you want `/docs`
- use `FRONTEND_HOST_PORT` and `BACKEND_HOST_PORT` to avoid host port collisions

## 3) Start services
- Base stack:
  - `docker compose up -d`
- Dev stack (also exposes backend on host via proxy):
  - `docker compose --profile dev up -d`
- Optional services:
  - Search sidecar: `docker compose --profile search up -d`
  - MCP sidecars: `docker compose --profile mcp up -d`
  - vLLM backend: `docker compose --profile gpu up -d`
  - Watchtower ops: `docker compose --profile ops up -d watchtower`

Before chat requests, make sure `${LOCALAI_MODELS_PATH}` contains your LocalAI model configs.

## 4) Initialize metadata/vector state
- Call `POST /ops/init` with an admin API key.
- This initializes:
  - approval tables
  - preferences table
  - ingestion metadata table
  - Qdrant collection bootstrap

## 5) Verify
- Health: `GET /health`
- Dependencies: `GET /health/dependencies`
- Readiness: `GET /ops/status` (admin)
- Runtime options: `GET /runtime/options`
- Frontend: open `/` in browser

## 6) Run guided setup in UI
1. Open `/setup`.
2. Paste API key and click `Save and verify`.
3. Confirm detected access level (`user` or `admin`).
4. Review dependency + runtime diagnostics and remediation checklist.
5. Continue to `/settings` and `/chat`.

## 7) Smoke tests
- Backend tests: `cd backend && python -m pytest -q`
- Frontend build: `cd frontend && npm run build`

## Optional one-command bootstrap
- Linux/macOS: `bash ops/bootstrap.sh`
- Windows PowerShell: `./ops/bootstrap.ps1`
- Script actions:
  - create `.env` if missing
  - generate required secrets if empty
  - keep `DATABASE_URL` password in sync with `POSTGRES_PASSWORD` when placeholder/default is used
  - start compose stack
  - wait for backend readiness
  - run `POST /ops/init` automatically (with retry)
