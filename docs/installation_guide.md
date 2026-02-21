# BLAIRE Installation and Setup Guide

This is the single, complete first-time setup guide.

## 1) Choose your install path

### Recommended: automated install (Linux/macOS)
```bash
bash ops/bootstrap.sh
```

### Recommended: automated install (Windows PowerShell)
```powershell
./ops/bootstrap.ps1
```

Use automated install if possible. It handles most setup steps for you.

### Manual install
If you prefer control over each step, continue below.

---

## 2) Prerequisites

Required:
- Docker Engine + Docker Compose plugin
- Git

Optional (for local validation):
- Python 3.12
- Node.js 22 + npm

Quick checks:

```bash
docker --version
docker compose version
git --version
```

---

## 3) Clone and prepare environment

```bash
git clone https://github.com/NZF-JDWang/BLAIRE_AI.git
cd BLAIRE_AI
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set these required values:
- `POSTGRES_PASSWORD`
- `DATABASE_URL`
- `ADMIN_API_KEYS`
- `USER_API_KEYS`

Important:
- `DATABASE_URL` password must match `POSTGRES_PASSWORD`.

Helpful first-time toggles:
- `API_DOCS_ENABLED=true` (enable `/docs` during setup)
- `FRONTEND_HOST_PORT` / `BACKEND_HOST_PORT` (avoid local port conflicts)

---

## 4) Start services

Base stack:

```bash
docker compose up -d
```

Optional profiles:

```bash
docker compose --profile dev up -d
docker compose --profile search up -d
docker compose --profile mcp up -d
docker compose --profile gpu up -d
```

Use only profiles you need:
- `dev`: publish backend to host
- `search`: run local SearxNG sidecar
- `mcp`: run MCP sidecars
- `gpu`: run vLLM

---

## 5) Initialize BLAIRE (required)

After containers are up, run one-time init:

```bash
curl -X POST http://localhost:8001/ops/init -H "X-API-Key: <ADMIN_API_KEY>"
```

If backend is not exposed with `--profile dev`, run from inside backend container:

```bash
docker compose exec backend sh -lc 'curl -s -X POST http://localhost:8000/ops/init -H "X-API-Key: <ADMIN_API_KEY>"'
```

---

## 6) Verify installation

Backend checks:

```bash
curl -s http://localhost:8001/health
curl -s http://localhost:8001/health/dependencies -H "X-API-Key: <USER_API_KEY>"
curl -s http://localhost:8001/runtime/options -H "X-API-Key: <USER_API_KEY>"
```

If backend is internal-only:

```bash
docker compose exec backend sh -lc 'curl -s http://localhost:8000/health'
docker compose exec backend sh -lc 'curl -s http://localhost:8000/health/dependencies -H "X-API-Key: <USER_API_KEY>"'
docker compose exec backend sh -lc 'curl -s http://localhost:8000/runtime/options -H "X-API-Key: <USER_API_KEY>"'
```

Frontend:
- open `http://localhost:${FRONTEND_HOST_PORT:-3000}`

---

## 7) Complete in-app setup

1. Open `/setup`.
2. Paste API key and click **Save and verify**.
3. Confirm access level (`user` or `admin`).
4. Review diagnostics and apply suggested fixes.
5. Continue to `/settings` and `/chat`.

Recommended first-use checks:
1. Send a test chat message.
2. Upload one file in Knowledge.
3. Trigger one approval-required action and approve it.

---

## 8) Troubleshooting and maintenance

Service status/logs:

```bash
docker compose ps
docker compose logs -f backend
docker compose logs -f frontend
```

Restart backend:

```bash
docker compose restart backend
```

Update stack:

```bash
git pull
docker compose pull
docker compose up -d --remove-orphans
```

If local Dockerfiles changed:

```bash
docker compose up -d --build --remove-orphans
```

For known issues and fixes, see `docs/troubleshooting.md`.

---

## 9) What automated bootstrap does

`ops/bootstrap.sh` and `ops/bootstrap.ps1` automate installation and setup by:
- running preflight checks
- creating `.env` when missing
- generating required secrets when empty
- syncing `DATABASE_URL` password with `POSTGRES_PASSWORD` when placeholders are detected
- auto-selecting compose profiles from env
- starting the stack and waiting for backend readiness
- running `POST /ops/init` automatically
- running smoke checks automatically

Use standalone scripts when needed:
- Linux/macOS: `bash ops/preflight.sh`, `bash ops/smoke-test.sh`
- Windows PowerShell: `./ops/preflight.ps1`, `./ops/smoke-test.ps1`
