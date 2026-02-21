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
Bootstrap auto-selects compose profiles from `.env` values. Typical default behavior is enabling `search` and `mcp` when their internal URLs are configured.

### Manual install
If you prefer control over each step, continue below.

---

## 2) Prerequisites

Required:
- Docker Engine + Docker Compose plugin
- Git
- Python 3 (`python3` on Linux, or `python` alias via `python-is-python3` on Ubuntu)

Optional (for local validation):
- Python 3.12
- Node.js 22 + npm

Recommended host resources (for LocalAI + SearxNG + app stack on one machine):
- CPU: 4+ cores
- RAM: 16 GB+ (8 GB minimum for lighter models)

Quick checks:

```bash
docker --version
docker compose version
git --version
python3 --version
```

If `python` is missing on Ubuntu:

```bash
sudo apt install -y python-is-python3
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

No manual edits are required for first run if you use bootstrap. It will auto-fill placeholders and required defaults.

Manual `.env` edits are only needed when you want custom values (for example custom ports, external endpoints, or pre-defined API keys).

Helpful first-time toggles:
- `API_DOCS_ENABLED=true` (enable `/docs` during setup)
- `FRONTEND_HOST_PORT` / `BACKEND_HOST_PORT` (avoid local port conflicts)
- `ENABLE_SEARCH=false` (force search profile off during bootstrap)

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

Default host ports:

| Service | Host port |
| --- | --- |
| Frontend | `3000` |
| Backend | `8001` |
| LocalAI | `8082` |
| SearxNG | `8080` |

---

## 5) Initialize BLAIRE (required)

If backend is exposed with `--profile dev`, run:

```bash
curl -X POST http://localhost:8001/ops/init -H "X-API-Key: <ADMIN_API_KEY>"
```

If backend is internal-only (default `docker compose up -d`), run from inside backend container:

```bash
docker compose exec backend sh -lc 'curl -s -X POST http://localhost:8000/ops/init -H "X-API-Key: <ADMIN_API_KEY>"'
```

---

## 6) Verify installation

Backend checks (if using `--profile dev`):

```bash
curl -s http://localhost:8001/health
curl -s http://localhost:8001/health/dependencies -H "X-API-Key: <USER_API_KEY>"
curl -s http://localhost:8001/runtime/options -H "X-API-Key: <USER_API_KEY>"
```

Backend checks (internal-only default):

```bash
docker compose exec backend sh -lc 'curl -s http://localhost:8000/health'
docker compose exec backend sh -lc 'curl -s http://localhost:8000/health/dependencies -H "X-API-Key: <USER_API_KEY>"'
docker compose exec backend sh -lc 'curl -s http://localhost:8000/runtime/options -H "X-API-Key: <USER_API_KEY>"'
```

Frontend:
- open `http://localhost:${FRONTEND_HOST_PORT:-3000}`

---

## 7) Complete in-app setup

1. Open `/settings`.
2. Paste API key and click **Connection test**.
3. Confirm access level (`user` or `admin`).
4. Review diagnostics and apply suggested fixes.
5. Continue to `/chat`.

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
- generating required secrets when empty or placeholder values are detected
- syncing `DATABASE_URL` password with `POSTGRES_PASSWORD` when placeholders are detected
- auto-selecting compose profiles from env
- starting the stack and waiting for backend readiness
- running `POST /ops/init` automatically
- running smoke checks automatically

Use standalone scripts when needed:
- Linux/macOS: `bash ops/preflight.sh`, `bash ops/smoke-test.sh`
- Windows PowerShell: `./ops/preflight.ps1`, `./ops/smoke-test.ps1`
