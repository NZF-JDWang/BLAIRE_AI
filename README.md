# BLAIRE

Blacksite Lab AI Hub: a self-hosted agent platform with FastAPI backend, Next.js frontend, RAG, HITL approvals, and MCP integrations.

## The fastest way to install and set up

### Option A (recommended): fully automated bootstrap

```bash
bash ops/bootstrap.sh
```

Windows PowerShell:

```powershell
./ops/bootstrap.ps1
```

The bootstrap flow does all first-time setup for you:
- checks required tools and `.env` completeness
- creates `.env` from `.env.example` if missing
- auto-replaces placeholder secrets and fills required defaults
- auto-selects compatible Docker compose profiles
- starts containers
- waits for backend readiness
- runs first-time `POST /ops/init`
- runs smoke tests and reports next steps

### Option B: manual install
Follow the single setup doc: `docs/installation_guide.md`.

## What to do after install
1. Open the frontend (`http://localhost:3000` by default).
2. Go to `/settings`.
3. Save your API key and verify access.
4. Review dependency/runtime checks.
5. Continue to `/chat`.

## Full docs map
See `docs/README.md` for install, setup, configuration, operations, and troubleshooting docs.
