# BLAIRE

Blacksite Lab AI Hub: a custom, self-hosted agent platform with FastAPI backend, Next.js frontend, RAG, HITL approvals, and MCP integrations.

## Stack
- Backend: FastAPI, Pydantic v2, structlog, LangGraph/LangChain/LlamaIndex, Qdrant, PostgreSQL
- Frontend: Next.js 16 App Router, React 19
- Inference: LocalAI (OpenAI-compatible) with optional vLLM backend profile
- Orchestration: multi-agent swarm + live trace API
- Safety: approval workflow, action classes, rate limits, filesystem + CLI sandboxing

## Quick Start
1. Copy `.env.example` to `.env` and fill required secrets.
2. Start services:
   - `docker compose up -d`
   - optional profiles: `search`, `mcp`, `ops`
3. Backend health: `GET /health`
4. Run backend tests: `cd backend && python -m pytest -q`

## Key Routes
- Chat: `POST /chat` (SSE streaming)
- Swarm: `POST /agents/research`, `GET /agents/swarm/live`
- Knowledge: `/knowledge/*` (status, ingest, retrieve, upload, obsidian reindex)
- Approvals: `/approvals/*`
- MCP: `/mcp/*` (obsidian, HA, homelab)
- Ops: `/ops/init`, `/ops/backup`, `/ops/sandbox/execute`, `/ops/cli/execute`
- Voice: `/voice/tts`, `/voice/stt`
- Integrations: Google Calendar/Gmail and IMAP routes
- Telegram: `POST /telegram/webhook`

## Deployment Notes
See `deployment_notes.md` for profile usage, bind mounts, and Watchtower rollout guidance.

## Full Documentation
See `docs/README.md` for complete setup instructions, user/admin manual, API reference, security model, and troubleshooting guides.
