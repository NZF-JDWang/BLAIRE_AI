# Architecture

## Overview
BLAIRE is a backend-first system with a Next.js frontend proxying to FastAPI.

Core subsystems:
- API layer (FastAPI routes)
- Agent orchestration (`AgentSwarmService`)
- RAG ingestion/retrieval (Qdrant + embeddings)
- Tool/MCP execution with HITL approvals
- Ops/sandbox services
- Frontend app pages for chat/settings/knowledge/swarm/approvals

## Data stores
- PostgreSQL:
  - approvals
  - approval audit events
  - user preferences
  - ingestion file metadata
- Qdrant:
  - multimodal chunk vectors and citation payload

## Request flow (chat)
1. Frontend sends chat request to `/api/chat`.
2. Proxy forwards to backend with explicit API key.
3. Backend chooses model via router policy and preferences.
4. Optional retrieval inserts context with citations.
5. SSE tokens stream back to frontend.

## Request flow (sensitive action)
1. Client calls tool/MCP route.
2. Route classifies action as sensitive/network-sensitive.
3. Pending approval record is created.
4. Admin approves to receive execution token.
5. Client executes with token + expected payload hash.

## Deployment model
- Docker Compose services on shared internal network.
- Optional sidecars via profiles (`search`, `mcp`, `ops`).
- Watchtower optional for rolling updates.
