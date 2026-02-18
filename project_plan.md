# BLAIRE Project Plan (Backend-First, Task-Level)

## 1. Scope and Planning Mode
- Single consolidated plan across MVP (Phase 1), Phase 2, and Phase 3+.
- Task-level implementation plan for a single contributor.
- Backend-first execution order.
- Functional delivery focus (not timeline- or SLO-driven).
- Human-in-the-loop (HITL) control is required for any action that can affect other machines on the network.
- Model usage must be selectable at runtime (not hardcoded defaults).

## 2. Execution Strategy (Narrative Roadmap)
Build the platform in vertical slices, starting with backend foundations and secure tool orchestration, then add ingestion and agent logic, then integrate the frontend, and finally expand into Phase 2/3 capabilities. The core sequence is:
1. Stand up infra and backend scaffolding.
2. Implement RAG ingestion and retrieval.
3. Implement supervisor + research agent orchestration.
4. Add tool execution with mandatory approval flow for network-affecting actions.
5. Integrate frontend chat and settings controls.
6. Extend to advanced orchestration, homelab MCP, and voice/bot/PWA features.

## 3. Task Checklist (Ordered)

### A. Foundation and Repository Setup
- [ ] Create root structure aligned to design doc paths (`backend/`, `frontend/`, `data/`, `knowledge/`, MCP folders).
- [x] Add `docker-compose.yml` with services: `frontend`, `backend`, `postgres`, `qdrant`, optional `searxng`.
- [x] Add `.env.example` with all required variables (Ollama URL, Qdrant URL, Brave API key, vault/drop paths, MCP endpoints, auth secrets).
- [ ] Add backend Python dependency baseline (`FastAPI`, `LangGraph`, `LangChain`, `LlamaIndex`, `Qdrant client`, `structlog`, `Pydantic v2`).
- [ ] Add frontend dependency baseline (Next.js 16 App Router, Tailwind, shadcn/radix, TanStack Query, Zustand, RHF, Zod).
- [ ] Add container networks/volumes and bind mounts exactly as required.
- [ ] Pin MCP Python SDK to stable v1.x line for initial implementation.

### B. Backend Core (First Functional Vertical Slice)
- [x] Scaffold `backend/app/main.py` with health endpoint and startup wiring.
- [x] Implement config system (`app/config.py`) loading environment variables with typed validation.
- [x] Implement structured logging and request correlation IDs.
- [x] Implement chat API endpoint with token-streaming response contract.
- [x] Implement Ollama client wrapper (model routing hooks, streaming support).
- [x] Implement model registry + runtime model selection policy (supervisor/research/embedding/vision selectable via config/API).
- [x] Define model classes and preferences (`general`, `vision`, `embedding`, optional `code`) with admin-configurable defaults.
- [x] Implement router policy that selects within an allowlisted pool per model class.
- [x] Add per-session/user override support with policy checks and safe fallback chain.
- [x] Log router decisions (selected model, reason, fallback usage, latency/error outcome) for tuning/debugging.
- [x] Add backend integration test scaffold for health and chat streaming.

### C. RAG Engine and Knowledge Pipeline
- [x] Implement ingestion service skeleton under `backend/app/rag/`.
- [x] Configure Qdrant collection creation and embedding pipeline.
- [x] Implement drop-folder watcher for `knowledge/drop` with debounce and retry handling.
- [ ] Implement multimodal parsing pipeline for text/PDF/image ingestion via LlamaIndex components.
- [ ] Implement Obsidian vault indexer with initial full index + delta updates.
- [x] Implement citation metadata model (source path, chunk id, timestamp).
- [x] Expose retrieval API used by agent runtime.

### D. Agent Orchestration (LangGraph)
- [x] Define state schema for supervisor + worker agents.
- [x] Implement supervisor node (task decomposition, delegation policy).
- [x] Implement two parallel research worker nodes.
- [x] Implement merge/synthesis node with citation consolidation.
- [x] Add guardrails for max tool calls, recursion depth, and timeout ceilings.
- [x] Add per-step trace logging for swarm debugging and later UI visualization.

### E. Tool Registry and Search Providers
- [x] Implement tool registry abstraction (`backend/app/tools/`) with typed tool contracts.
- [x] Integrate Brave Search provider.
- [x] Integrate SearxNG adapter and add per-request provider selection mode:
- [x] Add modes: `Brave only`, `SearxNG only`, `Auto fallback`, `Parallel`.
- [x] Implement provider failover policy and normalized result schema.
- [x] Add API surface for frontend settings to control search mode.
- [x] Set system default search mode to `SearxNG only` with explicit user override.

### F. MCP Integration Layer
- [x] Implement MCP client wrapper with connection management and retries.
- [x] Integrate Obsidian MCP read/write actions with scoped path permissions.
- [x] Integrate Home Assistant MCP read/write actions with explicit allowlisted operations.
- [x] Implement unified tool-call envelope for MCP tools (request, result, audit metadata).

### G. Human-in-the-Loop Safety for Network Touch
- [x] Define action classes: `local_safe`, `local_sensitive`, `network_sensitive`.
- [x] Mark any machine/network-affecting operation as `network_sensitive`.
- [x] Build approval workflow service:
- [x] Create pending action record with full preview (target host, command, tool, expected effect).
- [x] Require explicit human approval before execution (approve/reject endpoint).
- [x] Issue short-lived approval token tied to exact action payload hash.
- [x] Enforce single-use token validation in execution path.
- [x] Build allowlist policy model (allowed hosts, allowed operation types, blocked operations).
- [x] Implement immutable audit log for all sensitive requests, approvals, and executions.
- [x] Add emergency kill switch to disable all sensitive tool execution globally.
- [x] Persist approvals and interrupt state durably in PostgreSQL (no in-memory-only approval state).

### H. Filesystem and CLI Sandboxing
- [x] Implement strict filesystem write tool with path allowlist and denylist.
- [x] Add preflight checks to prevent path traversal and symlink escapes.
- [x] Add sandbox runner abstraction for future CLI execution.
- [ ] Implement CLI sandbox in Phase 2 using firejail/bubblewrap with command allowlist.
- [x] Ensure all sandboxed calls emit machine-readable execution audit records.

### I. Frontend Implementation (After Backend APIs Stabilize)
- [x] Scaffold chat page with streaming response rendering.
- [x] Add file upload UI for PDFs/images and ingestion status display.
- [x] Add citations panel with source linking from backend metadata.
- [x] Add settings page controls for search mode dropdown and model choices.
- [x] Add approval queue UI for HITL actions (pending, approved, rejected, executed).
- [x] Add knowledge page showing index status (drop folder + Obsidian).
- [x] Add swarm status panel (basic in Phase 1, richer in Phase 2).

### J. Deployment and Operations
- [ ] Finalize compose services and environment wiring for homelab deployment path.
- [x] Add startup dependency checks (Qdrant, Ollama, MCP endpoints, search providers).
- [ ] Add migration/init routines for vector collections and PostgreSQL metadata stores.
- [ ] Add backup hooks for Qdrant data and critical app state.
- [ ] Add Watchtower update strategy with safe rollout notes.

### K. Phase 2 Extensions
- [ ] Implement live multi-agent swarm visualization API and frontend view.
- [ ] Build custom homelab MCP server (Docker/Portainer/backup/media safe operations).
- [ ] Route homelab MCP operations through same HITL approval system.
- [ ] Implement production-ready CLI sandbox allowlist workflows.

### L. Phase 3+ Extensions
- [ ] Add Google Calendar and Gmail/IMAP tool integrations (with HITL for sensitive actions).
- [ ] Add local TTS (Piper) integration.
- [ ] Add local STT (faster-whisper) integration.
- [ ] Add Telegram bot channel with shared orchestration backend.
- [ ] Add responsive PWA shell around existing frontend routes.

## 4. Dependency Order (Critical Path)
1. Foundation setup (A)
2. Backend core (B)
3. RAG pipeline (C)
4. Agent orchestration (D)
5. Tool/search layer (E)
6. MCP integration (F)
7. HITL safety system (G)
8. Filesystem sandboxing (H local FS portion)
9. Frontend integration (I)
10. Deployment hardening (J)
11. Phase 2 and Phase 3+ expansions (K, L)

## 5. Open Decisions to Resolve Before Implementation Starts
- Approval UX authority: who/what identity approves HITL actions (local admin UI only, SSO user, or API token owner)?
- AuthN/AuthZ scope for frontend/backend: required from day 1 or temporary trusted-network mode during initial build.
- SearxNG deployment mode: bundled in compose vs external shared instance.
- Obsidian vault mount mode at deployment: always read/write vs environment-toggle read-only/read-write.
- Model selection governance: who can change active models at runtime and whether per-session overrides are allowed.
- Router strategy detail: deterministic rules-only, heuristic scoring, or hybrid (rules + score) for model picks.

## 6. Suggested Build Sequence (First 10 Execution Tasks)
1. Create `docker-compose.yml` and `.env.example`.
2. Scaffold backend app with health + config + logging.
3. Add Ollama connectivity and streaming chat endpoint.
4. Add Qdrant connectivity and embedding pipeline skeleton.
5. Implement drop-folder ingestion watcher.
6. Implement Obsidian full + delta indexing.
7. Build LangGraph supervisor + two workers.
8. Implement search providers and dropdown mode wiring.
9. Implement HITL approval service for network-sensitive actions.
10. Add frontend chat/settings/approval queue pages.
