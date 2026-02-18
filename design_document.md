# Blacksite Lab AI Hub (BLAIRE) — Design Document v0.2

**Purpose**  
Canonical, copy-paste-ready specification for a **pure custom**, open-source, agentic AI Hub that fully replaces OpenWebUI.  
Feed entire sections (or the whole document) directly into VS Code / Cursor / Claude / etc. for vibe-coding.

**Version:** 0.2 (Post-clarification round)  
**Date:** 2026-02-18  
**Target hardware:** BSL1 (RTX 5060 Ti 16 GB VRAM, 32 GB RAM)  
**Constraints enforced:**  
- 100% open-source & free software  
- Docker-first deployment  
- No exposed ports (Traefik + Cloudflare Zero Trust only)  
- Multimodal RAG from day 1  
- Full write access tools → **with mandatory heavy sandboxing**  
- Pure custom implementation (no AnythingLLM / LibreChat / similar bases)

## 1. High-Level Overview & Vision

BLAIRE = private, agentic “second brain + execution swarm” deeply integrated with the Blacksite Lab homelab.

**Core Pillars**
- Pure custom FastAPI backend + LangGraph orchestration
- Next.js 15 App Router frontend (polished chat + multi-agent swarm dashboard)
- Hybrid multimodal RAG (live Obsidian vault + configurable drop folder)
- Hierarchical multi-agent system (supervisor + parallel research agents)
- MCP-native integrations (Obsidian, Home Assistant, future custom Homelab MCP)
- Brave Search (primary) + SearxNG (configurable secondary/parallel/fallback)
- Sandboxed full-write tools (FS, CLI, Docker control, n8n, later Google services)
- Future extensions: local TTS/STT + Telegram bot + PWA

## 2. Phased Functional Requirements

**Phase 1 – MVP** (aim: 2–4 weeks intensive vibe coding)
- Next.js chat UI: streaming responses, citations, image/PDF upload support
- Configurable drop folder → automatic multimodal ingestion & indexing
- Live + delta indexing of Obsidian vault
- Brave Search primary + SearxNG fallback/parallel
- Supervisor agent + 2 parallel research agents
- Basic read/write via Obsidian MCP and HA MCP
- Sandboxed filesystem write tool (strict path allowlist)

**Phase 2**
- Live multi-agent swarm visualisation in UI
- Custom Homelab MCP server (safe Docker / Portainer / backup / media operations)
- Sandboxed CLI execution (allow-list + firejail / bubblewrap)

**Phase 3+**
- Google Calendar, Gmail/IMAP tools
- Local TTS (Piper) + STT (faster-whisper)
- Telegram bot + responsive PWA

## 3. Architecture (text diagram)
Internet / Mobile ── Cloudflare Zero Trust ── Traefik (edge network)
│
┌──────────┴──────────┐
│                     │
blaire-frontend           blaire-backend
(Next.js 15)               (FastAPI)
│                     │
└────── containers_core network ──────┘
│
┌─────────────────────┬───────────────┼───────────────┬─────────────────────┐
│                     │               │               │                     │
Agent Layer          RAG Engine      Tool Registry     MCP Client           Inference
(LangGraph)         (LlamaIndex)     (Brave/Searx/MCP/FS)   (Obsidian/HA/etc)      (Ollama)
│                     │               │               │                     │
└─────────────────────┴───────────────┴───────────────┴─────────────────────┘
│
Parallel sidecars:
• Qdrant
• SearxNG (optional)
• obsidian-mcp-server
• ha-mcp-server
• homelab-mcp (Phase 2)
text## 4. Final Tech Stack (all open-source)

**Frontend**  
Next.js 15 (App Router · TypeScript · Turbopack)  
Tailwind CSS + shadcn/ui + Radix UI  
TanStack Query v5 · Zustand · React Hook Form + Zod  
Streaming via native fetch / ReadableStream

**Backend**  
Python 3.12 + FastAPI (async)  
LangGraph + LangChain  
LlamaIndex (RAG + Obsidian loader + multimodal parsers)  
Qdrant vector database  
Pydantic v2 + structlog

**Inference**  
Ollama (http://ollama:11434 inside containers_core)  
Default models:  
• Supervisor → qwen2.5:7b-instruct or llama3.2:3b (Q5)  
• Research agents → phi-3.5-mini-instruct + gemma2:2b  
• Embeddings → nomic-embed-text-v1.5  
• Vision → llava:13b or bakllava (on-demand)

**Other**  
MCP protocol → official Python SDK + community servers  
Search → brave-search python client + SearxNG JSON API  
Sandboxing → firejail or bubblewrap + strict allow-lists  
Deployment → Docker Compose + Watchtower

## 5. Folder Structure & Authoritative Paths
/srv/containers/blaire/
├── docker-compose.yml
├── .env
├── .env.example
├── backend/
│   ├── app/
│   │   ├── init.py
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── agents/
│   │   ├── rag/
│   │   ├── tools/
│   │   ├── mcp/
│   │   └── models/
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   ├── chat/
│   │   ├── settings/
│   │   └── knowledge/
│   ├── components/
│   ├── lib/
│   │   └── api.ts
│   ├── Dockerfile
│   └── next.config.mjs
├── data/
│   ├── qdrant/           → persistent volume
│   └── logs/
├── knowledge/
│   └── drop/             → bind-mounted from ${DROP_FOLDER}
├── mcp-obsidian/         → git submodule or clone
└── mcp-ha/               → git submodule or clone
text**Important bind mounts / volumes**

- `${DROP_FOLDER:-/srv/ai-knowledge/drop}:/app/knowledge/drop`
- `/srv/obsidian/BlacksiteLabVault:/vault:ro` (or rw for MCP write support)
- `/srv/blacksitelab/blaire/qdrant:/qdrant/storage`
- `/mnt/backups/blaire:/backups`

## 6. Next realistic steps (you pick order)

1. Create directory `/srv/containers/blaire/`
2. Paste → save this file as `BLAIRE-DESIGN-v0.2.md`
3. Create `docker-compose.yml` + `.env.example`
4. Scaffold either:
   - frontend skeleton (Next.js + basic chat page + api proxy)
   - backend skeleton (FastAPI + health endpoint + LangGraph supervisor stub)
5. Deploy Qdrant + Ollama connectivity test
6. Implement drop-folder watcher + first ingestion pipeline

If you would like the next piece (docker-compose.yml skeleton, .env.example, FastAPI main.py structure, Next.js layout.tsx, etc.), just tell me which one to generate first.