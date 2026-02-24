## Purpose

BLAIRE Core is a persistent AI agent written in Python designed to:
- Maintain long-term structured memory
- Interact via Telegram
- Monitor homelab systems
- Execute safe tools
- Operate with a heartbeat to simulate autonomy

BLAIRE Core is:
- Model-agnostic (uses Ollama over HTTP)
- Stateful (memory stored locally in `/data`)
- Container-optional during development
- Deployable on BSL1 in production

---

## Architectural Layers

1. Orchestrator (Python)
   - Entry point
   - Manages memory
   - Builds prompts
   - Executes tools
   - Handles heartbeat

1. Memory Store
   - JSON + MD files under `/data`
   - Separate from Obsidian

1. Tool Layer
   - Controlled execution layer
   - Read-only by default

1. LLM Layer
   - Ollama endpoint
   - Small local models

1. Interface Layer
   - Telegram (primary)
   - CLI (dev)

---

## Design Principles

- Infinite storage, tiny working context
- Strict tool permission boundaries
- Explicit memory distillation
- No uncontrolled shell access
- No core memory in Obsidian