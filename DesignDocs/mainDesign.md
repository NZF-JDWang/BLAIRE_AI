You are an expert Python architect and developer. You are helping me scaffold and implement a project called **BLAIRE Core v0.1**.

BLAIRE Core is a Python service that:
- Talks to a local LLM via Ollama (HTTP API)
- Maintains its own file-based memory under a `/data` directory
- Supports conversational sessions (short-term memory + summaries)
- Has a heartbeat loop for periodic “ticks”
- Uses a tool layer to perform safe, read-only checks
- Will eventually integrate with Telegram (but that can be stubbed first)
- Does NOT store its core memory in Obsidian

IMPORTANT:
- Start small and build in phases.
- No Docker, no containers, no web frameworks in v0.1. This is a normal Python app.
- Prefer standard library where possible.
- Write clean, readable Python 3.11+.
- Use type hints.
- Use docstrings for public functions and classes.
- Put configuration in a simple JSON or YAML file.

==================================================
== HIGH-LEVEL DESIGN (FOLLOW THIS ARCHITECTURE) ==
==================================================

Project root name: `blaire_core`

Directory layout (initially):

blaire_core/
  src/
    blaire_core/
      __init__.py
      config.py
      main.py
      orchestrator.py
      memory/
        __init__.py
        store.py
        models.py
      llm/
        __init__.py
        client.py
      tools/
        __init__.py
        registry.py
        builtin_tools.py
      heartbeat/
        __init__.py
        loop.py
      interfaces/
        __init__.py
        cli.py
        telegram_stub.py
  data/
    profile.json
    preferences.json
    projects.json
    todos.json
    sessions/
    episodic/
    long_term/
      facts.jsonl
      lessons.jsonl
  config/
    dev.json
    prod.json
  tests/

You do NOT need to fully implement everything at once.
We will build this in stages. Each stage must leave the project in a runnable state.

==================================================
== MEMORY MODEL (IMPLEMENT THIS STRUCTURE) ==
==================================================

The memory layer is **file-based**, under `data/`.

Files:

- `profile.json`
  - Stable identity facts.
  - Example fields:
    - `name`: string
    - `environment_summary`: string
    - `long_term_goals`: list of strings
    - `behavioral_constraints`: list of strings

- `preferences.json`
  - Behavior and preferences.
  - Example fields:
    - `response_style`: string (e.g. "concise")
    - `autonomy_level`: string (e.g. "observe" | "assist")
    - `quiet_hours`: [start, end] strings ("23:00", "08:00")
    - `notification_limits`: { "max_per_day": int }

- `projects.json`
  - List of projects.
  - Each project:
    - `id`: string
    - `name`: string
    - `description`: string
    - `status`: string ("active" | "paused" | "done")
    - `priority`: string ("low" | "medium" | "high")
    - `summary_card`: string (compressed summary)
    - `next_actions`: list of strings

- `todos.json`
  - List of todos.
  - Each todo:
    - `id`: string
    - `project_id`: string
    - `title`: string
    - `description`: string
    - `priority`: string ("now" | "soon" | "later")
    - `status`: string ("open" | "in_progress" | "done")
    - `created_at`: ISO timestamp string
    - `last_updated`: ISO timestamp string

- `sessions/`
  - One file per session (e.g. `session-<id>.json`).
  - Contains:
    - `id`
    - `created_at`
    - `messages`: list of {role: "user" | "assistant", "content": string, "timestamp": string}
    - `running_summary`: string

- `episodic/`
  - One markdown file per day (e.g. `2026-02-24.md`).
  - Free-form summary and bullet points.

- `long_term/facts.jsonl`
  - JSON Lines file.
  - Each line:
    - `id`
    - `type`: "user_fact" | "system_fact" | "lesson"
    - `text`: string
    - `tags`: list of strings
    - `importance`: float 0–1
    - `created_at`: ISO string
    - `last_used`: ISO string or null

- `long_term/lessons.jsonl`
  - Same shape as `facts.jsonl` but focused on “lessons learned”.

The `memory/store.py` module should expose a clean Python API to:
- Load and save these structures.
- Append new facts.
- Update last_used.
- Create and update sessions.
- Append to episodic files.

Use simple file locking or careful write patterns (write to temp file, then rename) to avoid corrupting JSON.

==================================================
== LLM CLIENT (OLLAMA) ==
==================================================

Create `llm/client.py` with an `OllamaClient` class that:

- Reads base URL and model name from config.
- Has methods like:
  - `generate(system_prompt: str, messages: list[dict], max_tokens: int) -> str`
- Uses the Ollama HTTP API (openai-compatible or /api/chat, whichever is simpler).
- Is synchronous for now.
- Has basic error handling and timeouts.

We will use a **two-pass** pattern later:
- Planning pass (small prompt).
- Answer pass (full retrieved context).

For now, just support a single `generate()` that accepts a system prompt and a chat history.

==================================================
== PROMPT & CONTEXT BUDGET (HIGH-LEVEL) ==
==================================================

Don’t implement full budgeting logic yet, but design the orchestrator around these sections:

- Soul / rules (string)
- Profile + preferences (compressed “card”)
- Project cards (0–2 cards max)
- Long-term facts (0–10 small snippets)
- Session summary
- Last N messages
- Tool results (if any)

The orchestrator should be structured so it’s easy to plug in retrieval logic later.

==================================================
== TOOL LAYER (READ-ONLY FOR NOW) ==
==================================================

In `tools/registry.py` and `tools/builtin_tools.py`:

- Define a simple `Tool` data structure:
  - `name`
  - `description`
  - `risk_level` ("safe" for v0.1)
  - `callable` / function reference

- Implement a small registry where tools can be registered and looked up by name.

For v0.1, implement stub or basic versions of:

- `local_search` (for now, could just search within facts.jsonl by keyword)
- `web_search` (can be a stub that returns dummy data)
- `check_disk_space` (safe implementation that returns disk usage on the dev machine)
- `check_docker_containers` (optional in dev; can return a dummy list)

The important part is the **interface and registry**, not full functionality.

==================================================
== HEARTBEAT LAYER ==
==================================================

In `heartbeat/loop.py`:

- Implement a simple heartbeat loop that can be called manually or run in a background thread.
- It should:
  - Accept a config for interval (e.g. every N seconds).
  - On each tick, call a function in `orchestrator.py` like `run_heartbeat_tick()`.
- For v0.1, `run_heartbeat_tick()` can just:
  - Log that a heartbeat occurred.
  - Maybe append a line to today’s episodic file.

Later we will add modes (maintenance, reflection, project work, social), but for now keep it minimal and safe.

==================================================
== INTERFACE LAYER (CLI FIRST, TELEGRAM LATER) ==
==================================================

In `interfaces/cli.py`:

- Implement a simple REPL:
  - User types a message.
  - Orchestrator handles it.
  - Response is printed.
  - Memory gets updated.

In `interfaces/telegram_stub.py`:

- Just create placeholders:
  - A class or functions that *will* integrate Telegram later.
  - For now, they may just raise NotImplementedError.

==================================================
== ORCHESTRATOR ==
==================================================

In `orchestrator.py`:

- Implement the main high-level functions:

1. `handle_user_message(session_id: str, user_message: str) -> str`
   - Loads session or creates a new one.
   - Appends the user message to session.
   - Loads relevant memory (for v0.1 this can be very simple).
   - Calls `OllamaClient.generate(...)` with a basic prompt.
   - Appends assistant response to session.
   - Returns assistant response.

1. `run_heartbeat_tick()`
   - For v0.1, append a simple entry to today’s episodic memory: e.g. "Heartbeat tick at <time>".

Add hooks for a future “learning routine”, but you don’t have to fully implement that yet.

==================================================
== STAGED IMPLEMENTATION PLAN ==
==================================================

Follow these stages. After each stage, ensure the project runs without errors.

STAGE 1 – Project Skeleton
- Create the directory layout.
- Create `pyproject.toml` or `requirements.txt` if needed.
- Implement empty modules with basic classes/functions and docstrings.
- Implement `config.py` that loads a JSON config from `config/dev.json`.

STAGE 2 – Memory Store Basics
- Implement `memory/models.py` with simple dataclasses or TypedDicts for:
  - Profile
  - Preferences
  - Project
  - Todo
  - Fact
- Implement `memory/store.py` with:
  - Functions to load and save the main JSON files.
  - A function to initialize default files if they don’t exist.
- Add a small test or script to verify that reading/writing memory works.

STAGE 3 – LLM Client
- Implement `llm/client.py` with `OllamaClient`.
- Add config fields for:
  - `llm.base_url`
  - `llm.model`
- Add a small test script that sends a simple prompt and prints the response (this can be a separate test function or CLI command).

STAGE 4 – Orchestrator + CLI
- Implement `orchestrator.handle_user_message(...)` in a minimal way:
  - No fancy retrieval yet.
  - Just load profile, optionally include a small system prompt, and pass user message to the LLM.
- Implement `interfaces/cli.py`:
  - Simple loop that uses `handle_user_message` with a fixed `session_id`.
- Add an entry point in `main.py` to start the CLI.

STAGE 5 – Session Persistence
- Enhance `orchestrator.handle_user_message` to:
  - Load or create a session file under `data/sessions/`.
  - Append user and assistant messages to the session file.
  - Maintain a simple `running_summary` field (can be stubbed initially).
- Ensure that restarting the process and using the same `session_id` continues the conversation.

STAGE 6 – Heartbeat Skeleton
- Implement `heartbeat.loop` with:
  - A function to run a single tick.
  - Optionally, a function to run a simple loop with `time.sleep`.
- Implement `orchestrator.run_heartbeat_tick()` that:
  - Writes a line into today’s `episodic/YYYY-MM-DD.md`.
- Wire a CLI command or simple script to trigger one heartbeat tick for testing.

STAGE 7 – Tool Registry and One Example Tool
- Implement `tools/registry.py` with:
  - A registry dict.
  - Functions to register and retrieve tools.
- Implement one simple safe tool in `builtin_tools.py`, e.g. `check_disk_space` using `shutil.disk_usage`.
- Add a function in orchestrator to call a tool by name and return its result.

==================================================
== STYLE AND QUALITY EXPECTATIONS ==
==================================================

- Use Python 3.11+ features where helpful.
- Use type hints everywhere reasonable.
- Add inline comments where non-obvious decisions are made.
- Avoid premature optimization; readability first.
- No external dependencies unless truly necessary (standard library preferred for v0.1).

Start now by creating the project skeleton (STAGE 1) according to this specification.
As you complete each stage, I will run the code and then ask you to move to the next stage.