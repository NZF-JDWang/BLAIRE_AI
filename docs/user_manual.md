# User Manual

## Login/auth model
- BLAIRE uses API keys.
- Use `/setup` for first-run key setup and verification.
- You can also set/update key in `Settings` page (`API key` field).
- This key is sent on frontend API calls and required by backend routes when auth is enabled.
- The header status strip shows current role, effective search mode, sensitive-action state, and dependency health summary.

## Chat
1. Open `/chat`.
2. Select model class and optional model override.
3. Optionally set request-level controls (`temperature`, `top_p`, `max_tokens`, context window, RAG toggle, retrieval K).
4. Send prompt.
5. Response streams token-by-token.
6. RAG status and citations appear under the response.

Model tips:
- Overrides are validated against the backend allowlist for the selected class.
- Use `GET /runtime/options` or `GET /models` to see currently allowed models.
- For configuration details, see `docs/model_selection.md`.

## Search
- Open `/search` to run direct web search with configured mode:
  - `searxng_only`
  - `brave_only`
  - `auto_fallback`
  - `parallel`

## Swarm research
1. Open `/swarm`.
2. Enter a topic and run.
3. View supervisor summary + worker outputs.
4. View live swarm runs and trace steps in the live panel.

## Knowledge
Open `/knowledge` to:
- Upload supported files (txt/md/pdf/image)
- Refresh index status
- Trigger Obsidian delta/full reindex

## Approvals
Open `/approvals` (admin key required) to:
- View pending queue and recent history
- Approve/reject requests
- Execute approved actions with token/payload hash
- Load audit events per approval

## Settings
- Configure:
  - API key
  - search mode preference
  - model class/override preference
  - generation defaults (`temperature`, `top_p`, `max_tokens`, `context_window_tokens`)
  - default retrieval controls (`use_rag`, `retrieval_k`)
- Admin keys can also configure live runtime policy overrides:
  - search default
  - sensitive actions toggle
  - approval TTL
  - network/obsidian/HA/homelab allowlists

## Capabilities
- Open `/capabilities` to view:
  - tool registry and action class
  - dependency and MCP readiness
  - policy-related availability context

## Tools
- Open `/tools` to inspect:
  - action class and sensitivity level
  - approval requirement implications
  - active allowlists and policy constraints
