# User Manual

## Login/auth model
- BLAIRE uses API keys.
- Set your user API key in `Settings` page (`API key` field).
- This key is sent on frontend API calls and required by backend routes when auth is enabled.

## Chat
1. Open `/chat`.
2. Select model class and optional model override.
3. Send prompt.
4. Response streams token-by-token.
5. RAG status and citations appear under the response.

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
- View pending/recent approvals
- Approve/reject requests
- Execute approved actions with token/payload hash

## Settings
- Configure:
  - API key
  - search mode preference
  - model class/override preference
