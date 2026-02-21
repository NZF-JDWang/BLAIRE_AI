# Quickstart

Use this for the fastest first run. For full detail, see `docs/installation_guide.md`.

## TL;DR (Linux / Docker)

1. Clone and enter repo:
   ```bash
   git clone https://github.com/NZF-JDWang/BLAIRE_AI.git
   cd BLAIRE_AI
   ```
2. Create env file:
   ```bash
   cp .env.example .env
   ```
3. Run bootstrap:
   ```bash
   bash ops/bootstrap.sh
   ```
4. Open UI: `http://localhost:3000`
5. Open `Settings` and paste one value from `USER_API_KEYS` in `.env`.

## Default host ports

| Service | Host port |
| --- | --- |
| Frontend | `3000` |
| Backend | `8001` |
| LocalAI | `8082` |
| SearxNG | `8080` |

## Profile behavior in bootstrap

- Bootstrap auto-enables compose profiles from `.env` values.
- Common default behavior is `search` and `mcp` enabled when their internal service URLs are configured.
- To force search off before bootstrap, set `ENABLE_SEARCH=false` in `.env`.
- For MCP and GPU, use:
  - `ENABLE_MCP_SERVICES=true|false`
  - `ENABLE_VLLM=true|false`

For troubleshooting, see `docs/troubleshooting.md`.
