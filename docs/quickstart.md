# Quickstart

For first-time users, use `docs/installation_guide.md` as the primary guide.

If you already know the stack and want the shortest flow:

1. Run automated bootstrap:
   - Linux/macOS: `bash ops/bootstrap.sh`
   - Windows PowerShell: `./ops/bootstrap.ps1`
   - no manual `.env` edits required for default first run
2. Open frontend (`http://localhost:${FRONTEND_HOST_PORT:-3000}`).
3. Complete `/setup` (save API key, verify access, review diagnostics).
4. Start using `/chat`.

For troubleshooting, see `docs/troubleshooting.md`.
