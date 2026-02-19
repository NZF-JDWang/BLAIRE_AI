#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

generate_secret() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex 24
    return
  fi
  python - <<'PY'
import secrets
print(secrets.token_hex(24))
PY
}

ensure_env_value() {
  local key="$1"
  local value="$2"
  if grep -qE "^${key}=" .env; then
    local current
    current="$(grep -E "^${key}=" .env | head -n 1 | cut -d= -f2-)"
    if [[ -n "${current}" ]]; then
      return
    fi
    sed -i "s|^${key}=.*|${key}=${value}|" .env
    return
  fi
  printf "\n%s=%s\n" "$key" "$value" >> .env
}

ensure_env_value "POSTGRES_PASSWORD" "$(generate_secret)"
ensure_env_value "ADMIN_API_KEYS" "$(generate_secret)"
ensure_env_value "USER_API_KEYS" "$(generate_secret)"
ensure_env_value "FRONTEND_PROXY_API_KEY" "$(generate_secret)"

sync_database_url_password() {
  local postgres_password
  postgres_password="$(grep -E '^POSTGRES_PASSWORD=' .env | head -n 1 | cut -d= -f2-)"
  if [[ -z "${postgres_password}" ]]; then
    return
  fi

  python - "$postgres_password" <<'PY'
import re
import sys
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit

env_path = Path(".env")
text = env_path.read_text(encoding="utf-8")
match = re.search(r"^DATABASE_URL=(.+)$", text, re.MULTILINE)
if not match:
    sys.exit(0)

database_url = match.group(1).strip()
parsed = urlsplit(database_url)
if not parsed.scheme.startswith("postgresql"):
    sys.exit(0)
if parsed.username is None or parsed.hostname is None:
    sys.exit(0)

existing_password = unquote(parsed.password or "")
if existing_password not in ("", "change_me"):
    sys.exit(0)

new_password = sys.argv[1]
username = quote(unquote(parsed.username), safe="")
host = parsed.hostname
port = f":{parsed.port}" if parsed.port is not None else ""
netloc = f"{username}:{quote(new_password, safe='')}@{host}{port}"
updated_url = urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))

updated = text[: match.start(1)] + updated_url + text[match.end(1) :]
env_path.write_text(updated, encoding="utf-8")
print("Updated DATABASE_URL password to match POSTGRES_PASSWORD")
PY
}

sync_database_url_password

if command -v ss >/dev/null 2>&1; then
  frontend_port="$(grep -E '^FRONTEND_HOST_PORT=' .env | head -n 1 | cut -d= -f2-)"
  backend_port="$(grep -E '^BACKEND_HOST_PORT=' .env | head -n 1 | cut -d= -f2-)"
  frontend_port="${frontend_port:-3000}"
  backend_port="${backend_port:-8001}"

  if ss -ltn "( sport = :${frontend_port} )" | grep -q LISTEN; then
    echo "Warning: FRONTEND_HOST_PORT ${frontend_port} is already in use"
  fi
  if ss -ltn "( sport = :${backend_port} )" | grep -q LISTEN; then
    echo "Warning: BACKEND_HOST_PORT ${backend_port} is already in use"
  fi
fi

docker compose up -d

admin_key="$(grep -E '^ADMIN_API_KEYS=' .env | head -n 1 | cut -d= -f2- | cut -d, -f1)"
if [[ -z "${admin_key}" ]]; then
  echo "ADMIN_API_KEYS is empty; cannot call /ops/init" >&2
  exit 1
fi

for attempt in {1..30}; do
  if docker compose exec -T backend python -c "import urllib.request; req=urllib.request.Request('http://localhost:8000/health', headers={'X-API-Key':'${admin_key}'}); urllib.request.urlopen(req, timeout=5).read()"; then
    break
  fi
  if [[ "$attempt" -eq 30 ]]; then
    echo "Backend did not become ready in time." >&2
    exit 1
  fi
  sleep 2
done

for attempt in {1..10}; do
  if docker compose exec -T backend python -c "import urllib.request; req=urllib.request.Request('http://localhost:8000/ops/init', method='POST', headers={'X-API-Key':'${admin_key}'}); urllib.request.urlopen(req, timeout=30).read(); print('ops/init completed')"; then
    break
  fi
  if [[ "$attempt" -eq 10 ]]; then
    echo "Failed to call /ops/init after retries." >&2
    exit 1
  fi
  sleep 2
done

echo "Bootstrap completed successfully."
