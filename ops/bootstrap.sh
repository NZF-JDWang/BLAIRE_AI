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

docker compose exec -T backend python -c "import urllib.request; req=urllib.request.Request('http://localhost:8000/ops/init', method='POST', headers={'X-API-Key':'${admin_key}'}); urllib.request.urlopen(req, timeout=30).read(); print('ops/init completed')"

echo "Bootstrap completed successfully."
