#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f .env ]]; then
  echo ".env is missing. Copy .env.example to .env first." >&2
  exit 1
fi

compose_cmd=(docker compose)
for profile in "$@"; do
  if [[ -n "$profile" ]]; then
    compose_cmd+=(--profile "$profile")
  fi
done

get_env_value() {
  local key="$1"
  local value
  value="$(grep -E "^${key}=" .env | head -n 1 | cut -d= -f2- || true)"
  printf '%s' "$value"
}

admin_key="$(get_env_value ADMIN_API_KEYS | cut -d, -f1)"
user_key="$(get_env_value USER_API_KEYS | cut -d, -f1)"

if [[ -z "$admin_key" || -z "$user_key" ]]; then
  echo "ADMIN_API_KEYS and USER_API_KEYS must be set for smoke tests." >&2
  exit 1
fi

call_endpoint() {
  local path="$1"
  local key="$2"
  local method="${3:-GET}"
  "${compose_cmd[@]}" exec -T backend python -c "import urllib.request; headers={}; key='${key}'; headers.update({'X-API-Key': key} if key else {}); req=urllib.request.Request('http://localhost:8000${path}', method='${method}', headers=headers); urllib.request.urlopen(req, timeout=20).read(); print('${path} ok')"
}

echo "Running smoke tests..."
call_endpoint "/health" ""
call_endpoint "/health/dependencies" "$user_key"
call_endpoint "/runtime/options" "$user_key"
call_endpoint "/models" "$user_key"
call_endpoint "/ops/status" "$admin_key"
echo "Smoke tests passed."
