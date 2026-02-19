#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f .env ]]; then
  echo ".env is missing. Copy .env.example to .env first." >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required." >&2
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "docker compose plugin is required." >&2
  exit 1
fi

get_env_value() {
  local key="$1"
  local value
  value="$(grep -E "^${key}=" .env | head -n 1 | cut -d= -f2- || true)"
  printf '%s' "$value"
}

require_nonempty() {
  local key="$1"
  local value
  value="$(get_env_value "$key")"
  if [[ -z "$value" ]]; then
    echo "Missing required .env value: ${key}" >&2
    exit 1
  fi
}

reject_placeholder() {
  local key="$1"
  local value
  value="$(get_env_value "$key")"
  if [[ "$value" == "change_me" || "$value" == "change_me_admin_key" || "$value" == "change_me_user_key" ]]; then
    echo "Placeholder value detected for ${key}; set a real value first." >&2
    exit 1
  fi
}

ensure_path_writable() {
  local label="$1"
  local path="$2"
  if [[ -z "$path" ]]; then
    return
  fi
  mkdir -p "$path" 2>/dev/null || {
    echo "Cannot create/access ${label} at ${path}" >&2
    exit 1
  }
}

require_nonempty "POSTGRES_PASSWORD"
require_nonempty "DATABASE_URL"
require_nonempty "ADMIN_API_KEYS"
require_nonempty "USER_API_KEYS"
require_nonempty "FRONTEND_PROXY_API_KEY"

reject_placeholder "POSTGRES_PASSWORD"
reject_placeholder "ADMIN_API_KEYS"
reject_placeholder "USER_API_KEYS"
reject_placeholder "FRONTEND_PROXY_API_KEY"

for path_key in POSTGRES_DATA_PATH QDRANT_STORAGE_PATH DROP_FOLDER OBSIDIAN_VAULT_PATH BACKUP_PATH LOG_PATH LOCALAI_MODELS_PATH; do
  ensure_path_writable "$path_key" "$(get_env_value "$path_key")"
done

frontend_port="$(get_env_value FRONTEND_HOST_PORT)"
backend_port="$(get_env_value BACKEND_HOST_PORT)"
frontend_port="${frontend_port:-3000}"
backend_port="${backend_port:-8001}"
if [[ "$frontend_port" == "$backend_port" ]]; then
  echo "FRONTEND_HOST_PORT and BACKEND_HOST_PORT cannot be the same (${frontend_port})." >&2
  exit 1
fi

if command -v ss >/dev/null 2>&1; then
  if ss -ltn "( sport = :${frontend_port} )" | grep -q LISTEN; then
    echo "Warning: FRONTEND_HOST_PORT ${frontend_port} is currently in use."
  fi
  if ss -ltn "( sport = :${backend_port} )" | grep -q LISTEN; then
    echo "Warning: BACKEND_HOST_PORT ${backend_port} is currently in use."
  fi
fi

echo "Preflight checks passed."
