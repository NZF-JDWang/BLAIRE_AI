#!/usr/bin/env bash
set -Eeuo pipefail

# Remote deployment script for Docker Compose stacks.
# Expected to run on the deployment host inside a checked-out repo.

DEPLOY_GIT_REMOTE="${DEPLOY_GIT_REMOTE:-origin}"
DEPLOY_GIT_BRANCH="${DEPLOY_GIT_BRANCH:-main}"
DEPLOY_COMPOSE_FILE="${DEPLOY_COMPOSE_FILE:-docker-compose.yml}"
DEPLOY_COMPOSE_PROFILES="${DEPLOY_COMPOSE_PROFILES:-}"
DEPLOY_BUILD_IMAGES="${DEPLOY_BUILD_IMAGES:-true}"
DEPLOY_HEALTH_PATHS="${DEPLOY_HEALTH_PATHS:-/health,/health/dependencies}"
DEPLOY_HEALTH_RETRIES="${DEPLOY_HEALTH_RETRIES:-30}"
DEPLOY_HEALTH_INTERVAL_SECONDS="${DEPLOY_HEALTH_INTERVAL_SECONDS:-5}"
DEPLOY_HEALTH_URL="${DEPLOY_HEALTH_URL:-}"
DEPLOY_API_KEY="${DEPLOY_API_KEY:-}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required on deploy host" >&2
  exit 1
fi

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "script must be run inside a git repository" >&2
  exit 1
fi

previous_sha="$(git rev-parse HEAD)"
rolled_back="false"

compose_cmd=(docker compose -f "$DEPLOY_COMPOSE_FILE")
if [[ -n "$DEPLOY_COMPOSE_PROFILES" ]]; then
  IFS=',' read -r -a profiles <<< "$DEPLOY_COMPOSE_PROFILES"
  for profile in "${profiles[@]}"; do
    trimmed="${profile#"${profile%%[![:space:]]*}"}"
    trimmed="${trimmed%"${trimmed##*[![:space:]]}"}"
    if [[ -n "$trimmed" ]]; then
      compose_cmd+=(--profile "$trimmed")
    fi
  done
fi

rollback() {
  if [[ "$rolled_back" == "true" ]]; then
    return
  fi

  rolled_back="true"
  echo "Deploy failed. Rolling back to ${previous_sha}..."
  set +e
  git reset --hard "$previous_sha"
  if [[ "${DEPLOY_BUILD_IMAGES,,}" == "true" ]]; then
    "${compose_cmd[@]}" up -d --build --remove-orphans
  else
    "${compose_cmd[@]}" up -d --remove-orphans
  fi
  set -e
}

trap rollback ERR

echo "Fetching latest ${DEPLOY_GIT_REMOTE}/${DEPLOY_GIT_BRANCH}..."
git fetch "$DEPLOY_GIT_REMOTE" "$DEPLOY_GIT_BRANCH"
git checkout "$DEPLOY_GIT_BRANCH"
git reset --hard "${DEPLOY_GIT_REMOTE}/${DEPLOY_GIT_BRANCH}"

echo "Pulling available images..."
"${compose_cmd[@]}" pull --ignore-pull-failures

echo "Applying compose update..."
if [[ "${DEPLOY_BUILD_IMAGES,,}" == "true" ]]; then
  "${compose_cmd[@]}" up -d --build --remove-orphans
else
  "${compose_cmd[@]}" up -d --remove-orphans
fi

headers=()
if [[ -n "$DEPLOY_API_KEY" ]]; then
  headers=(-H "X-API-Key: ${DEPLOY_API_KEY}")
fi

health_check() {
  local path="$1"
  if [[ -n "$DEPLOY_HEALTH_URL" ]]; then
    local endpoint="${DEPLOY_HEALTH_URL%/}${path}"
    curl -fsS "${headers[@]}" "$endpoint" >/dev/null
    return
  fi

  local header_key="${DEPLOY_API_KEY}"
  "${compose_cmd[@]}" exec -T backend python -c "import urllib.request; req=urllib.request.Request('http://localhost:8000${path}', headers={'X-API-Key':'${header_key}'} if '${header_key}' else {}); urllib.request.urlopen(req, timeout=10).read()"
}

IFS=',' read -r -a health_paths <<< "$DEPLOY_HEALTH_PATHS"
for path in "${health_paths[@]}"; do
  trimmed="${path#"${path%%[![:space:]]*}"}"
  trimmed="${trimmed%"${trimmed##*[![:space:]]}"}"
  if [[ -z "$trimmed" ]]; then
    continue
  fi

  if [[ -n "$DEPLOY_HEALTH_URL" ]]; then
    echo "Health checking ${DEPLOY_HEALTH_URL%/}${trimmed}..."
  else
    echo "Health checking backend container ${trimmed}..."
  fi
  ok="false"
  for ((i = 1; i <= DEPLOY_HEALTH_RETRIES; i++)); do
    if health_check "$trimmed"; then
      ok="true"
      break
    fi
    sleep "$DEPLOY_HEALTH_INTERVAL_SECONDS"
  done

  if [[ "$ok" != "true" ]]; then
    echo "Health check failed for ${trimmed}" >&2
    exit 1
  fi
done

echo "Deployment succeeded."
"${compose_cmd[@]}" ps
