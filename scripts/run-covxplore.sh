#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UI_DIR="$ROOT_DIR/ui"
BACKEND_HOST="${COVXPLORE_BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${COVXPLORE_BACKEND_PORT:-8765}"

BACKEND_PID=""
UI_PID=""

cleanup() {
  local exit_code=$?

  if [[ -n "$UI_PID" ]] && kill -0 "$UI_PID" 2>/dev/null; then
    kill "$UI_PID" 2>/dev/null || true
  fi

  if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi

  wait "$UI_PID" "$BACKEND_PID" 2>/dev/null || true
  exit "$exit_code"
}

trap cleanup EXIT INT TERM

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing command: $1" >&2
    exit 1
  fi
}

wait_for_backend() {
  local url="http://${BACKEND_HOST}:${BACKEND_PORT}/api/health"
  local attempts="${COVXPLORE_BACKEND_WAIT_ATTEMPTS:-60}"

  for ((attempt=1; attempt<=attempts; attempt++)); do
    if curl --silent --fail "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done

  echo "Backend did not become ready: $url" >&2
  return 1
}

require_command curl
require_command npm

if command -v uv >/dev/null 2>&1; then
  BACKEND_CMD=(uv run covxplore-ui)
else
  require_command python
  BACKEND_CMD=(python -m covxplore_ui.main)
fi

if [[ ! -d "$UI_DIR/node_modules" ]]; then
  echo "Installing UI dependencies..."
  npm --prefix "$UI_DIR" install
fi

echo "Starting Covxplore API on http://${BACKEND_HOST}:${BACKEND_PORT} ..."
(
  cd "$ROOT_DIR"
  PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "${BACKEND_CMD[@]}"
) &
BACKEND_PID=$!

wait_for_backend

echo "Starting Covxplore UI on http://127.0.0.1:5173 ..."
VITE_API_BASE="http://${BACKEND_HOST}:${BACKEND_PORT}" npm --prefix "$UI_DIR" run dev &
UI_PID=$!

echo
echo "Covxplore running:"
echo "  UI:  http://127.0.0.1:5173"
echo "  API: http://${BACKEND_HOST}:${BACKEND_PORT}"
echo
echo "Press Ctrl+C to stop."

wait -n "$BACKEND_PID" "$UI_PID"
