#!/bin/bash
# Stop a broken Streamlit instance and reset cached venv(s) for Resistance Viewer.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PORT="${RESISTANCE_VIEWER_PORT:-8501}"
CONFIG_DIR="${HOME}/Library/Application Support/Resistance Viewer"
PID_FILE="${CONFIG_DIR}/streamlit.pid"

stop_port() {
  if [[ -f "${PID_FILE}" ]]; then
    local pid
    pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "Stopping Streamlit PID ${pid}"
      kill "${pid}" 2>/dev/null || true
    fi
    rm -f "${PID_FILE}"
  fi
  local pids
  pids="$(/usr/sbin/lsof -t -iTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    echo "Stopping process(es) on port ${PORT}: ${pids}"
    # shellcheck disable=SC2086
    kill ${pids} 2>/dev/null || true
  fi
}

echo "Stopping anything on port ${PORT}…"
stop_port

if [[ -d "${CONFIG_DIR}/venvs" ]]; then
  echo "Removing cached virtual environments:"
  echo "  ${CONFIG_DIR}/venvs"
  rm -rf "${CONFIG_DIR}/venvs"
fi

echo ""
echo "Re-bundling app sources…"
"${SCRIPT_DIR}/register_app.sh"

echo ""
echo "Done. Double-click Resistance Viewer.app (or run ./scripts/launch_viewer.sh)."
echo "First launch will recreate the venv and reinstall dependencies."
