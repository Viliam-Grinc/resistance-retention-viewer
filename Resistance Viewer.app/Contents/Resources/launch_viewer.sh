#!/bin/bash
# Bundled inside Resistance Viewer.app — starts Streamlit and opens the browser.
set -euo pipefail

PORT="${RESISTANCE_VIEWER_PORT:-8501}"
URL="http://127.0.0.1:${PORT}"
LOG_DIR="${HOME}/Library/Logs/Resistance Viewer"
LOG_FILE="${LOG_DIR}/viewer.log"
CONFIG_DIR="${HOME}/Library/Application Support/Resistance Viewer"
PID_FILE="${CONFIG_DIR}/streamlit.pid"

notify() {
  /usr/bin/osascript -e "display notification \"${2}\" with title \"Resistance Viewer\"" 2>/dev/null || true
}

alert() {
  /usr/bin/osascript -e "display alert \"Resistance Viewer\" message \"${1}\"" 2>/dev/null || true
}

log() {
  mkdir -p "${LOG_DIR}"
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >>"${LOG_FILE}"
}

open_browser() {
  /usr/bin/open "${URL}" >/dev/null 2>&1 || true
}

server_listening() {
  /usr/sbin/lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1
}

server_healthy() {
  /usr/bin/curl -sf --max-time 2 "${URL}/_stcore/health" >/dev/null 2>&1
}

if ! command -v python3 >/dev/null 2>&1; then
  alert "Python 3 is not installed. Install Python 3.10+ and try again."
  exit 1
fi

if [[ -z "${PROJECT_ROOT:-}" ]] || [[ ! -f "${PROJECT_ROOT}/requirements.txt" ]]; then
  alert "Could not find the project folder (missing requirements.txt). Keep Resistance Viewer.app in the project folder, or select the folder when prompted."
  exit 1
fi

mkdir -p "${LOG_DIR}" "${CONFIG_DIR}"
log "Launch requested (project=${PROJECT_ROOT})"

if server_listening; then
  log "Server already listening on port ${PORT}; opening browser"
  open_browser
  exit 0
fi

# venv lives outside the project so Finder-launched apps can write (Documents is restricted).
VENV_KEY="$(printf '%s' "${PROJECT_ROOT}" | /usr/bin/shasum -a 256 | awk '{print $1}' | cut -c1-16)"
VENV="${CONFIG_DIR}/venvs/${VENV_KEY}"
PY="${VENV}/bin/python"

if [[ ! -x "${PY}" ]]; then
  notify "Resistance Viewer" "First run: creating virtual environment…"
  log "Creating venv at ${VENV}"
  python3 -m venv "${VENV}" >>"${LOG_FILE}" 2>&1
fi

if ! "${PY}" -c "import streamlit" >>"${LOG_FILE}" 2>&1; then
  notify "Resistance Viewer" "First run: installing dependencies…"
  log "Installing requirements"
  "${PY}" -m pip install -q --upgrade pip >>"${LOG_FILE}" 2>&1
  "${PY}" -m pip install -q -r "${PROJECT_ROOT}/requirements.txt" >>"${LOG_FILE}" 2>&1
fi

log "Starting Streamlit on port ${PORT}"
(
  cd "${PROJECT_ROOT}"
  exec "${PY}" -m streamlit run src/resistance_viewer/app.py \
    --server.headless true \
    --server.port "${PORT}" \
    --browser.serverAddress 127.0.0.1 \
    --browser.gatherUsageStats false
) >>"${LOG_FILE}" 2>&1 &

disown 2>/dev/null || true
echo $! >"${PID_FILE}"
log "Streamlit PID $(cat "${PID_FILE}")"

notify "Resistance Viewer" "Starting… your browser will open when ready."

attempt=0
max_attempts=180
while (( attempt < max_attempts )); do
  if server_healthy; then
    log "Server healthy; opening browser"
    open_browser
    exit 0
  fi
  if ! server_listening && (( attempt > 5 )); then
    log "Streamlit exited before listening (see ${LOG_FILE})"
    alert "The viewer stopped unexpectedly. See ${LOG_FILE} for details."
    exit 1
  fi
  sleep 1
  (( attempt += 1 )) || true
done

log "Timed out waiting for health check; opening browser anyway"
notify "Resistance Viewer" "Still starting… open ${URL} if the browser did not appear."
open_browser
exit 0
