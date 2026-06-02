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

stop_listening_server() {
  if [[ -f "${PID_FILE}" ]]; then
    local pid
    pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      log "Stopping Streamlit PID ${pid}"
      kill "${pid}" 2>/dev/null || true
      sleep 1
    fi
    rm -f "${PID_FILE}"
  fi
  local pids
  pids="$(/usr/sbin/lsof -t -iTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    log "Stopping process(es) on port ${PORT}: ${pids}"
    # shellcheck disable=SC2086
    kill ${pids} 2>/dev/null || true
    sleep 1
  fi
}

select_python3() {
  local py candidates=(
    "${RESISTANCE_VIEWER_PYTHON:-}"
    "/opt/homebrew/bin/python3"
    "/usr/local/bin/python3"
    "/Library/Frameworks/Python.framework/Versions/Current/bin/python3"
    "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
    "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"
    "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3"
  )
  local host_arch
  host_arch="$(uname -m)"

  for py in "${candidates[@]}"; do
    [[ -n "${py}" && -x "${py}" ]] || continue
    if ! "${py}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      continue
    fi
    if [[ "${host_arch}" == "arm64" ]]; then
      local py_arch
      py_arch="$("${py}" -c 'import platform; print(platform.machine())' 2>/dev/null || echo unknown)"
      [[ "${py_arch}" == "arm64" ]] || continue
    fi
    printf '%s' "${py}"
    return 0
  done
  return 1
}

python_runtime_id() {
  "${SYSTEM_PY}" -c 'import platform, sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}-{platform.machine()}")'
}

venv_deps_ok() {
  [[ -x "${PY}" ]] || return 1
  "${PY}" -c "
import pathlib
import numpy
import pandas
import plotly
import streamlit
static_index = pathlib.Path(streamlit.__file__).resolve().parent / 'static' / 'index.html'
raise SystemExit(0 if static_index.is_file() else 1)
" >>"${LOG_FILE}" 2>&1
}

app_import_ok() {
  (cd "${PROJECT_ROOT}" && "${PY}" -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('src').resolve()))
import resistance_viewer.app  # noqa: F401
") >>"${LOG_FILE}" 2>&1
}

install_deps() {
  notify "Resistance Viewer" "Installing dependencies…"
  log "Installing requirements from ${PROJECT_ROOT}/requirements.txt (python=${SYSTEM_PY})"
  "${PY}" -m pip install -q --upgrade pip >>"${LOG_FILE}" 2>&1
  "${PY}" -m pip install -q --force-reinstall --no-cache-dir -r "${PROJECT_ROOT}/requirements.txt" >>"${LOG_FILE}" 2>&1
  printf '%s' "${REQ_HASH}" >"${STAMP_FILE}"
}

create_venv() {
  notify "Resistance Viewer" "First run: creating virtual environment…"
  log "Creating venv at ${VENV} with ${SYSTEM_PY}"
  rm -rf "${VENV}"
  "${SYSTEM_PY}" -m venv "${VENV}" >>"${LOG_FILE}" 2>&1
}

ensure_venv_and_deps() {
  if [[ ! -x "${PY}" ]]; then
    create_venv
  fi

  local needs_install=0
  if ! venv_deps_ok; then
    log "Broken or incomplete install in venv (repairing)"
    needs_install=1
  elif [[ ! -f "${STAMP_FILE}" ]] || [[ "$(cat "${STAMP_FILE}")" != "${REQ_HASH}" ]]; then
    log "requirements.txt changed since last install (reinstalling)"
    needs_install=1
  fi

  if (( needs_install )); then
    install_deps
  fi

  if ! venv_deps_ok; then
    log "Venv still broken after pip install; recreating venv"
    create_venv
    install_deps
  fi

  if ! venv_deps_ok || ! app_import_ok; then
    log "Could not repair venv at ${VENV}"
    alert "Dependencies could not be installed. See ${LOG_FILE} or run: ./scripts/repair_viewer.sh"
    exit 1
  fi
}

if [[ -z "${PROJECT_ROOT:-}" ]] || [[ ! -f "${PROJECT_ROOT}/requirements.txt" ]]; then
  alert "Could not find the project folder (missing requirements.txt). Keep Resistance Viewer.app in the project folder, or select the folder when prompted."
  exit 1
fi

if ! SYSTEM_PY="$(select_python3)"; then
  alert "Python 3.10+ is required. Install from python.org or Homebrew, then try again."
  exit 1
fi

mkdir -p "${LOG_DIR}" "${CONFIG_DIR}"
log "Launch requested (project=${PROJECT_ROOT}, python=${SYSTEM_PY})"

# venv key includes project path + Python version/arch so Finder vs Terminal never share a broken mix.
VENV_KEY="$(printf '%s|%s' "${PROJECT_ROOT}" "$(python_runtime_id)" | /usr/bin/shasum -a 256 | awk '{print $1}' | cut -c1-16)"
VENV="${CONFIG_DIR}/venvs/${VENV_KEY}"
PY="${VENV}/bin/python"
STAMP_FILE="${CONFIG_DIR}/venvs/${VENV_KEY}.requirements.sha256"
REQ_HASH="$(/usr/bin/shasum -a 256 "${PROJECT_ROOT}/requirements.txt" | awk '{print $1}')"

ensure_venv_and_deps

if server_listening; then
  if server_healthy && app_import_ok; then
    log "Server already healthy on port ${PORT}; opening browser"
    open_browser
    exit 0
  fi
  log "Server on port ${PORT} is stale or app failed to load; restarting"
  stop_listening_server
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
  if server_healthy && app_import_ok; then
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
