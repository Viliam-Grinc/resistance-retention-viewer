#!/bin/bash
# macOS .app entry (no Terminal): use app code inside the bundle (sandbox-safe).
set -euo pipefail

on_error() {
  /usr/bin/osascript -e "display alert \"Resistance Viewer\" message \"Launch failed. See ${HOME}/Library/Logs/Resistance Viewer/app-launch.log and viewer.log.\"" 2>/dev/null || true
}
trap on_error ERR

RESOURCES="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bundled_app_root() {
  if [[ -f "${RESOURCES}/app/requirements.txt" && -f "${RESOURCES}/app/src/resistance_viewer/app.py" ]]; then
    printf '%s' "${RESOURCES}/app"
    return 0
  fi
  return 1
}

if ! PROJECT_ROOT="$(bundled_app_root)"; then
  /usr/bin/osascript -e 'display alert "Resistance Viewer" message "App files are missing inside the bundle. From Terminal, run: ./scripts/register_app.sh"' >/dev/null 2>&1 || true
  exit 1
fi

export PROJECT_ROOT
exec "${RESOURCES}/launch_viewer.sh"
