#!/bin/bash
# Copy app sources into the .app bundle, fix permissions, quarantine, and sign.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
APP="${ROOT}/Resistance Viewer.app"
RESOURCES="${APP}/Contents/Resources"
BUNDLE_APP="${RESOURCES}/app"
MACOS="${APP}/Contents/MacOS"

if [[ ! -d "${APP}" ]]; then
  echo "Missing ${APP}" >&2
  exit 1
fi

mkdir -p "${RESOURCES}" "${MACOS}" "${BUNDLE_APP}"
rm -rf "${BUNDLE_APP:?}"/*
cp "${ROOT}/requirements.txt" "${BUNDLE_APP}/"
cp -R "${ROOT}/src" "${BUNDLE_APP}/"
find "${BUNDLE_APP}" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

chmod +x "${SCRIPT_DIR}/launch_viewer.sh" "${SCRIPT_DIR}/register_app.sh"
chmod +x "${MACOS}/Resistance Viewer" "${RESOURCES}/launch.sh" "${RESOURCES}/launch_viewer.sh"

xattr -cr "${APP}" 2>/dev/null || true

if command -v codesign >/dev/null 2>&1; then
  codesign --force --deep -s - "${APP}" 2>/dev/null || true
fi

echo "Bundled app into:"
echo "  ${BUNDLE_APP}"
echo ""
echo "Double-click:"
echo "  ${APP}"
echo ""
echo "If Finder does nothing the first time: right-click → Open."
echo "Logs: ~/Library/Logs/Resistance Viewer/"
