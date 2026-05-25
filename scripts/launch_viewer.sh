#!/bin/bash
# CLI wrapper — runs the launcher bundled inside Resistance Viewer.app.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-${REPO}}"
exec "${REPO}/Resistance Viewer.app/Contents/Resources/launch_viewer.sh"
