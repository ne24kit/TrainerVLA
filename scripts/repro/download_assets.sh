#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-./.venv/bin/python}"

"$PYTHON" scripts/repro/download_assets.py
"$PYTHON" scripts/repro/apply_venv_patches.py
