#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-./.venv/bin/python}"
SRC="data/libero_goal/videos/observation.images.wrist_image/chunk-000/file-000.mp4"
BAK="$SRC.bak"
FIXED="$SRC.fixed.mp4"
EXPECTED_FRAMES=52042

if [[ ! -f "$SRC" ]]; then
  echo "Video not found: $SRC" >&2
  exit 1
fi

if [[ ! -f "$BAK" ]]; then
  cp "$SRC" "$BAK"
fi

ffmpeg -y -err_detect ignore_err \
  -i "$BAK" \
  -an -r 20 \
  -c:v libsvtav1 -pix_fmt yuv420p -crf 30 -preset 12 \
  "$FIXED"

frames="$("$PYTHON" -c 'import av, sys; c=av.open(sys.argv[1]); s=c.streams.video[0]; print(sum(1 for _ in c.decode(s)))' "$FIXED")"

if [[ "$frames" != "$EXPECTED_FRAMES" ]]; then
  echo "Unexpected frame count: $frames, expected $EXPECTED_FRAMES" >&2
  echo "Original video was not overwritten." >&2
  exit 1
fi

mv "$FIXED" "$SRC"

echo "Fixed video written: $SRC"
echo "Backup kept: $BAK"
