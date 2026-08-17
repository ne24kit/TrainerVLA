#!/bin/bash
set -euo pipefail

usage() {
  echo "Usage: $0 TASK_ID DEMOS"
  echo "  TASK_ID: 0, 1, or 2"
  echo "  DEMOS:   5, 10, or 25"
  echo "  Example: $0 0 5"
  echo
  echo "Set FORCE=1 to rebuild an existing output dataset."
}

if [[ $# -ne 2 ]]; then
  usage >&2
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

TASK_ID="$1"
DEMOS="$2"
EDIT_CMD="${LEROBOT_EDIT_DATASET:-./.venv/bin/lerobot-edit-dataset}"

if [[ ! -x "$EDIT_CMD" ]]; then
  echo "lerobot-edit-dataset not found: $EDIT_CMD" >&2
  exit 1
fi

if ! command -v jq >/dev/null; then
  echo "jq not found. Install jq to align LeRobot metadata before merge." >&2
  exit 1
fi

case "$TASK_ID" in
  0)
    target_episodes=(20 26 31 42 58 64 84 90 94 111 117 118 137 140 146 162 168 173 182 187 198 206 220 232 252)
    ;;
  1)
    target_episodes=(13 15 16 22 36 45 66 76 116 121 145 151 165 166 171 178 179 186 201 219 225 233 237 239 250)
    ;;
  2)
    target_episodes=(6 10 17 18 25 38 40 44 48 51 53 57 75 89 91 96 97 100 101 103 133 136 149 154 164)
    ;;
  *)
    echo "Unknown TASK_ID: $TASK_ID. Expected 0, 1, or 2." >&2
    exit 1
    ;;
esac

case "$DEMOS" in
  5|10|25)
    ;;
  *)
    echo "Unknown DEMOS: $DEMOS. Expected 5, 10, or 25." >&2
    exit 1
    ;;
esac

# Deterministic LIBERO-90 replay. This is deliberately not hand-picked by success.
replay_episodes=(0 400 800 1200 1600 2000 2400 2800 3200 3600)
selected_target=("${target_episodes[@]:0:$DEMOS}")

TARGET_JSON="[${selected_target[*]}]"
TARGET_JSON="${TARGET_JSON// /, }"

REPLAY_JSON="[${replay_episodes[*]}]"
REPLAY_JSON="${REPLAY_JSON// /, }"

NAME="libero_goal_replay_t${TASK_ID}_d${DEMOS}"
OUT_DIR="./data/$NAME"
TMP_DIR="./data/_tmp_$NAME"

REPLAY_SPLIT_ROOT="$TMP_DIR/replay"
REPLAY_COMMON_ROOT="$TMP_DIR/replay_common"
TARGET_SPLIT_ROOT="$TMP_DIR/target"

if [[ -e "$OUT_DIR" ]]; then
  if [[ "${FORCE:-0}" == "1" ]]; then
    echo "FORCE=1: removing existing dataset $OUT_DIR"
    rm -rf "$OUT_DIR"
  else
    echo "Output dataset already exists: $OUT_DIR" >&2
    echo "Set FORCE=1 to rebuild it." >&2
    exit 1
  fi
fi

rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

echo "TASK_ID=$TASK_ID"
echo "DEMOS=$DEMOS"
echo "TARGET_EPISODES=$TARGET_JSON"
echo "REPLAY_EPISODES=$REPLAY_JSON"
echo "OUTPUT=$OUT_DIR"

echo "[1/5] Split LIBERO-90 replay subset"
"$EDIT_CMD" \
  --repo_id=local/libero_90 \
  --root=./data/libero_90 \
  --new_root="$TMP_DIR" \
  --operation.type=split \
  --operation.splits="{\"replay\": $REPLAY_JSON}"

echo "[2/5] Remove LIBERO-90 extra state features"
"$EDIT_CMD" \
  --repo_id=local/libero_90_replay \
  --root="$REPLAY_SPLIT_ROOT" \
  --new_repo_id=local/libero_90_replay_common \
  --new_root="$REPLAY_COMMON_ROOT" \
  --operation.type=remove_feature \
  --operation.feature_names="['observation.states.ee_state', 'observation.states.joint_state', 'observation.states.gripper_state']"

echo "[3/5] Align LIBERO-90 replay feature names for merge"
jq '
  .features."observation.state".names.motors = ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper", "gripper"] |
  .features.action.names.motors = ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper"]
' "$REPLAY_COMMON_ROOT/meta/info.json" > "$REPLAY_COMMON_ROOT/meta/info.json.tmp"
mv "$REPLAY_COMMON_ROOT/meta/info.json.tmp" "$REPLAY_COMMON_ROOT/meta/info.json"

echo "[4/5] Split LIBERO Goal target subset"
"$EDIT_CMD" \
  --repo_id=local/libero_goal \
  --root=./data/libero_goal \
  --new_root="$TMP_DIR" \
  --operation.type=split \
  --operation.splits="{\"target\": $TARGET_JSON}"

echo "[5/5] Merge replay + target into $OUT_DIR"
"$EDIT_CMD" \
  --new_repo_id="local/$NAME" \
  --new_root="$OUT_DIR" \
  --operation.type=merge \
  --operation.repo_ids="['local/libero_90_replay_common', 'local/libero_goal_target']" \
  --operation.roots="['$REPLAY_COMMON_ROOT', '$TARGET_SPLIT_ROOT']" \
  --operation.concatenate_videos=false \
  --operation.concatenate_data=false

echo "Done: $OUT_DIR"
