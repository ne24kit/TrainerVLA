#!/bin/bash

set -euo pipefail

usage() {
  echo "Usage: $0 TASK_ID DEMOS SEED"
  echo "  TASK_ID:    0, 1, or 2"
  echo "  DEMOS:      5, 10, or 25"
  echo "  SEED:       eval seed"
  echo "  Example: $0 0 25 1001"
}

if [[ $# -ne 3 ]]; then
  usage >&2
  exit 1
fi

TASK_ID="$1"
DEMOS="$2"
SEED="$3"

case "$TASK_ID" in
  0|1|2)
    ;;
  *)
    echo "Unknown TASK_ID: $TASK_ID. Expected 0, 1, or 2." >&2
    exit 1
    ;;
esac


case "$DEMOS" in
  0)
    RUN="smolvla_libero90_pretrain"
    POLICY_PATH="./outputs/$RUN/checkpoints/070000/pretrained_model"
    LOG_DIR="./logs_task_4/smolvla_libero90_t${TASK_ID}_d0/s_${SEED}"
    ;;
  5|10|25)
    RUN="smolvla_libero90_pretrained_finetune_t${TASK_ID}_d${DEMOS}"
    POLICY_PATH="./outputs/$RUN/checkpoints/005000/pretrained_model"
    LOG_DIR="./logs_task_4/smolvla_libero90_t${TASK_ID}_d${DEMOS}/s_${SEED}"
    ;;
  *)
    echo "Unknown DEMOS: $DEMOS. Expected 0, 5, 10, or 25." >&2
    exit 1
    ;;
esac


VIDEO_DIR="$LOG_DIR/videos/libero_goal_${TASK_ID}"



PYTHON="./.venv/bin/python"

INFER_SCRIPT="./scripts/task_4/eval_sarm_video.py"

SARM_MODEL="./outputs/sarm_libero90_single_stage_zero_state/checkpoints/015000/pretrained_model"


# --------------------------------------------------
# Checks
# --------------------------------------------------

if [[ ! -d "$VIDEO_DIR" ]]; then
  echo "Video directory not found:" >&2
  echo "  $VIDEO_DIR" >&2
  exit 1
fi


if [[ ! -d "$SARM_MODEL" ]]; then
  echo "SARM model not found:" >&2
  echo "  $SARM_MODEL" >&2
  exit 1
fi


if [[ ! -f "$INFER_SCRIPT" ]]; then
  echo "SARM inference script not found:" >&2
  echo "  $INFER_SCRIPT" >&2
  exit 1
fi


echo "========================================"
echo "SARM evaluation"
echo "========================================"
echo "RUN=$RUN"
echo "TASK_ID=$TASK_ID"
echo "DEMOS=$DEMOS"
echo "SEED=$SEED"
echo "VIDEO_DIR=$VIDEO_DIR"
echo "SARM_MODEL=$SARM_MODEL"
echo "========================================"


"$PYTHON" "$INFER_SCRIPT" \
  --reward-model-path "$SARM_MODEL" \
  --video-dir "$VIDEO_DIR" \
  --task-id "$TASK_ID" \
  --device cuda


echo
echo "========================================"
echo "SARM evaluation finished"
echo "========================================"
echo "Results are in:"
echo "$VIDEO_DIR"