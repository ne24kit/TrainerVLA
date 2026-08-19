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

N_EPISODES=10
FPS=20

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


VIDEO_DIR="$LOG_DIR/videos/libero_goal_"$TASK_ID""


PYTHON="./.venv/bin/python"

MODEL="$PWD/models/Robometer-4B-LIBERO"

INFER_SCRIPT="./robometer/scripts/example_inference_local.py"


case "$TASK_ID" in
  0)
    TASK="open the middle drawer of the cabinet"
    ;;
  1)
    TASK="put the bowl on the stove"
    ;;
  2)
    TASK="put the wine bottle on top of the cabinet"
    ;;
esac


if [[ ! -d "$VIDEO_DIR" ]]; then
  echo "Video directory not found:" >&2
  echo "  $VIDEO_DIR" >&2
  exit 1
fi


if [[ ! -d "$MODEL" ]]; then
  echo "Robometer model not found:" >&2
  echo "  $MODEL" >&2
  exit 1
fi


if [[ ! -f "$INFER_SCRIPT" ]]; then
  echo "Inference script not found:" >&2
  echo "  $INFER_SCRIPT" >&2
  exit 1
fi


# Чтобы import robometer работал из локального репозитория
export PYTHONPATH="$PWD/robometer:${PYTHONPATH:-}"


echo "========================================"
echo "Robometer evaluation"
echo "========================================"
echo "RUN=$RUN"
echo "TASK_ID=$TASK_ID"
echo "DEMOS=$DEMOS"
echo "SEED=$SEED"
echo "N_EPISODES=$N_EPISODES"
echo "TASK=$TASK"
echo "FPS=$FPS"
echo "VIDEO_DIR=$VIDEO_DIR"
echo "MODEL=$MODEL"
echo "========================================"


for EPISODE_ID in $(seq 0 $((N_EPISODES - 1))); do

  VIDEO="$VIDEO_DIR/eval_episode_${EPISODE_ID}.mp4"

  if [[ ! -f "$VIDEO" ]]; then
    echo
    echo "[SKIP] Episode $EPISODE_ID"
    echo "Video not found:"
    echo "  $VIDEO"
    continue
  fi


  #
  # ВАЖНО:
  # сохраняем rewards прямо рядом с MP4.
  #
  # example_inference_local.py сам создаст рядом:
  #
  # eval_episode_0_rewards.npy
  # eval_episode_0_rewards_success_probs.npy
  # eval_episode_0_rewards_progress_success.png
  #

  OUT="$VIDEO_DIR/eval_episode_${EPISODE_ID}_rewards.npy"


  echo
  echo "----------------------------------------"
  echo "Episode $EPISODE_ID / $((N_EPISODES - 1))"
  echo "----------------------------------------"
  echo "VIDEO=$VIDEO"
  echo "OUT=$OUT"


  "$PYTHON" "$INFER_SCRIPT" \
    --model-path "$MODEL" \
    --video "$VIDEO" \
    --task "$TASK" \
    --fps "$FPS" \
    --out "$OUT"\
    --max-frames 8 \

done


echo
echo "========================================"
echo "Robometer evaluation finished"
echo "========================================"
echo "Results are in:"
echo "$VIDEO_DIR"