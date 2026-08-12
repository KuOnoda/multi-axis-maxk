#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 {race5|gender2|race5_gender2} [maxk|max1|count]" >&2
  exit 2
fi

AXIS="$1"
METHOD="${2:-maxk}"
case "${AXIS}" in
  race5) REWARD_KEYS="clip_prompt_basis_race5_v4"; PAPER_K=5 ;;
  gender2) REWARD_KEYS="clip_prompt_basis_gender2_ens"; PAPER_K=2 ;;
  race5_gender2) REWARD_KEYS="clip_prompt_basis_race5_v4_gender2_ens"; PAPER_K=10 ;;
  *) echo "unknown axis: ${AXIS}" >&2; exit 2 ;;
esac

case "${METHOD}" in
  maxk) ADVANTAGE_MODE=coverage; K="${PAPER_K}" ;;
  max1) ADVANTAGE_MODE=coverage; K=1 ;;
  count)
    ADVANTAGE_MODE=count
    K="${PAPER_K}"
    ;;
  *) echo "unknown method: ${METHOD}" >&2; exit 2 ;;
esac

export STOCHASTIC_REWARD_KEYS="${REWARD_KEYS}"
export CLIP_PROMPT_REWARD_MODE=prob
export CLIP_PROMPT_TEMPERATURE=1.0
export MAXATK_K="${K}"
export NUM_IMAGE_PER_PROMPT=16
export ADVANTAGE_MODE
export TRAIN_BETA="${TRAIN_BETA:-0.05}"
export TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-1e-4}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export STOP_AT_GLOBAL_STEP="${STOP_AT_GLOBAL_STEP:-360}"
export SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-120}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export FORCED_RUN_NAME="${FORCED_RUN_NAME:-${AXIS}_${METHOD}_seed${TRAIN_SEED}}"

# Reported: 32 processes and local sampling batch 8. One-GPU functional run:
# NUM_PROCESSES=1 SAMPLE_BATCH_SIZE=16 TRAIN_BATCH_SIZE=16 (not compute-matched).
NUM_PROCESSES="${NUM_PROCESSES:-32}"
if [[ "${NUM_PROCESSES}" -eq 1 ]]; then
  DEFAULT_ACCELERATE_CONFIG=configs/accelerate/single_gpu.yaml
else
  DEFAULT_ACCELERATE_CONFIG=configs/accelerate/multi_gpu.yaml
fi
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${DEFAULT_ACCELERATE_CONFIG}}"

accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${NUM_PROCESSES}" \
  training/train_sd3.py \
  --config=configs/sd35_paper.py:paper
