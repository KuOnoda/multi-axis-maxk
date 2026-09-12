#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "usage: $0 {race5|gender2|race5_gender2|color7} [maxk|max1|count]"
  echo "Launch: NUM_PROCESSES, NUM_MACHINES, MACHINE_RANK, MAIN_PROCESS_IP, MAIN_PROCESS_PORT"
  echo "Set DRY_RUN=1 to print the command without starting training."
  exit 0
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 {race5|gender2|race5_gender2|color7} [maxk|max1|count]" >&2
  exit 2
fi

AXIS="$1"
METHOD="${2:-maxk}"
RECIPE=paper
case "${AXIS}" in
  race5) REWARD_KEYS="clip_prompt_basis_race5_v4"; PAPER_K=5 ;;
  gender2) REWARD_KEYS="clip_prompt_basis_gender2_ens"; PAPER_K=2 ;;
  race5_gender2) REWARD_KEYS="clip_prompt_basis_race5_v4_gender2_ens"; PAPER_K=10 ;;
  color7) REWARD_KEYS="pixel_partition_basis"; PAPER_K=7; RECIPE=color7 ;;
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
export MAXATK_K="${K}"
export ADVANTAGE_MODE
export TRAIN_SEED="${TRAIN_SEED:-42}"
export SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-120}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export FORCED_RUN_NAME="${FORCED_RUN_NAME:-${AXIS}_${METHOD}_seed${TRAIN_SEED}}"

if [[ "${RECIPE}" == "color7" ]]; then
  # Controlled rule-based color experiment (paper Sec. 5.2): m = k = 7, every
  # axis multiplied by frozen PickScore, KL beta 0, GenEval single-object
  # prompts, one GPU. Set REWARD_ANCHOR_KEY=none for the pure rule-reward arm.
  export NUM_IMAGE_PER_PROMPT="${NUM_IMAGE_PER_PROMPT:-7}"
  export REWARD_ANCHOR_KEY="${REWARD_ANCHOR_KEY:-pickscore}"
  export MAXK_M_EQ_K_BASELINE="${MAXK_M_EQ_K_BASELINE:-group_mean}"
  export TRAIN_BETA="${TRAIN_BETA:-0.0}"
  export TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-3e-4}"
  export STOP_AT_GLOBAL_STEP="${STOP_AT_GLOBAL_STEP:-960}"
  NUM_PROCESSES="${NUM_PROCESSES:-1}"
  SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-${NUM_IMAGE_PER_PROMPT}}"
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${NUM_IMAGE_PER_PROMPT}}"
else
  export CLIP_PROMPT_REWARD_MODE=prob
  export CLIP_PROMPT_TEMPERATURE=1.0
  export NUM_IMAGE_PER_PROMPT=16
  export REWARD_ANCHOR_KEY="${REWARD_ANCHOR_KEY:-none}"
  export TRAIN_BETA="${TRAIN_BETA:-0.05}"
  export TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-1e-4}"
  export STOP_AT_GLOBAL_STEP="${STOP_AT_GLOBAL_STEP:-360}"
  # Reported: 32 processes and local sampling batch 8. One-GPU functional run:
  # NUM_PROCESSES=1 SAMPLE_BATCH_SIZE=16 TRAIN_BATCH_SIZE=16 (not compute-matched).
  NUM_PROCESSES="${NUM_PROCESSES:-32}"
  SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
fi
NUM_MACHINES="${NUM_MACHINES:-1}"
MACHINE_RANK="${MACHINE_RANK:-0}"
for value in "${NUM_PROCESSES}" "${NUM_MACHINES}" "${SAMPLE_BATCH_SIZE}" "${TRAIN_BATCH_SIZE}"; do
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "process, machine, and batch counts must be positive integers" >&2
    exit 2
  fi
done
if [[ ! "${MACHINE_RANK}" =~ ^(0|[1-9][0-9]*)$ ]] || (( MACHINE_RANK >= NUM_MACHINES )); then
  echo "MACHINE_RANK must be between 0 and NUM_MACHINES-1" >&2
  exit 2
fi
if (( NUM_PROCESSES % NUM_MACHINES != 0 )); then
  echo "NUM_PROCESSES must be divisible by NUM_MACHINES" >&2
  exit 2
fi
if (( NUM_PROCESSES * SAMPLE_BATCH_SIZE % NUM_IMAGE_PER_PROMPT != 0 )); then
  echo "global rollout batch must be divisible by NUM_IMAGE_PER_PROMPT=${NUM_IMAGE_PER_PROMPT}" >&2
  exit 2
fi
if (( SAMPLE_BATCH_SIZE != TRAIN_BATCH_SIZE )); then
  echo "the paper recipe requires equal SAMPLE_BATCH_SIZE and TRAIN_BATCH_SIZE" >&2
  exit 2
fi
export SAMPLE_BATCH_SIZE TRAIN_BATCH_SIZE

if [[ "${NUM_PROCESSES}" -eq 1 ]]; then
  DEFAULT_ACCELERATE_CONFIG=configs/accelerate/single_gpu.yaml
else
  DEFAULT_ACCELERATE_CONFIG=configs/accelerate/multi_gpu.yaml
fi
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${DEFAULT_ACCELERATE_CONFIG}}"

launch=(accelerate launch
  --config_file "${ACCELERATE_CONFIG}"
  --num_processes "${NUM_PROCESSES}"
  --num_machines "${NUM_MACHINES}"
  --machine_rank "${MACHINE_RANK}"
  --main_process_port "${MAIN_PROCESS_PORT:-29500}"
)
if (( NUM_MACHINES > 1 )); then
  if [[ -z "${MAIN_PROCESS_IP:-}" ]]; then
    echo "MAIN_PROCESS_IP must be set for multi-node training" >&2
    exit 2
  fi
  launch+=(--main_process_ip "${MAIN_PROCESS_IP}")
fi
launch+=(training/train_sd3.py "--config=configs/sd35_paper.py:${RECIPE}")

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'axis=%s method=%s k=%s seed=%s run=%s\n' \
    "${AXIS}" "${METHOD}" "${K}" "${TRAIN_SEED}" "${FORCED_RUN_NAME}"
  printf '%q ' "${launch[@]}"
  printf '\n'
  exit 0
fi
exec "${launch[@]}"
