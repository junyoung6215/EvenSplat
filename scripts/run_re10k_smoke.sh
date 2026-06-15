#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-smoke_$(date +%Y%m%d_%H%M%S)}"
SMOKE_ROOT="${SMOKE_ROOT:-outputs/olp_re10k/smoke/${RUN_ID}}"

# Use this before full GPU runs. It verifies checkpoint loading, CUDA kernels,
# OLP-H forward/backward, checkpoint writing, and the command/fingerprint logs
# with a small memory footprint.
COMMON_ENV=(
  "RUN_ID=${RUN_ID}"
  "MAX_STEPS=${MAX_STEPS:-2}"
  "BATCH_SIZE=${BATCH_SIZE:-1}"
  "ACCUMULATE_GRAD_BATCHES=${ACCUMULATE_GRAD_BATCHES:-1}"
  "CKPT_EVERY_N_STEPS=${CKPT_EVERY_N_STEPS:-1}"
  "NUM_SANITY_VAL_STEPS=${NUM_SANITY_VAL_STEPS:-0}"
  "VAL_CHECK_INTERVAL=${VAL_CHECK_INTERVAL:-100000000}"
  "TRAIN_NUM_WORKERS=${TRAIN_NUM_WORKERS:-0}"
  "VAL_NUM_WORKERS=${VAL_NUM_WORKERS:-0}"
  "TEST_NUM_WORKERS=${TEST_NUM_WORKERS:-0}"
  "WANDB_MODE=${WANDB_MODE:-disabled}"
  "OLP_SEED=${OLP_SEED:-0}"
)

env "${COMMON_ENV[@]}" \
  RUN_DIR="${SMOKE_ROOT}/re10k_ft_control" \
  scripts/run_re10k_ft_control.sh

env "${COMMON_ENV[@]}" \
  RUN_DIR="${SMOKE_ROOT}/re10k_olp_ft" \
  scripts/run_re10k_olp_ft.sh
