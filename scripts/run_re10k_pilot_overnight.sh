#!/usr/bin/env bash
set -euo pipefail

# Time-limited pilot protocol for next-day reporting.
#
# This uses the official pretrained re10k.ckpt as the baseline checkpoint and
# skips the baseline FT control stage. The result should be reported as a pilot:
# official NoPoSplat vs OLP-H adapted NoPoSplat, not as a same-budget FT control.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export BASELINE_CKPT="${BASELINE_CKPT:-./pretrained_weights/re10k.ckpt}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
export MAX_STEPS="${MAX_STEPS:-18751}"
export LR="${LR:-2e-4}"
export OLP_SEED="${OLP_SEED:-0}"
export FINAL_SAVE_IMAGE="${FINAL_SAVE_IMAGE:-true}"
export BASELINE_EVAL_NAME="${BASELINE_EVAL_NAME:-re10k_official_eval_stats}"
export OLP_EVAL_NAME="${OLP_EVAL_NAME:-re10k_olp_ft_eval}"

if [[ "${DOWNLOAD_RE10K:-1}" == "1" ]]; then
  CHECKPOINT_PATH="${BASELINE_CKPT}" scripts/download_re10k_checkpoint.sh
fi

exec scripts/run_re10k_overnight_protocol.sh
