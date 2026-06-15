#!/usr/bin/env bash
set -euo pipefail

SESSION_ID="${SESSION_ID:-$(date +%Y%m%d_%H%M%S)}"
ROOT_DIR="${ROOT_DIR:-outputs/olp_re10k/overnight/${SESSION_ID}}"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export OLP_SEED="${OLP_SEED:-0}"

START_STAGE="${START_STAGE:-0}"
END_STAGE="${END_STAGE:-6}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-1}"
FULL_BATCH_SIZE="${BATCH_SIZE:-8}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
SMOKE_ACCUMULATE_GRAD_BATCHES="${SMOKE_ACCUMULATE_GRAD_BATCHES:-1}"
MAX_STEPS="${MAX_STEPS:-18751}"
LR="${LR:-2e-4}"
CKPT_EVERY_N_STEPS="${CKPT_EVERY_N_STEPS:-5000}"
OLP_WARMUP_STEPS="${OLP_WARMUP_STEPS:-5000}"
FINAL_SAVE_IMAGE="${FINAL_SAVE_IMAGE:-true}"

BASELINE_RUN_DIR="outputs/olp_re10k/re10k_ft_control/${SESSION_ID}"
OLP_RUN_DIR="outputs/olp_re10k/re10k_olp_ft/${SESSION_ID}"
EXTERNAL_BASELINE_CKPT="${BASELINE_CKPT:-}"
EXTERNAL_OLP_CKPT="${OLP_CKPT:-}"
BASELINE_CKPT_PATH="${BASELINE_CKPT:-${BASELINE_RUN_DIR}/checkpoints/last.ckpt}"
OLP_CKPT_PATH="${OLP_CKPT:-${OLP_RUN_DIR}/checkpoints/last.ckpt}"
BASELINE_EVAL_NAME="${BASELINE_EVAL_NAME:-re10k_ft_control_eval}"
OLP_EVAL_NAME="${OLP_EVAL_NAME:-re10k_olp_ft_eval}"

MANIFEST="${ROOT_DIR}/manifest.txt"

log_manifest() {
  printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${MANIFEST}"
}

run_step() {
  local name="$1"
  shift
  local log_path="${LOG_DIR}/${name}.log"

  log_manifest "START ${name}"
  log_manifest "LOG ${log_path}"
  {
    echo "### ${name}"
    echo "### started_at=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "### command=$*"
    "$@"
    echo "### finished_at=$(date '+%Y-%m-%d %H:%M:%S')"
  } 2>&1 | tee "${log_path}"
  log_manifest "DONE ${name}"
}

should_run_stage() {
  local stage="$1"
  [[ "${stage}" -ge "${START_STAGE}" && "${stage}" -le "${END_STAGE}" ]]
}

skip_stage() {
  local name="$1"
  log_manifest "SKIP ${name}"
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    log_manifest "MISSING ${path}"
    echo "Required file does not exist: ${path}" >&2
    exit 1
  fi
}

cat > "${ROOT_DIR}/run_config.env" <<EOF
SESSION_ID=${SESSION_ID}
ROOT_DIR=${ROOT_DIR}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
WANDB_MODE=${WANDB_MODE}
OLP_SEED=${OLP_SEED}
START_STAGE=${START_STAGE}
END_STAGE=${END_STAGE}
SMOKE_BATCH_SIZE=${SMOKE_BATCH_SIZE}
FULL_BATCH_SIZE=${FULL_BATCH_SIZE}
ACCUMULATE_GRAD_BATCHES=${ACCUMULATE_GRAD_BATCHES}
SMOKE_ACCUMULATE_GRAD_BATCHES=${SMOKE_ACCUMULATE_GRAD_BATCHES}
MAX_STEPS=${MAX_STEPS}
LR=${LR}
CKPT_EVERY_N_STEPS=${CKPT_EVERY_N_STEPS}
OLP_WARMUP_STEPS=${OLP_WARMUP_STEPS}
FINAL_SAVE_IMAGE=${FINAL_SAVE_IMAGE}
BASELINE_RUN_DIR=${BASELINE_RUN_DIR}
OLP_RUN_DIR=${OLP_RUN_DIR}
EXTERNAL_BASELINE_CKPT=${EXTERNAL_BASELINE_CKPT}
EXTERNAL_OLP_CKPT=${EXTERNAL_OLP_CKPT}
BASELINE_CKPT_PATH=${BASELINE_CKPT_PATH}
OLP_CKPT_PATH=${OLP_CKPT_PATH}
BASELINE_EVAL_NAME=${BASELINE_EVAL_NAME}
OLP_EVAL_NAME=${OLP_EVAL_NAME}
EOF

log_manifest "OLP-H overnight protocol started"
log_manifest "SESSION_ID=${SESSION_ID}"
log_manifest "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log_manifest "START_STAGE=${START_STAGE}"
log_manifest "END_STAGE=${END_STAGE}"
log_manifest "FULL_BATCH_SIZE=${FULL_BATCH_SIZE}"
log_manifest "ACCUMULATE_GRAD_BATCHES=${ACCUMULATE_GRAD_BATCHES}"
log_manifest "MAX_STEPS=${MAX_STEPS}"
if [[ -n "${EXTERNAL_BASELINE_CKPT}" ]]; then
  log_manifest "EXTERNAL_BASELINE_CKPT=${EXTERNAL_BASELINE_CKPT}"
fi
if [[ -n "${EXTERNAL_OLP_CKPT}" ]]; then
  log_manifest "EXTERNAL_OLP_CKPT=${EXTERNAL_OLP_CKPT}"
fi

if should_run_stage 0; then
  run_step "00_env_check" \
    python scripts/check_cuda118_env.py --output-dir "${ROOT_DIR}/env_check"
else
  skip_stage "00_env_check"
fi

if should_run_stage 1; then
  run_step "01_smoke" \
    env \
      RUN_ID="${SESSION_ID}" \
      SMOKE_ROOT="outputs/olp_re10k/smoke/${SESSION_ID}" \
      BATCH_SIZE="${SMOKE_BATCH_SIZE}" \
      ACCUMULATE_GRAD_BATCHES="${SMOKE_ACCUMULATE_GRAD_BATCHES}" \
      MAX_STEPS=2 \
      CKPT_EVERY_N_STEPS=1 \
      WANDB_MODE="${WANDB_MODE}" \
      OLP_SEED="${OLP_SEED}" \
      scripts/run_re10k_smoke.sh
else
  skip_stage "01_smoke"
fi

if should_run_stage 2; then
  run_step "02_original_eval" \
    env \
      RUN_ID="${SESSION_ID}" \
      WANDB_MODE="${WANDB_MODE}" \
      OLP_SEED="${OLP_SEED}" \
      scripts/run_re10k_original_eval.sh
else
  skip_stage "02_original_eval"
fi

if should_run_stage 3; then
  run_step "03_olp_zeroshot" \
    env \
      RUN_ID="${SESSION_ID}" \
      WANDB_MODE="${WANDB_MODE}" \
      OLP_SEED="${OLP_SEED}" \
      scripts/run_re10k_olp_zeroshot.sh
else
  skip_stage "03_olp_zeroshot"
fi

if [[ -n "${EXTERNAL_BASELINE_CKPT}" ]]; then
  require_file "${BASELINE_CKPT_PATH}"
  log_manifest "USE_EXISTING_BASELINE_CKPT=${BASELINE_CKPT_PATH}"
  skip_stage "04_ft_control"
elif should_run_stage 4; then
  run_step "04_ft_control" \
    env \
      RUN_ID="${SESSION_ID}" \
      RUN_DIR="${BASELINE_RUN_DIR}" \
      BATCH_SIZE="${FULL_BATCH_SIZE}" \
      ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES}" \
      MAX_STEPS="${MAX_STEPS}" \
      LR="${LR}" \
      CKPT_EVERY_N_STEPS="${CKPT_EVERY_N_STEPS}" \
      WANDB_MODE="${WANDB_MODE}" \
      OLP_SEED="${OLP_SEED}" \
      scripts/run_re10k_ft_control.sh

  require_file "${BASELINE_CKPT_PATH}"
  log_manifest "BASELINE_CKPT=${BASELINE_CKPT_PATH}"
else
  skip_stage "04_ft_control"
fi

if [[ -n "${EXTERNAL_OLP_CKPT}" ]]; then
  require_file "${OLP_CKPT_PATH}"
  log_manifest "USE_EXISTING_OLP_CKPT=${OLP_CKPT_PATH}"
  skip_stage "05_olp_ft"
elif should_run_stage 5; then
  run_step "05_olp_ft" \
    env \
      RUN_ID="${SESSION_ID}" \
      RUN_DIR="${OLP_RUN_DIR}" \
      BATCH_SIZE="${FULL_BATCH_SIZE}" \
      ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES}" \
      MAX_STEPS="${MAX_STEPS}" \
      LR="${LR}" \
      CKPT_EVERY_N_STEPS="${CKPT_EVERY_N_STEPS}" \
      OLP_WARMUP_STEPS="${OLP_WARMUP_STEPS}" \
      WANDB_MODE="${WANDB_MODE}" \
      OLP_SEED="${OLP_SEED}" \
      scripts/run_re10k_olp_ft.sh

  require_file "${OLP_CKPT_PATH}"
  log_manifest "OLP_CKPT=${OLP_CKPT_PATH}"
else
  skip_stage "05_olp_ft"
fi

if should_run_stage 6; then
  require_file "${BASELINE_CKPT_PATH}"
  require_file "${OLP_CKPT_PATH}"
  run_step "06_collect_stats" \
    env \
      RUN_ID="${SESSION_ID}" \
      ROOT_DIR="outputs/olp_re10k/collect_stats/${SESSION_ID}" \
      BASELINE_CKPT="${BASELINE_CKPT_PATH}" \
      OLP_CKPT="${OLP_CKPT_PATH}" \
      BASELINE_EVAL_NAME="${BASELINE_EVAL_NAME}" \
      OLP_EVAL_NAME="${OLP_EVAL_NAME}" \
      SAVE_IMAGE="${FINAL_SAVE_IMAGE}" \
      WANDB_MODE="${WANDB_MODE}" \
      OLP_SEED="${OLP_SEED}" \
      scripts/run_re10k_collect_stats.sh
else
  skip_stage "06_collect_stats"
fi

log_manifest "OLP-H overnight protocol completed"
log_manifest "Final baseline stats: outputs/olp_re10k/${BASELINE_EVAL_NAME}/olp_stats_summary.json"
log_manifest "Final OLP stats: outputs/olp_re10k/${OLP_EVAL_NAME}/olp_stats_summary.json"
