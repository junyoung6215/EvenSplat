#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-re10k_olp_ft_7k_bs4}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ROOT_DIR="${ROOT_DIR:-/home/data/olp_h_re10k_slot0/noposplat_runs/olp_re10k/${RUN_NAME}/${RUN_ID}}"
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-${ROOT_DIR}/train}"
EVAL_RUN_NAME="${EVAL_RUN_NAME:-${RUN_NAME}_eval}"
EVAL_RUN_DIR="${EVAL_RUN_DIR:-${ROOT_DIR}/eval}"
TEST_OUTPUT_PATH="${TEST_OUTPUT_PATH:-${ROOT_DIR}/test_outputs}"
DATA_ROOT="${DATA_ROOT:-datasets/re10k}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if command -v python >/dev/null 2>&1; then
  TORCH_LIB_DIR="$(
    python - <<'PY'
from pathlib import Path

try:
    import torch
except Exception:
    raise SystemExit(0)

print(Path(torch.__file__).resolve().parent / "lib")
PY
  )"
  if [[ -n "${TORCH_LIB_DIR}" && -d "${TORCH_LIB_DIR}" ]]; then
    export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${LD_LIBRARY_PATH:-}"
  fi
fi

MAX_STEPS="${MAX_STEPS:-7000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-2e-4}"
OLP_WARMUP_STEPS="${OLP_WARMUP_STEPS:-1500}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-3000}"
CKPT_EVERY_N_STEPS="${CKPT_EVERY_N_STEPS:-1000}"
SAVE_TOP_K="${SAVE_TOP_K:-1}"
SAVE_WEIGHTS_ONLY="${SAVE_WEIGHTS_ONLY:-false}"
PRINT_LOG_EVERY_N_STEPS="${PRINT_LOG_EVERY_N_STEPS:-100}"
RUN_EVAL="${RUN_EVAL:-true}"
SAVE_IMAGE="${SAVE_IMAGE:-false}"
OLP_SEED="${OLP_SEED:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-30}"

check_free_space() {
  local path="$1"
  local min_gb="$2"
  local label="$3"
  local probe="${path}"

  while [[ ! -e "${probe}" && "${probe}" != "/" ]]; do
    probe="$(dirname "${probe}")"
  done

  local available_kb
  available_kb="$(df -Pk "${probe}" | awk 'NR == 2 {print $4}')"
  local min_kb=$((min_gb * 1024 * 1024))
  if (( available_kb < min_kb )); then
    echo "Not enough free space for ${label}: ${path}" >&2
    echo "Available: $((available_kb / 1024 / 1024))GB, required: ${min_gb}GB" >&2
    exit 1
  fi
}

check_free_space "${ROOT_DIR}" "${MIN_FREE_GB}" "training outputs"
if [[ "${RUN_EVAL}" == "true" ]]; then
  check_free_space "${TEST_OUTPUT_PATH}" "${MIN_FREE_GB}" "test outputs"
fi

mkdir -p "${TRAIN_RUN_DIR}/fingerprint" "${EVAL_RUN_DIR}" "${LOG_DIR}"

git rev-parse HEAD > "${TRAIN_RUN_DIR}/fingerprint/git_commit.txt"
echo "${OLP_SEED}" > "${TRAIN_RUN_DIR}/fingerprint/olp_seed.txt"
cat > "${ROOT_DIR}/run_config.env" <<EOF
RUN_NAME=${RUN_NAME}
RUN_ID=${RUN_ID}
ROOT_DIR=${ROOT_DIR}
TRAIN_RUN_DIR=${TRAIN_RUN_DIR}
EVAL_RUN_NAME=${EVAL_RUN_NAME}
EVAL_RUN_DIR=${EVAL_RUN_DIR}
TEST_OUTPUT_PATH=${TEST_OUTPUT_PATH}
DATA_ROOT=${DATA_ROOT}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
WANDB_MODE=${WANDB_MODE}
MAX_STEPS=${MAX_STEPS}
BATCH_SIZE=${BATCH_SIZE}
LR=${LR}
OLP_WARMUP_STEPS=${OLP_WARMUP_STEPS}
VAL_CHECK_INTERVAL=${VAL_CHECK_INTERVAL}
CKPT_EVERY_N_STEPS=${CKPT_EVERY_N_STEPS}
SAVE_TOP_K=${SAVE_TOP_K}
SAVE_WEIGHTS_ONLY=${SAVE_WEIGHTS_ONLY}
PRINT_LOG_EVERY_N_STEPS=${PRINT_LOG_EVERY_N_STEPS}
RUN_EVAL=${RUN_EVAL}
SAVE_IMAGE=${SAVE_IMAGE}
OLP_SEED=${OLP_SEED}
MIN_FREE_GB=${MIN_FREE_GB}
EOF

if [[ "${CHECK_ENV:-1}" == "1" ]]; then
  python scripts/check_cuda118_env.py --output-dir "${TRAIN_RUN_DIR}/fingerprint"
fi

TRAIN_CMD=(
  python -u -m src.main
  +experiment=re10k
  mode=train
  "wandb.mode=${WANDB_MODE}"
  "wandb.name=${RUN_NAME}"
  model.encoder.pretrained_weights=./pretrained_weights/re10k.ckpt
  checkpointing.load=null
  "dataset.re10k.roots=[${DATA_ROOT}]"
  model.encoder.orthogonal_preconditioner.enabled=true
  model.encoder.orthogonal_preconditioner.mode=hadamard
  model.encoder.orthogonal_preconditioner.dim=256
  model.encoder.orthogonal_preconditioner.mix=0.0
  model.encoder.orthogonal_preconditioner.target_mix=1.0
  "model.encoder.orthogonal_preconditioner.warmup_steps=${OLP_WARMUP_STEPS}"
  "model.encoder.orthogonal_preconditioner.seed=${OLP_SEED}"
  model.encoder.orthogonal_preconditioner.collect_stats=true
  "trainer.max_steps=${MAX_STEPS}"
  "trainer.num_sanity_val_steps=${NUM_SANITY_VAL_STEPS:-0}"
  "trainer.val_check_interval=${VAL_CHECK_INTERVAL}"
  "optimizer.lr=${LR}"
  "data_loader.train.batch_size=${BATCH_SIZE}"
  "data_loader.train.num_workers=${TRAIN_NUM_WORKERS:-16}"
  "data_loader.val.num_workers=${VAL_NUM_WORKERS:-1}"
  "data_loader.test.num_workers=${TEST_NUM_WORKERS:-4}"
  "checkpointing.every_n_train_steps=${CKPT_EVERY_N_STEPS}"
  "checkpointing.save_top_k=${SAVE_TOP_K}"
  "checkpointing.save_weights_only=${SAVE_WEIGHTS_ONLY}"
  "train.print_log_every_n_steps=${PRINT_LOG_EVERY_N_STEPS}"
  "hydra.run.dir=${TRAIN_RUN_DIR}"
)

printf "%q " "${TRAIN_CMD[@]}" > "${TRAIN_RUN_DIR}/command.sh"
printf "\n" >> "${TRAIN_RUN_DIR}/command.sh"
"${TRAIN_CMD[@]}"

CKPT_PATH="${TRAIN_RUN_DIR}/checkpoints/last.ckpt"
if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Missing checkpoint: ${CKPT_PATH}" >&2
  exit 1
fi

if [[ "${RUN_EVAL}" != "true" ]]; then
  echo "Training finished. Checkpoint: ${CKPT_PATH}"
  exit 0
fi

EVAL_CMD=(
  python -u -m src.main
  +experiment=re10k
  mode=test
  "wandb.mode=${WANDB_MODE}"
  "wandb.name=${EVAL_RUN_NAME}"
  dataset/view_sampler@dataset.re10k.view_sampler=evaluation
  dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json
  "dataset.re10k.roots=[${DATA_ROOT}]"
  "checkpointing.load=${CKPT_PATH}"
  model.encoder.orthogonal_preconditioner.enabled=true
  model.encoder.orthogonal_preconditioner.mode=hadamard
  model.encoder.orthogonal_preconditioner.dim=256
  model.encoder.orthogonal_preconditioner.mix=1.0
  model.encoder.orthogonal_preconditioner.target_mix=1.0
  model.encoder.orthogonal_preconditioner.warmup_steps=0
  "model.encoder.orthogonal_preconditioner.seed=${OLP_SEED}"
  model.encoder.orthogonal_preconditioner.collect_stats=true
  "test.save_image=${SAVE_IMAGE}"
  test.save_olp_stats=true
  "test.olp_tau_alpha=${OLP_TAU_ALPHA:-0.95}"
  "test.olp_tau_scale=${OLP_TAU_SCALE:-null}"
  "test.olp_tau_volume=${OLP_TAU_VOLUME:-null}"
  "test.olp_adaptive_tau_scale_quantile=${OLP_ADAPTIVE_TAU_SCALE_QUANTILE:-0.95}"
  "test.output_path=${TEST_OUTPUT_PATH}"
  "hydra.run.dir=${EVAL_RUN_DIR}"
)

printf "%q " "${EVAL_CMD[@]}" > "${EVAL_RUN_DIR}/command.sh"
printf "\n" >> "${EVAL_RUN_DIR}/command.sh"
"${EVAL_CMD[@]}"
