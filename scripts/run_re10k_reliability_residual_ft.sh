#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="re10k_olp_reliability_residual_ft"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-outputs/reliability_residual_ft/${RUN_NAME}_${RUN_ID}}"
mkdir -p "${RUN_DIR}/fingerprint"

# Default: original 300k micro-batch budget at BATCH_SIZE=4 converted to
# effective batch 16 via gradient accumulation.
BASE_CKPT="${BASE_CKPT:-/home/data/olp_h_re10k_slot0/noposplat_runs/re10k/re10k_olp_ft/full_20260511_300k_v2_resume122k/train/checkpoints/epoch_0-step_300000.ckpt}"
DATA_ROOT="${DATA_ROOT:-/home/data/olp_h_re10k_slot0/re10k_hf}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-4}"
MAX_STEPS="${MAX_STEPS:-75000}"
CKPT_EVERY_N_STEPS="${CKPT_EVERY_N_STEPS:-25000}"
LR="${LR:-2e-4}"
OLP_SEED="${OLP_SEED:-0}"
REL_WEIGHT="${REL_WEIGHT:-1.0e-4}"
REL_WARMUP_START="${REL_WARMUP_START:-3000}"
REL_WARMUP_END="${REL_WARMUP_END:-8000}"
WANDB_MODE="${WANDB_MODE:-disabled}"

if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "Missing BASE_CKPT: ${BASE_CKPT}" >&2
  exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Missing DATA_ROOT: ${DATA_ROOT}" >&2
  exit 1
fi

git rev-parse HEAD > "${RUN_DIR}/fingerprint/git_commit.txt"
printf "%s\n" "${BASE_CKPT}" > "${RUN_DIR}/fingerprint/base_ckpt.txt"
printf "%s\n" "${DATA_ROOT}" > "${RUN_DIR}/fingerprint/data_root.txt"
printf "%s\n" "${OLP_SEED}" > "${RUN_DIR}/fingerprint/olp_seed.txt"
printf "%s\n" "${BATCH_SIZE}" > "${RUN_DIR}/fingerprint/batch_size.txt"
printf "%s\n" "${ACCUMULATE_GRAD_BATCHES}" > "${RUN_DIR}/fingerprint/accumulate_grad_batches.txt"
printf "%s\n" "$(( BATCH_SIZE * ACCUMULATE_GRAD_BATCHES ))" > "${RUN_DIR}/fingerprint/effective_batch_size_single_gpu.txt"

if [[ "${CHECK_ENV:-0}" == "1" ]]; then
  python scripts/check_cuda118_env.py --output-dir "${RUN_DIR}/fingerprint"
fi

CMD=(
  python -m src.main
  +experiment=re10k
  mode=train
  "wandb.mode=${WANDB_MODE}"
  "wandb.name=${RUN_NAME}"
  "model.encoder.pretrained_weights=${BASE_CKPT}"
  checkpointing.load=null
  "dataset.re10k.roots=[${DATA_ROOT}]"
  model.encoder.orthogonal_preconditioner.enabled=true
  model.encoder.orthogonal_preconditioner.mode=hadamard
  model.encoder.orthogonal_preconditioner.dim=256
  model.encoder.orthogonal_preconditioner.mix=1.0
  model.encoder.orthogonal_preconditioner.target_mix=1.0
  model.encoder.orthogonal_preconditioner.warmup_steps=0
  "model.encoder.orthogonal_preconditioner.seed=${OLP_SEED}"
  model.encoder.orthogonal_preconditioner.collect_stats=true
  model.encoder.gaussian_reliability.enabled=true
  model.encoder.gaussian_reliability.r_min=0.05
  model.encoder.gaussian_reliability.r_init=0.95
  model.encoder.gaussian_reliability.detach_gate=false
  "loss=[mse,lpips,reliability]"
  loss.reliability.enabled=true
  "loss.reliability.weight=${REL_WEIGHT}"
  "loss.reliability.warmup_start=${REL_WARMUP_START}"
  "loss.reliability.warmup_end=${REL_WARMUP_END}"
  "trainer.max_steps=${MAX_STEPS}"
  "trainer.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES}"
  "trainer.num_sanity_val_steps=${NUM_SANITY_VAL_STEPS:-0}"
  "trainer.val_check_interval=${VAL_CHECK_INTERVAL:-100000000}"
  "optimizer.lr=${LR}"
  "data_loader.train.batch_size=${BATCH_SIZE}"
  "data_loader.train.num_workers=${TRAIN_NUM_WORKERS:-16}"
  "data_loader.val.num_workers=${VAL_NUM_WORKERS:-1}"
  "data_loader.test.num_workers=${TEST_NUM_WORKERS:-4}"
  "checkpointing.every_n_train_steps=${CKPT_EVERY_N_STEPS}"
  checkpointing.save_top_k=3
  checkpointing.save_weights_only=true
  "train.print_log_every_n_steps=${PRINT_LOG_EVERY_N_STEPS:-100}"
  "hydra.run.dir=${RUN_DIR}"
)

printf "%q " "${CMD[@]}" > "${RUN_DIR}/command.sh"
printf "\n" >> "${RUN_DIR}/command.sh"

cat > "${RUN_DIR}/run_config.env" <<EOF
RUN_NAME=${RUN_NAME}
RUN_ID=${RUN_ID}
RUN_DIR=${RUN_DIR}
BASE_CKPT=${BASE_CKPT}
DATA_ROOT=${DATA_ROOT}
BATCH_SIZE=${BATCH_SIZE}
ACCUMULATE_GRAD_BATCHES=${ACCUMULATE_GRAD_BATCHES}
EFFECTIVE_BATCH_SINGLE_GPU=$(( BATCH_SIZE * ACCUMULATE_GRAD_BATCHES ))
MAX_STEPS=${MAX_STEPS}
CKPT_EVERY_N_STEPS=${CKPT_EVERY_N_STEPS}
LR=${LR}
OLP_SEED=${OLP_SEED}
REL_WEIGHT=${REL_WEIGHT}
REL_WARMUP_START=${REL_WARMUP_START}
REL_WARMUP_END=${REL_WARMUP_END}
WANDB_MODE=${WANDB_MODE}
EOF

"${CMD[@]}"
