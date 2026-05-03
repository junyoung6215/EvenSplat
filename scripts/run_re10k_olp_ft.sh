#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="re10k_olp_ft"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-outputs/olp_re10k/${RUN_NAME}/${RUN_ID}}"
mkdir -p "${RUN_DIR}/fingerprint"

git rev-parse HEAD > "${RUN_DIR}/fingerprint/git_commit.txt"
if [[ "${CHECK_ENV:-1}" == "1" ]]; then
  python scripts/check_a100_env.py --output-dir "${RUN_DIR}/fingerprint"
fi

CMD=(
  python -m src.main
  +experiment=re10k
  mode=train
  "wandb.mode=${WANDB_MODE:-disabled}"
  "wandb.name=${RUN_NAME}"
  model.encoder.pretrained_weights=./pretrained_weights/re10k.ckpt
  checkpointing.load=null
  model.encoder.orthogonal_preconditioner.enabled=true
  model.encoder.orthogonal_preconditioner.mode=hadamard
  model.encoder.orthogonal_preconditioner.dim=256
  model.encoder.orthogonal_preconditioner.mix=0.0
  model.encoder.orthogonal_preconditioner.target_mix=1.0
  "model.encoder.orthogonal_preconditioner.warmup_steps=${OLP_WARMUP_STEPS:-5000}"
  model.encoder.orthogonal_preconditioner.seed="${OLP_SEED:-0}"
  model.encoder.orthogonal_preconditioner.collect_stats=true
  "trainer.max_steps=${MAX_STEPS:-18751}"
  "optimizer.lr=${LR:-2e-4}"
  "data_loader.train.batch_size=${BATCH_SIZE:-16}"
  "checkpointing.every_n_train_steps=${CKPT_EVERY_N_STEPS:-5000}"
  "hydra.run.dir=${RUN_DIR}"
)

printf "%q " "${CMD[@]}" > "${RUN_DIR}/command.sh"
printf "\n" >> "${RUN_DIR}/command.sh"
"${CMD[@]}"
