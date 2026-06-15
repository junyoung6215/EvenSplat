#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="re10k_olp_ft"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-outputs/olp_re10k/${RUN_NAME}/${RUN_ID}}"
mkdir -p "${RUN_DIR}/fingerprint"

git rev-parse HEAD > "${RUN_DIR}/fingerprint/git_commit.txt"
echo "${OLP_SEED:-0}" > "${RUN_DIR}/fingerprint/olp_seed.txt"
if [[ "${CHECK_ENV:-1}" == "1" ]]; then
  python scripts/check_cuda118_env.py --output-dir "${RUN_DIR}/fingerprint"
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
  "trainer.num_sanity_val_steps=${NUM_SANITY_VAL_STEPS:-0}"
  "trainer.val_check_interval=${VAL_CHECK_INTERVAL:-100000000}"
  "trainer.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES:-1}"
  "optimizer.lr=${LR:-2e-4}"
  "data_loader.train.batch_size=${BATCH_SIZE:-16}"
  "data_loader.train.num_workers=${TRAIN_NUM_WORKERS:-16}"
  "data_loader.val.num_workers=${VAL_NUM_WORKERS:-1}"
  "data_loader.test.num_workers=${TEST_NUM_WORKERS:-4}"
  "checkpointing.every_n_train_steps=${CKPT_EVERY_N_STEPS:-5000}"
  "hydra.run.dir=${RUN_DIR}"
)

printf "%q " "${CMD[@]}" > "${RUN_DIR}/command.sh"
printf "\n" >> "${RUN_DIR}/command.sh"
"${CMD[@]}"
