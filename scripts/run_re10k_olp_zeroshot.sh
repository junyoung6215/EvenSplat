#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="re10k_olp_zeroshot"
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
  mode=test
  "wandb.mode=${WANDB_MODE:-disabled}"
  "wandb.name=${RUN_NAME}"
  dataset/view_sampler@dataset.re10k.view_sampler=evaluation
  dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json
  checkpointing.load=./pretrained_weights/re10k.ckpt
  model.encoder.orthogonal_preconditioner.enabled=true
  model.encoder.orthogonal_preconditioner.mode=hadamard
  model.encoder.orthogonal_preconditioner.dim=256
  model.encoder.orthogonal_preconditioner.mix=1.0
  model.encoder.orthogonal_preconditioner.target_mix=1.0
  model.encoder.orthogonal_preconditioner.warmup_steps=0
  model.encoder.orthogonal_preconditioner.seed="${OLP_SEED:-0}"
  model.encoder.orthogonal_preconditioner.collect_stats=true
  test.save_image=true
  test.save_olp_stats=true
  "test.olp_tau_alpha=${OLP_TAU_ALPHA:-0.95}"
  "test.olp_tau_scale=${OLP_TAU_SCALE:-null}"
  "test.olp_tau_volume=${OLP_TAU_VOLUME:-null}"
  "test.olp_adaptive_tau_scale_quantile=${OLP_ADAPTIVE_TAU_SCALE_QUANTILE:-0.95}"
  "test.output_path=outputs/olp_re10k"
  "hydra.run.dir=${RUN_DIR}"
)

printf "%q " "${CMD[@]}" > "${RUN_DIR}/command.sh"
printf "\n" >> "${RUN_DIR}/command.sh"
"${CMD[@]}"
