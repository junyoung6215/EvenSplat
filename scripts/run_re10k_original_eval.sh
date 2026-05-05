#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="re10k_original"
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
  model.encoder.orthogonal_preconditioner.enabled=false
  test.save_image=true
  test.save_olp_stats=false
  "test.output_path=outputs/olp_re10k"
  "hydra.run.dir=${RUN_DIR}"
)

printf "%q " "${CMD[@]}" > "${RUN_DIR}/command.sh"
printf "\n" >> "${RUN_DIR}/command.sh"
"${CMD[@]}"
