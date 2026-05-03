#!/usr/bin/env bash
set -euo pipefail

: "${BASELINE_CKPT:?Set BASELINE_CKPT to the fine-tuned control checkpoint path.}"
: "${OLP_CKPT:?Set OLP_CKPT to the fine-tuned OLP checkpoint path.}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ROOT_DIR="${ROOT_DIR:-outputs/olp_re10k/collect_stats/${RUN_ID}}"
mkdir -p "${ROOT_DIR}/fingerprint"

git rev-parse HEAD > "${ROOT_DIR}/fingerprint/git_commit.txt"
if [[ "${CHECK_ENV:-1}" == "1" ]]; then
  python scripts/check_a100_env.py --output-dir "${ROOT_DIR}/fingerprint"
fi

run_eval() {
  local run_name="$1"
  local checkpoint="$2"
  local olp_enabled="$3"
  local run_dir="${ROOT_DIR}/${run_name}"
  mkdir -p "${run_dir}"

  CMD=(
    python -m src.main
    +experiment=re10k
    mode=test
    "wandb.mode=${WANDB_MODE:-disabled}"
    "wandb.name=${run_name}"
    dataset/view_sampler@dataset.re10k.view_sampler=evaluation
    dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json
    "checkpointing.load=${checkpoint}"
    "model.encoder.orthogonal_preconditioner.enabled=${olp_enabled}"
    model.encoder.orthogonal_preconditioner.mode=hadamard
    model.encoder.orthogonal_preconditioner.dim=256
    model.encoder.orthogonal_preconditioner.mix=1.0
    model.encoder.orthogonal_preconditioner.target_mix=1.0
    model.encoder.orthogonal_preconditioner.warmup_steps=0
    model.encoder.orthogonal_preconditioner.seed="${OLP_SEED:-0}"
    model.encoder.orthogonal_preconditioner.collect_stats=true
    test.save_image="${SAVE_IMAGE:-false}"
    test.save_olp_stats=true
    "test.output_path=outputs/olp_re10k"
    "hydra.run.dir=${run_dir}"
  )

  printf "%q " "${CMD[@]}" > "${run_dir}/command.sh"
  printf "\n" >> "${run_dir}/command.sh"
  "${CMD[@]}"
}

run_eval "re10k_ft_control_eval" "${BASELINE_CKPT}" false
run_eval "re10k_olp_ft_eval" "${OLP_CKPT}" true
