#!/usr/bin/env bash
set -euo pipefail

: "${BASELINE_CKPT:?Set BASELINE_CKPT to the fine-tuned control checkpoint path.}"
: "${OLP_CKPT:?Set OLP_CKPT to the fine-tuned OLP checkpoint path.}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ROOT_DIR="${ROOT_DIR:-outputs/olp_re10k/collect_stats/${RUN_ID}}"
BASELINE_EVAL_NAME="${BASELINE_EVAL_NAME:-re10k_ft_control_eval}"
OLP_EVAL_NAME="${OLP_EVAL_NAME:-re10k_olp_ft_eval}"
mkdir -p "${ROOT_DIR}/fingerprint"

git rev-parse HEAD > "${ROOT_DIR}/fingerprint/git_commit.txt"
echo "${OLP_SEED:-0}" > "${ROOT_DIR}/fingerprint/olp_seed.txt"
if [[ "${CHECK_ENV:-1}" == "1" ]]; then
  python scripts/check_cuda118_env.py --output-dir "${ROOT_DIR}/fingerprint"
fi

run_eval() {
  local run_name="$1"
  local checkpoint="$2"
  local olp_mix="$3"
  local olp_target_mix="$4"
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
    model.encoder.orthogonal_preconditioner.enabled=true
    model.encoder.orthogonal_preconditioner.mode=hadamard
    model.encoder.orthogonal_preconditioner.dim=256
    "model.encoder.orthogonal_preconditioner.mix=${olp_mix}"
    "model.encoder.orthogonal_preconditioner.target_mix=${olp_target_mix}"
    model.encoder.orthogonal_preconditioner.warmup_steps=0
    model.encoder.orthogonal_preconditioner.seed="${OLP_SEED:-0}"
    model.encoder.orthogonal_preconditioner.collect_stats=true
    test.save_image="${SAVE_IMAGE:-false}"
    test.save_olp_stats=true
    "test.olp_tau_alpha=${OLP_TAU_ALPHA:-0.95}"
    "test.olp_tau_scale=${OLP_TAU_SCALE:-null}"
    "test.olp_tau_volume=${OLP_TAU_VOLUME:-null}"
    "test.olp_adaptive_tau_scale_quantile=${OLP_ADAPTIVE_TAU_SCALE_QUANTILE:-0.95}"
    "test.output_path=outputs/olp_re10k"
    "hydra.run.dir=${run_dir}"
  )

  printf "%q " "${CMD[@]}" > "${run_dir}/command.sh"
  printf "\n" >> "${run_dir}/command.sh"
  "${CMD[@]}"
}

# Baseline eval keeps the OLP-H module active with mix=0 so latent stats are
# collected while the output remains identity-equivalent.
run_eval "${BASELINE_EVAL_NAME}" "${BASELINE_CKPT}" 0.0 0.0
run_eval "${OLP_EVAL_NAME}" "${OLP_CKPT}" 1.0 1.0
