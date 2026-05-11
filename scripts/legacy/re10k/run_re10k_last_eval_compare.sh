#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ROOT_DIR="${ROOT_DIR:-outputs/olp_re10k/last_eval_compare/${RUN_ID}}"
TEST_OUTPUT_PATH="${TEST_OUTPUT_PATH:-outputs/olp_re10k}"
DATA_ROOT="${DATA_ROOT:-datasets/re10k}"
BASELINE_CKPT="${BASELINE_CKPT:-./pretrained_weights/re10k.ckpt}"
OLP_CKPT="${OLP_CKPT:-outputs/olp_re10k/re10k_olp_ft_7k_bs4/20260506_151346/train/checkpoints/last.ckpt}"
METRIC_COUNT="${METRIC_COUNT:-2000}"
VIS_COUNT="${VIS_COUNT:-10}"
VIS_TARGETS_PER_SCENE="${VIS_TARGETS_PER_SCENE:-1}"
METRIC_INDEX="${METRIC_INDEX:-${ROOT_DIR}/evaluation_index_re10k_metric${METRIC_COUNT}.json}"
VIS_INDEX="${VIS_INDEX:-${ROOT_DIR}/evaluation_index_re10k_vis${VIS_COUNT}.json}"
METRIC_RUN_NAME="${METRIC_RUN_NAME:-re10k_olp_last_metric${METRIC_COUNT}}"
VIS_ORIGINAL_RUN_NAME="${VIS_ORIGINAL_RUN_NAME:-re10k_noposplat_vis${VIS_COUNT}}"
VIS_BASELINE_RUN_NAME="${VIS_BASELINE_RUN_NAME:-re10k_olp_noft_vis${VIS_COUNT}}"
VIS_OLP_RUN_NAME="${VIS_OLP_RUN_NAME:-re10k_olp_last_vis${VIS_COUNT}}"
SIDE_BY_SIDE_PATH="${SIDE_BY_SIDE_PATH:-${TEST_OUTPUT_PATH}/re10k_noposplat_vs_olp_noft_vs_olp_ft_vis${VIS_COUNT}}"
WANDB_MODE="${WANDB_MODE:-disabled}"
OLP_SEED="${OLP_SEED:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  elif [[ -x "/root/miniconda3/envs/noposplat-olp-cu118-clean/bin/python" ]]; then
    PYTHON_BIN="/root/miniconda3/envs/noposplat-olp-cu118-clean/bin/python"
  else
    PYTHON_BIN=python3
  fi
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found. Activate the NoPoSplat conda env or set PYTHON_BIN." >&2
  exit 127
fi

mkdir -p "${ROOT_DIR}/fingerprint"

git rev-parse HEAD > "${ROOT_DIR}/fingerprint/git_commit.txt"
echo "${OLP_SEED}" > "${ROOT_DIR}/fingerprint/olp_seed.txt"

"${PYTHON_BIN}" scripts/make_eval_subset.py \
  --input assets/evaluation_index_re10k.json \
  --output "${METRIC_INDEX}" \
  --count "${METRIC_COUNT}"

"${PYTHON_BIN}" scripts/make_eval_subset.py \
  --input assets/evaluation_index_re10k.json \
  --output "${VIS_INDEX}" \
  --count "${VIS_COUNT}" \
  --targets-per-scene "${VIS_TARGETS_PER_SCENE}"

run_eval() {
  local run_name="$1"
  local checkpoint="$2"
  local index_path="$3"
  local save_image="$4"
  local olp_enabled="$5"
  local olp_mix="$6"
  local olp_target_mix="$7"
  local save_olp_stats="$8"
  local run_dir="${ROOT_DIR}/${run_name}"

  mkdir -p "${run_dir}"
  CMD=(
    "${PYTHON_BIN}" -u -m src.main
    +experiment=re10k
    mode=test
    "wandb.mode=${WANDB_MODE}"
    "wandb.name=${run_name}"
    dataset/view_sampler@dataset.re10k.view_sampler=evaluation
    "dataset.re10k.view_sampler.index_path=${index_path}"
    "dataset.re10k.roots=[${DATA_ROOT}]"
    "checkpointing.load=${checkpoint}"
    "model.encoder.orthogonal_preconditioner.enabled=${olp_enabled}"
    model.encoder.orthogonal_preconditioner.mode=hadamard
    model.encoder.orthogonal_preconditioner.dim=256
    "model.encoder.orthogonal_preconditioner.mix=${olp_mix}"
    "model.encoder.orthogonal_preconditioner.target_mix=${olp_target_mix}"
    model.encoder.orthogonal_preconditioner.warmup_steps=0
    "model.encoder.orthogonal_preconditioner.seed=${OLP_SEED}"
    model.encoder.orthogonal_preconditioner.collect_stats=true
    "test.save_image=${save_image}"
    "test.save_olp_stats=${save_olp_stats}"
    "test.olp_tau_alpha=${OLP_TAU_ALPHA:-0.95}"
    "test.olp_tau_scale=${OLP_TAU_SCALE:-null}"
    "test.olp_tau_volume=${OLP_TAU_VOLUME:-null}"
    "test.olp_adaptive_tau_scale_quantile=${OLP_ADAPTIVE_TAU_SCALE_QUANTILE:-0.95}"
    "test.output_path=${TEST_OUTPUT_PATH}"
    "hydra.run.dir=${run_dir}"
  )

  printf "%q " "${CMD[@]}" > "${run_dir}/command.sh"
  printf "\n" >> "${run_dir}/command.sh"
  "${CMD[@]}"
}

run_eval "${METRIC_RUN_NAME}" "${OLP_CKPT}" "${METRIC_INDEX}" false true 1.0 1.0 true
run_eval "${VIS_ORIGINAL_RUN_NAME}" "${BASELINE_CKPT}" "${VIS_INDEX}" true false 0.0 0.0 false
run_eval "${VIS_BASELINE_RUN_NAME}" "${BASELINE_CKPT}" "${VIS_INDEX}" true true 1.0 1.0 true
run_eval "${VIS_OLP_RUN_NAME}" "${OLP_CKPT}" "${VIS_INDEX}" true true 1.0 1.0 true

"${PYTHON_BIN}" -u -m src.scripts.compute_metrics \
  +experiment=re10k \
  +evaluation=re10k \
  "dataset.re10k.view_sampler.index_path=${VIS_INDEX}" \
  "dataset.re10k.roots=[${DATA_ROOT}]" \
  "evaluation.methods=[{name:NoPoSplat,key:noposplat,path:${TEST_OUTPUT_PATH}/${VIS_ORIGINAL_RUN_NAME}},{name:OLP-NoFT,key:olp_noft,path:${TEST_OUTPUT_PATH}/${VIS_BASELINE_RUN_NAME}},{name:OLP-FT,key:olp_ft,path:${TEST_OUTPUT_PATH}/${VIS_OLP_RUN_NAME}}]" \
  "evaluation.side_by_side_path=${SIDE_BY_SIDE_PATH}" \
  evaluation.animate_side_by_side=false \
  "evaluation.output_metrics_path=${SIDE_BY_SIDE_PATH}/metrics.json" \
  "hydra.run.dir=${ROOT_DIR}/side_by_side"

cat > "${ROOT_DIR}/summary.txt" <<EOF
metric_index=${METRIC_INDEX}
test_output_path=${TEST_OUTPUT_PATH}
metric_output=${TEST_OUTPUT_PATH}/${METRIC_RUN_NAME}
metric_metrics_json=${ROOT_DIR}/${METRIC_RUN_NAME}/test_metrics.json
vis_index=${VIS_INDEX}
vis_targets_per_scene=${VIS_TARGETS_PER_SCENE}
noposplat_output=${TEST_OUTPUT_PATH}/${VIS_ORIGINAL_RUN_NAME}
noft_output=${TEST_OUTPUT_PATH}/${VIS_BASELINE_RUN_NAME}
ft_output=${TEST_OUTPUT_PATH}/${VIS_OLP_RUN_NAME}
side_by_side=${SIDE_BY_SIDE_PATH}
side_by_side_metrics=${SIDE_BY_SIDE_PATH}/metrics.json
olp_checkpoint=${OLP_CKPT}
baseline_checkpoint=${BASELINE_CKPT}
EOF

cat "${ROOT_DIR}/summary.txt"
