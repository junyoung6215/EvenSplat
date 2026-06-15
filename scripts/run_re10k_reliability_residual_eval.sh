#!/usr/bin/env bash
set -euo pipefail

# Evaluate the reliability-residual checkpoint and, by default, the OLP 300k
# baseline with the same RE10K evaluation protocol. The OLP/Gaussian recorder
# writes opacity, reliability, active-count, and compression proxy statistics.

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EVAL_ROOT="${EVAL_ROOT:-outputs/reliability_residual_eval/${RUN_ID}}"
DATA_ROOT="${DATA_ROOT:-/home/data/olp_h_re10k_slot0/re10k_hf}"
EVAL_INDEX="${EVAL_INDEX:-assets/evaluation_index_re10k.json}"

REL_CKPT="${REL_CKPT:-outputs/reliability_residual_ft/re10k_olp_reliability_residual_ft_20260527_084014/checkpoints/epoch_1-step_75000.ckpt}"
BASE_CKPT="${BASE_CKPT:-/home/data/olp_h_re10k_slot0/noposplat_runs/re10k/re10k_olp_ft/full_20260511_300k_v2_resume122k/train/checkpoints/epoch_0-step_300000.ckpt}"

RUN_REL="${RUN_REL:-1}"
RUN_BASELINE="${RUN_BASELINE:-1}"
REL_RUN_NAME="${REL_RUN_NAME:-re10k_relctx_75k_full}"
BASE_RUN_NAME="${BASE_RUN_NAME:-re10k_olp300k_baseline_full}"

WANDB_MODE="${WANDB_MODE:-disabled}"
SAVE_IMAGE="${SAVE_IMAGE:-false}"
SAVE_COMPARE="${SAVE_COMPARE:-false}"
SAVE_VIDEO="${SAVE_VIDEO:-false}"
ALIGN_POSE="${ALIGN_POSE:-true}"
LIMIT_TEST_BATCHES="${LIMIT_TEST_BATCHES:-null}"
TEST_NUM_WORKERS="${TEST_NUM_WORKERS:-4}"
OLP_SEED="${OLP_SEED:-0}"
OLP_TAU_ALPHA="${OLP_TAU_ALPHA:-0.95}"
OLP_TAU_SCALE="${OLP_TAU_SCALE:-null}"
OLP_TAU_VOLUME="${OLP_TAU_VOLUME:-null}"
OLP_ADAPTIVE_TAU_SCALE_QUANTILE="${OLP_ADAPTIVE_TAU_SCALE_QUANTILE:-0.95}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/noposplat-olp-cu118-clean/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

mkdir -p "${EVAL_ROOT}/fingerprint"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Missing DATA_ROOT: ${DATA_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${EVAL_INDEX}" ]]; then
  echo "Missing EVAL_INDEX: ${EVAL_INDEX}" >&2
  exit 1
fi
if [[ "${RUN_REL}" == "1" && ! -f "${REL_CKPT}" ]]; then
  echo "Missing REL_CKPT: ${REL_CKPT}" >&2
  exit 1
fi
if [[ "${RUN_BASELINE}" == "1" && ! -f "${BASE_CKPT}" ]]; then
  echo "Missing BASE_CKPT: ${BASE_CKPT}" >&2
  exit 1
fi

if git rev-parse HEAD > "${EVAL_ROOT}/fingerprint/git_commit.txt" 2>/dev/null; then
  git status --short > "${EVAL_ROOT}/fingerprint/git_status.txt" || true
fi
printf "%s\n" "${DATA_ROOT}" > "${EVAL_ROOT}/fingerprint/data_root.txt"
printf "%s\n" "${EVAL_INDEX}" > "${EVAL_ROOT}/fingerprint/eval_index.txt"
printf "%s\n" "${REL_CKPT}" > "${EVAL_ROOT}/fingerprint/rel_ckpt.txt"
printf "%s\n" "${BASE_CKPT}" > "${EVAL_ROOT}/fingerprint/base_ckpt.txt"

print_summary() {
  local run_dir="$1"
  local label="$2"
  "${PYTHON_BIN}" - "${run_dir}" "${label}" <<'PYSUMMARY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
label = sys.argv[2]
metrics_path = run_dir / "test_metrics.json"
stats_path = run_dir / "olp_stats_summary.json"

print(f"\n[{label}] {run_dir}")
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text())
    if isinstance(metrics, list) and metrics:
        metrics = metrics[0]
    for key in ("psnr_ours", "ssim_ours", "lpips_ours"):
        if key in metrics:
            print(f"  {key}: {metrics[key]:.6f}")
else:
    print("  test_metrics.json: missing")

if stats_path.exists():
    stats = json.loads(stats_path.read_text())
    keys = [
        "gaussian_count",
        "opacity_mean",
        "opacity_before_mean",
        "opacity_after_over_before_mean",
        "opacity_reduction_mean",
        "reliability_mean",
        "reliability_below_095",
        "opacity_active_frac_gt_0p05",
        "opacity_active_count_gt_0p05",
        "opacity_compression_factor_gt_0p05",
        "opacity_before_active_frac_gt_0p05",
        "opacity_after_active_frac_gt_0p05",
        "opacity_gate_pruned_cond_frac_gt_0p05",
        "opacity_after_compression_factor_gt_0p05",
    ]
    for key in keys:
        if key in stats:
            print(f"  {key}: {stats[key]:.6f}")
else:
    print("  olp_stats_summary.json: missing")
PYSUMMARY
}

run_eval() {
  local run_name="$1"
  local checkpoint="$2"
  local reliability_enabled="$3"
  local run_dir="${EVAL_ROOT}/${run_name}"
  mkdir -p "${run_dir}"

  local cmd=(
    "${PYTHON_BIN}" -m src.main
    +experiment=re10k
    mode=test
    "wandb.mode=${WANDB_MODE}"
    "wandb.name=${run_name}"
    dataset/view_sampler@dataset.re10k.view_sampler=evaluation
    "dataset.re10k.view_sampler.index_path=${EVAL_INDEX}"
    "dataset.re10k.roots=[${DATA_ROOT}]"
    "checkpointing.load=${checkpoint}"
    model.encoder.orthogonal_preconditioner.enabled=true
    model.encoder.orthogonal_preconditioner.mode=hadamard
    model.encoder.orthogonal_preconditioner.dim=256
    model.encoder.orthogonal_preconditioner.mix=1.0
    model.encoder.orthogonal_preconditioner.target_mix=1.0
    model.encoder.orthogonal_preconditioner.warmup_steps=0
    "model.encoder.orthogonal_preconditioner.seed=${OLP_SEED}"
    model.encoder.orthogonal_preconditioner.collect_stats=true
    "model.encoder.gaussian_reliability.enabled=${reliability_enabled}"
    model.encoder.gaussian_reliability.r_min=0.05
    model.encoder.gaussian_reliability.r_init=0.95
    model.encoder.gaussian_reliability.detach_gate=false
    test.save_olp_stats=true
    "test.olp_tau_alpha=${OLP_TAU_ALPHA}"
    "test.olp_tau_scale=${OLP_TAU_SCALE}"
    "test.olp_tau_volume=${OLP_TAU_VOLUME}"
    "test.olp_adaptive_tau_scale_quantile=${OLP_ADAPTIVE_TAU_SCALE_QUANTILE}"
    "test.align_pose=${ALIGN_POSE}"
    "test.save_image=${SAVE_IMAGE}"
    "test.save_compare=${SAVE_COMPARE}"
    "test.save_video=${SAVE_VIDEO}"
    "test.output_path=${EVAL_ROOT}"
    "data_loader.test.num_workers=${TEST_NUM_WORKERS}"
    "trainer.limit_test_batches=${LIMIT_TEST_BATCHES}"
    "hydra.run.dir=${run_dir}"
  )

  printf "%q " "${cmd[@]}" > "${run_dir}/command.sh"
  printf "\n" >> "${run_dir}/command.sh"

  cat > "${run_dir}/run_config.env" <<EOF
RUN_ID=${RUN_ID}
EVAL_ROOT=${EVAL_ROOT}
RUN_NAME=${run_name}
DATA_ROOT=${DATA_ROOT}
EVAL_INDEX=${EVAL_INDEX}
CHECKPOINT=${checkpoint}
GAUSSIAN_RELIABILITY_ENABLED=${reliability_enabled}
ALIGN_POSE=${ALIGN_POSE}
LIMIT_TEST_BATCHES=${LIMIT_TEST_BATCHES}
OLP_SEED=${OLP_SEED}
EOF

  echo "Running ${run_name}"
  "${cmd[@]}" 2>&1 | tee "${run_dir}/stdout.log"
  print_summary "${run_dir}" "${run_name}"
}

if [[ "${RUN_REL}" == "1" ]]; then
  run_eval "${REL_RUN_NAME}" "${REL_CKPT}" true
fi

if [[ "${RUN_BASELINE}" == "1" ]]; then
  run_eval "${BASE_RUN_NAME}" "${BASE_CKPT}" false
fi

cat <<EOF

Done.
Outputs are under: ${EVAL_ROOT}
Key files per run:
  test_metrics.json
  olp_stats_summary.json
  olp_stats_per_scene.csv
  stdout.log
EOF
