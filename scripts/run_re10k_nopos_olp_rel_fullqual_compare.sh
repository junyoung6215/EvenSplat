#!/usr/bin/env bash
set -euo pipefail

# One-shot comparison script:
# 1) Run full RE10K eval for the original NoPoSplat baseline.
# 2) Create a seed-controlled random qualitative subset.
# 3) Run NoPoSplat, OLP-FT, and OLP+Reliability on that subset with images.
# 4) Build side-by-side qualitative panels and print metric summaries.

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ROOT_DIR="${ROOT_DIR:-outputs/reliability_residual_eval/nopos_olp_rel_compare/${RUN_ID}}"
FULL_ROOT="${FULL_ROOT:-${ROOT_DIR}/full_eval}"
QUAL_ROOT="${QUAL_ROOT:-${ROOT_DIR}/qual_eval}"
SIDE_BY_SIDE_DIR="${SIDE_BY_SIDE_DIR:-${ROOT_DIR}/side_by_side}"
FINGERPRINT_DIR="${ROOT_DIR}/fingerprint"

DATA_ROOT="${DATA_ROOT:-/home/data/olp_h_re10k_slot0/re10k_hf}"
FULL_INDEX="${FULL_INDEX:-assets/evaluation_index_re10k.json}"
QUAL_SOURCE_INDEX="${QUAL_SOURCE_INDEX:-${FULL_INDEX}}"
QUAL_INDEX="${QUAL_INDEX:-${ROOT_DIR}/evaluation_index_re10k_qual${QUAL_COUNT:-10}_seed${QUAL_SEED:-0}.json}"
QUAL_COUNT="${QUAL_COUNT:-10}"
QUAL_SEED="${QUAL_SEED:-0}"
QUAL_TARGETS_PER_SCENE="${QUAL_TARGETS_PER_SCENE:-1}"

NOPOS_CKPT="${NOPOS_CKPT:-/workspace/olp_h_re10k_slot0/NoPoSplat/pretrained_weights/re10k.ckpt}"
OLP_CKPT="${OLP_CKPT:-/home/data/olp_h_re10k_slot0/noposplat_runs/re10k/re10k_olp_ft/full_20260511_300k_v2_resume122k/train/checkpoints/epoch_0-step_300000.ckpt}"
REL_CKPT="${REL_CKPT:-outputs/reliability_residual_ft/re10k_olp_reliability_residual_ft_20260527_084014/checkpoints/epoch_1-step_75000.ckpt}"

# Existing full evals already produced for OLP and Reliability. These are used
# only for final summary unless RUN_FULL_OLP/RUN_FULL_REL are enabled.
EXISTING_FULL_ROOT="${EXISTING_FULL_ROOT:-outputs/reliability_residual_eval/20260603_094830}"

NOPOS_FULL_RUN_NAME="${NOPOS_FULL_RUN_NAME:-re10k_nopos_full}"
OLP_FULL_RUN_NAME="${OLP_FULL_RUN_NAME:-re10k_olp300k_baseline_full}"
REL_FULL_RUN_NAME="${REL_FULL_RUN_NAME:-re10k_relctx_75k_full}"

NOPOS_QUAL_RUN_NAME="${NOPOS_QUAL_RUN_NAME:-re10k_nopos_qual${QUAL_COUNT}}"
OLP_QUAL_RUN_NAME="${OLP_QUAL_RUN_NAME:-re10k_olp300k_qual${QUAL_COUNT}}"
REL_QUAL_RUN_NAME="${REL_QUAL_RUN_NAME:-re10k_relctx_75k_qual${QUAL_COUNT}}"

RUN_FULL_NOPOS="${RUN_FULL_NOPOS:-1}"
RUN_FULL_OLP="${RUN_FULL_OLP:-0}"
RUN_FULL_REL="${RUN_FULL_REL:-0}"
RUN_QUAL="${RUN_QUAL:-1}"
RUN_SIDE_BY_SIDE="${RUN_SIDE_BY_SIDE:-1}"

WANDB_MODE="${WANDB_MODE:-disabled}"
ALIGN_POSE="${ALIGN_POSE:-true}"
TEST_NUM_WORKERS="${TEST_NUM_WORKERS:-4}"
FULL_LIMIT_TEST_BATCHES="${FULL_LIMIT_TEST_BATCHES:-null}"
QUAL_LIMIT_TEST_BATCHES="${QUAL_LIMIT_TEST_BATCHES:-null}"
OLP_SEED="${OLP_SEED:-0}"
OLP_TAU_ALPHA="${OLP_TAU_ALPHA:-0.95}"
OLP_TAU_SCALE="${OLP_TAU_SCALE:-null}"
OLP_TAU_VOLUME="${OLP_TAU_VOLUME:-null}"
OLP_ADAPTIVE_TAU_SCALE_QUANTILE="${OLP_ADAPTIVE_TAU_SCALE_QUANTILE:-0.95}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/noposplat-olp-cu118-clean/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

mkdir -p "${FULL_ROOT}" "${QUAL_ROOT}" "${SIDE_BY_SIDE_DIR}" "${FINGERPRINT_DIR}"

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local label="$1"
  local path="$2"
  if [[ ! -d "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

require_dir DATA_ROOT "${DATA_ROOT}"
require_file FULL_INDEX "${FULL_INDEX}"
require_file QUAL_SOURCE_INDEX "${QUAL_SOURCE_INDEX}"
require_file NOPOS_CKPT "${NOPOS_CKPT}"
require_file OLP_CKPT "${OLP_CKPT}"
require_file REL_CKPT "${REL_CKPT}"

if git rev-parse HEAD > "${FINGERPRINT_DIR}/git_commit.txt" 2>/dev/null; then
  git status --short > "${FINGERPRINT_DIR}/git_status.txt" || true
fi
cat > "${FINGERPRINT_DIR}/run_config.env" <<EOF
RUN_ID=${RUN_ID}
ROOT_DIR=${ROOT_DIR}
DATA_ROOT=${DATA_ROOT}
FULL_INDEX=${FULL_INDEX}
QUAL_SOURCE_INDEX=${QUAL_SOURCE_INDEX}
QUAL_INDEX=${QUAL_INDEX}
QUAL_COUNT=${QUAL_COUNT}
QUAL_SEED=${QUAL_SEED}
QUAL_TARGETS_PER_SCENE=${QUAL_TARGETS_PER_SCENE}
NOPOS_CKPT=${NOPOS_CKPT}
OLP_CKPT=${OLP_CKPT}
REL_CKPT=${REL_CKPT}
EXISTING_FULL_ROOT=${EXISTING_FULL_ROOT}
ALIGN_POSE=${ALIGN_POSE}
EOF

make_random_qual_index() {
  "${PYTHON_BIN}" - "${QUAL_SOURCE_INDEX}" "${QUAL_INDEX}" "${QUAL_COUNT}" "${QUAL_SEED}" "${QUAL_TARGETS_PER_SCENE}" <<'PYINDEX'
import copy
import json
import random
import sys
from pathlib import Path

source = Path(sys.argv[1])
output = Path(sys.argv[2])
count = int(sys.argv[3])
seed = int(sys.argv[4])
targets_per_scene = int(sys.argv[5])

index = json.loads(source.read_text())
valid = [(scene, entry) for scene, entry in index.items() if entry is not None]
if len(valid) < count:
    raise SystemExit(f"Only {len(valid)} valid scenes in {source}, requested {count}.")

rng = random.Random(seed)
selected = rng.sample(valid, count)
subset = {}
for scene, entry in selected:
    item = copy.deepcopy(entry)
    targets = list(item.get("target", []))
    if targets_per_scene > 0 and len(targets) > targets_per_scene:
        item["target"] = rng.sample(targets, targets_per_scene)
    subset[scene] = item

output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(subset, indent=2, sort_keys=True) + "\n")
target_count = sum(len(entry.get("target", [])) for entry in subset.values())
print(f"Wrote random qualitative index: {output}")
print(f"  scenes={len(subset)} targets={target_count} seed={seed}")
PYINDEX
}

print_metric_summary() {
  local title="$1"
  shift
  "${PYTHON_BIN}" - "${title}" "$@" <<'PYSUMMARY'
import json
import sys
from pathlib import Path

title = sys.argv[1]
args = sys.argv[2:]
print(f"\n{title}")
print("model,psnr,ssim,lpips,path")
for spec in args:
    label, path = spec.split("=", 1)
    p = Path(path) / "test_metrics.json"
    if not p.exists():
        print(f"{label},missing,missing,missing,{p}")
        continue
    data = json.loads(p.read_text())
    if isinstance(data, list) and data:
        data = data[0]
    print(
        f"{label},{data.get('psnr_ours', float('nan')):.6f},"
        f"{data.get('ssim_ours', float('nan')):.6f},"
        f"{data.get('lpips_ours', float('nan')):.6f},{p.parent}"
    )
PYSUMMARY
}

run_eval() {
  local run_name="$1"
  local checkpoint="$2"
  local precond_enabled="$3"
  local reliability_enabled="$4"
  local index_path="$5"
  local output_root="$6"
  local limit_batches="$7"
  local save_image="$8"
  local save_compare="$9"
  local run_dir="${output_root}/${run_name}"
  mkdir -p "${run_dir}"

  local mix="0.0"
  if [[ "${precond_enabled}" == "true" ]]; then
    mix="1.0"
  fi

  local cmd=(
    "${PYTHON_BIN}" -m src.main
    +experiment=re10k
    mode=test
    "wandb.mode=${WANDB_MODE}"
    "wandb.name=${run_name}"
    dataset/view_sampler@dataset.re10k.view_sampler=evaluation
    "dataset.re10k.view_sampler.index_path=${index_path}"
    "dataset.re10k.roots=[${DATA_ROOT}]"
    "checkpointing.load=${checkpoint}"
    "model.encoder.orthogonal_preconditioner.enabled=${precond_enabled}"
    model.encoder.orthogonal_preconditioner.mode=hadamard
    model.encoder.orthogonal_preconditioner.dim=256
    "model.encoder.orthogonal_preconditioner.mix=${mix}"
    "model.encoder.orthogonal_preconditioner.target_mix=${mix}"
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
    "test.save_image=${save_image}"
    "test.save_compare=${save_compare}"
    test.save_video=false
    "test.output_path=${output_root}"
    "data_loader.test.num_workers=${TEST_NUM_WORKERS}"
    "trainer.limit_test_batches=${limit_batches}"
    "hydra.run.dir=${run_dir}"
  )

  printf "%q " "${cmd[@]}" > "${run_dir}/command.sh"
  printf "\n" >> "${run_dir}/command.sh"

  printf "\n=== Running %s ===\n" "${run_name}"
  "${cmd[@]}" 2>&1 | tee "${run_dir}/stdout.log"
}

run_side_by_side() {
  local metrics_path="${SIDE_BY_SIDE_DIR}/metrics.json"
  local cmd=(
    "${PYTHON_BIN}" -u -m src.scripts.compute_metrics
    +experiment=re10k
    +evaluation=re10k
    "dataset.re10k.view_sampler.index_path=${QUAL_INDEX}"
    "dataset.re10k.roots=[${DATA_ROOT}]"
    "evaluation.methods=[{name:NoPoSplat,key:nopos,path:${QUAL_ROOT}/${NOPOS_QUAL_RUN_NAME}},{name:OLP-FT,key:olp,path:${QUAL_ROOT}/${OLP_QUAL_RUN_NAME}},{name:OLP-Reliability,key:rel,path:${QUAL_ROOT}/${REL_QUAL_RUN_NAME}}]"
    "evaluation.side_by_side_path=${SIDE_BY_SIDE_DIR}/panels"
    evaluation.animate_side_by_side=false
    "evaluation.output_metrics_path=${metrics_path}"
    "hydra.run.dir=${SIDE_BY_SIDE_DIR}/hydra"
  )

  printf "%q " "${cmd[@]}" > "${SIDE_BY_SIDE_DIR}/command.sh"
  printf "\n" >> "${SIDE_BY_SIDE_DIR}/command.sh"

  printf "\n=== Building side-by-side qualitative panels ===\n"
  "${cmd[@]}" 2>&1 | tee "${SIDE_BY_SIDE_DIR}/stdout.log"
  echo "Side-by-side panels: ${SIDE_BY_SIDE_DIR}/panels"
  echo "Side-by-side metrics: ${metrics_path}"
}

if [[ "${RUN_FULL_NOPOS}" == "1" ]]; then
  run_eval "${NOPOS_FULL_RUN_NAME}" "${NOPOS_CKPT}" false false "${FULL_INDEX}" "${FULL_ROOT}" "${FULL_LIMIT_TEST_BATCHES}" false false
fi

if [[ "${RUN_FULL_OLP}" == "1" ]]; then
  run_eval "${OLP_FULL_RUN_NAME}" "${OLP_CKPT}" true false "${FULL_INDEX}" "${FULL_ROOT}" "${FULL_LIMIT_TEST_BATCHES}" false false
fi

if [[ "${RUN_FULL_REL}" == "1" ]]; then
  run_eval "${REL_FULL_RUN_NAME}" "${REL_CKPT}" true true "${FULL_INDEX}" "${FULL_ROOT}" "${FULL_LIMIT_TEST_BATCHES}" false false
fi

if [[ "${RUN_QUAL}" == "1" ]]; then
  make_random_qual_index
  run_eval "${NOPOS_QUAL_RUN_NAME}" "${NOPOS_CKPT}" false false "${QUAL_INDEX}" "${QUAL_ROOT}" "${QUAL_LIMIT_TEST_BATCHES}" true true
  run_eval "${OLP_QUAL_RUN_NAME}" "${OLP_CKPT}" true false "${QUAL_INDEX}" "${QUAL_ROOT}" "${QUAL_LIMIT_TEST_BATCHES}" true true
  run_eval "${REL_QUAL_RUN_NAME}" "${REL_CKPT}" true true "${QUAL_INDEX}" "${QUAL_ROOT}" "${QUAL_LIMIT_TEST_BATCHES}" true true
fi

if [[ "${RUN_SIDE_BY_SIDE}" == "1" ]]; then
  run_side_by_side
fi

# Full summary: NoPo may be from this run; OLP/Rel default to the already completed run.
NOPOS_FULL_DIR="${FULL_ROOT}/${NOPOS_FULL_RUN_NAME}"
OLP_FULL_DIR="${FULL_ROOT}/${OLP_FULL_RUN_NAME}"
REL_FULL_DIR="${FULL_ROOT}/${REL_FULL_RUN_NAME}"
if [[ "${RUN_FULL_OLP}" != "1" ]]; then
  OLP_FULL_DIR="${EXISTING_FULL_ROOT}/${OLP_FULL_RUN_NAME}"
fi
if [[ "${RUN_FULL_REL}" != "1" ]]; then
  REL_FULL_DIR="${EXISTING_FULL_ROOT}/${REL_FULL_RUN_NAME}"
fi
print_metric_summary \
  "FULL_EVAL_SUMMARY" \
  "NoPo=${NOPOS_FULL_DIR}" \
  "OLP=${OLP_FULL_DIR}" \
  "Rel=${REL_FULL_DIR}"

print_metric_summary \
  "QUAL_EVAL_SUMMARY" \
  "NoPo=${QUAL_ROOT}/${NOPOS_QUAL_RUN_NAME}" \
  "OLP=${QUAL_ROOT}/${OLP_QUAL_RUN_NAME}" \
  "Rel=${QUAL_ROOT}/${REL_QUAL_RUN_NAME}"

cat <<EOF

Done.
Root: ${ROOT_DIR}
Full eval outputs: ${FULL_ROOT}
Qual eval outputs: ${QUAL_ROOT}
Qual index: ${QUAL_INDEX}
Side-by-side panels: ${SIDE_BY_SIDE_DIR}/panels
EOF
