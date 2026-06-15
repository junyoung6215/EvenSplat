#!/usr/bin/env bash
set -euo pipefail

# Evaluate a quick transfer ablation:
#   NoPoSplat base weights + reliability head copied from an OLP+Reliability ckpt,
#   with the OLP/orthogonal preconditioner disabled at inference.
#
# This is not a separately trained NoPo+Rel ablation. It is a transfer sanity
# check for the reliability head learned in the OLP+Rel run.

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ROOT_DIR="${ROOT_DIR:-outputs/reliability_residual_eval/nopos_transferred_rel/${RUN_ID}}"
FULL_ROOT="${FULL_ROOT:-${ROOT_DIR}/full_eval}"
QUAL_ROOT="${QUAL_ROOT:-${ROOT_DIR}/qual_eval}"
SIDE_BY_SIDE_DIR="${SIDE_BY_SIDE_DIR:-${ROOT_DIR}/side_by_side}"
FINGERPRINT_DIR="${ROOT_DIR}/fingerprint"
MERGED_CKPT="${MERGED_CKPT:-${ROOT_DIR}/checkpoints/nopos_with_transferred_rel_head.ckpt}"

DATA_ROOT="${DATA_ROOT:-/home/data/olp_h_re10k_slot0/re10k_hf}"
FULL_INDEX="${FULL_INDEX:-assets/evaluation_index_re10k.json}"
QUAL_COUNT="${QUAL_COUNT:-10}"
QUAL_SEED="${QUAL_SEED:-0}"
QUAL_TARGETS_PER_SCENE="${QUAL_TARGETS_PER_SCENE:-1}"

REFERENCE_ROOT="${REFERENCE_ROOT:-outputs/reliability_residual_eval/nopos_olp_rel_compare/20260609_111408}"
REFERENCE_QUAL_INDEX="${REFERENCE_ROOT}/evaluation_index_re10k_qual${QUAL_COUNT}_seed${QUAL_SEED}.json"
if [[ -z "${QUAL_INDEX:-}" ]]; then
  if [[ -f "${REFERENCE_QUAL_INDEX}" ]]; then
    QUAL_INDEX="${REFERENCE_QUAL_INDEX}"
  else
    QUAL_INDEX="${ROOT_DIR}/evaluation_index_re10k_qual${QUAL_COUNT}_seed${QUAL_SEED}.json"
  fi
fi
QUAL_SOURCE_INDEX="${QUAL_SOURCE_INDEX:-${FULL_INDEX}}"

NOPOS_CKPT="${NOPOS_CKPT:-/workspace/olp_h_re10k_slot0/NoPoSplat/pretrained_weights/re10k.ckpt}"
REL_CKPT="${REL_CKPT:-outputs/reliability_residual_ft/re10k_olp_reliability_residual_ft_20260527_084014/checkpoints/epoch_1-step_75000.ckpt}"

TRANSFER_RUN_NAME="${TRANSFER_RUN_NAME:-re10k_nopos_transferred_rel}"
TRANSFER_QUAL_RUN_NAME="${TRANSFER_QUAL_RUN_NAME:-re10k_nopos_transferred_rel_qual${QUAL_COUNT}}"

RUN_FULL="${RUN_FULL:-1}"
RUN_QUAL="${RUN_QUAL:-1}"
RUN_SIDE_BY_SIDE="${RUN_SIDE_BY_SIDE:-1}"
FORCE_REMERGE="${FORCE_REMERGE:-0}"

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

mkdir -p "${FULL_ROOT}" "${QUAL_ROOT}" "${SIDE_BY_SIDE_DIR}" "${FINGERPRINT_DIR}" "$(dirname "${MERGED_CKPT}")"

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
require_file REL_CKPT "${REL_CKPT}"

if git rev-parse HEAD > "${FINGERPRINT_DIR}/git_commit.txt" 2>/dev/null; then
  git status --short > "${FINGERPRINT_DIR}/git_status.txt" || true
fi

cat > "${FINGERPRINT_DIR}/run_config.env" <<EOF
RUN_ID=${RUN_ID}
ROOT_DIR=${ROOT_DIR}
DATA_ROOT=${DATA_ROOT}
FULL_INDEX=${FULL_INDEX}
QUAL_INDEX=${QUAL_INDEX}
QUAL_COUNT=${QUAL_COUNT}
QUAL_SEED=${QUAL_SEED}
QUAL_TARGETS_PER_SCENE=${QUAL_TARGETS_PER_SCENE}
NOPOS_CKPT=${NOPOS_CKPT}
REL_CKPT=${REL_CKPT}
MERGED_CKPT=${MERGED_CKPT}
REFERENCE_ROOT=${REFERENCE_ROOT}
TRANSFER_RUN_NAME=${TRANSFER_RUN_NAME}
ALIGN_POSE=${ALIGN_POSE}
EOF

make_merged_checkpoint() {
  if [[ -f "${MERGED_CKPT}" && "${FORCE_REMERGE}" != "1" ]]; then
    echo "Using existing merged checkpoint: ${MERGED_CKPT}"
    return
  fi

  "${PYTHON_BIN}" - "${NOPOS_CKPT}" "${REL_CKPT}" "${MERGED_CKPT}" <<'PYMERGE'
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

nopos_path = Path(sys.argv[1])
rel_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])

nopos_ckpt = torch.load(nopos_path, map_location="cpu")
rel_ckpt = torch.load(rel_path, map_location="cpu")

nopos_sd = nopos_ckpt.get("state_dict", nopos_ckpt)
rel_sd = rel_ckpt.get("state_dict", rel_ckpt)

rel_keys = sorted(k for k in rel_sd if "reliability_head" in k)
if not rel_keys:
    raise SystemExit(f"No reliability_head weights found in {rel_path}")

copied = {}
for key in rel_keys:
    tensor = rel_sd[key]
    copied[key] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }
    nopos_sd[key] = tensor.detach().cpu().clone()

if "state_dict" in nopos_ckpt:
    nopos_ckpt["state_dict"] = nopos_sd

nopos_ckpt["transferred_reliability_head"] = {
    "base_checkpoint": str(nopos_path),
    "source_reliability_checkpoint": str(rel_path),
    "copied_keys": copied,
}

out_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(nopos_ckpt, out_path)

audit_path = out_path.with_suffix(".audit.json")
audit_path.write_text(json.dumps(nopos_ckpt["transferred_reliability_head"], indent=2) + "\n")
print(f"Wrote merged checkpoint: {out_path}")
print(f"Wrote merge audit: {audit_path}")
print(f"Copied reliability_head tensors: {len(rel_keys)}")
PYMERGE
}

make_random_qual_index() {
  if [[ -f "${QUAL_INDEX}" ]]; then
    echo "Using existing qualitative index: ${QUAL_INDEX}"
    return
  fi

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

print_run_summary() {
  local title="$1"
  local run_dir="$2"
  "${PYTHON_BIN}" - "${title}" "${run_dir}" <<'PYSUMMARY'
import json
import sys
from pathlib import Path

title = sys.argv[1]
run_dir = Path(sys.argv[2])
metrics_path = run_dir / "test_metrics.json"
stats_path = run_dir / "olp_stats_summary.json"

print(f"\n[{title}] {run_dir}")
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
        "opacity_reduction_mean",
        "reliability_mean",
        "reliability_below_095",
        "opacity_active_frac_gt_0p05",
        "opacity_compression_factor_gt_0p05",
        "opacity_after_compression_factor_gt_0p05",
    ]
    for key in keys:
        if key in stats:
            print(f"  {key}: {stats[key]:.6f}")
else:
    print("  olp_stats_summary.json: missing")
PYSUMMARY
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
  local index_path="$2"
  local output_root="$3"
  local limit_batches="$4"
  local save_image="$5"
  local save_compare="$6"
  local run_dir="${output_root}/${run_name}"
  mkdir -p "${run_dir}"

  local cmd=(
    "${PYTHON_BIN}" -m src.main
    +experiment=re10k
    mode=test
    "wandb.mode=${WANDB_MODE}"
    "wandb.name=${run_name}"
    dataset/view_sampler@dataset.re10k.view_sampler=evaluation
    "dataset.re10k.view_sampler.index_path=${index_path}"
    "dataset.re10k.roots=[${DATA_ROOT}]"
    "checkpointing.load=${MERGED_CKPT}"
    model.encoder.orthogonal_preconditioner.enabled=false
    model.encoder.orthogonal_preconditioner.mode=hadamard
    model.encoder.orthogonal_preconditioner.dim=256
    model.encoder.orthogonal_preconditioner.mix=0.0
    model.encoder.orthogonal_preconditioner.target_mix=0.0
    model.encoder.orthogonal_preconditioner.warmup_steps=0
    "model.encoder.orthogonal_preconditioner.seed=${OLP_SEED}"
    model.encoder.orthogonal_preconditioner.collect_stats=true
    model.encoder.gaussian_reliability.enabled=true
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
  print_run_summary "${run_name}" "${run_dir}"
}

run_side_by_side() {
  local nopos_dir="${REFERENCE_ROOT}/qual_eval/re10k_nopos_qual${QUAL_COUNT}"
  local olp_dir="${REFERENCE_ROOT}/qual_eval/re10k_olp300k_qual${QUAL_COUNT}"
  local rel_dir="${REFERENCE_ROOT}/qual_eval/re10k_relctx_75k_qual${QUAL_COUNT}"
  local transfer_dir="${QUAL_ROOT}/${TRANSFER_QUAL_RUN_NAME}"

  require_dir "reference NoPo qualitative output" "${nopos_dir}"
  require_dir "reference OLP qualitative output" "${olp_dir}"
  require_dir "reference OLP+Rel qualitative output" "${rel_dir}"

  local metrics_path="${SIDE_BY_SIDE_DIR}/metrics.json"
  local cmd=(
    "${PYTHON_BIN}" -u -m src.scripts.compute_metrics
    +experiment=re10k
    +evaluation=re10k
    "dataset.re10k.view_sampler.index_path=${QUAL_INDEX}"
    "dataset.re10k.roots=[${DATA_ROOT}]"
    "evaluation.methods=[{name:NoPoSplat,key:nopos,path:${nopos_dir}},{name:OLP-FT,key:olp,path:${olp_dir}},{name:OLP-Reliability,key:rel,path:${rel_dir}},{name:NoPo-Transferred-Rel,key:nopos_rel_transfer,path:${transfer_dir}}]"
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

make_merged_checkpoint

if [[ "${RUN_FULL}" == "1" ]]; then
  run_eval "${TRANSFER_RUN_NAME}" "${FULL_INDEX}" "${FULL_ROOT}" "${FULL_LIMIT_TEST_BATCHES}" false false
fi

if [[ "${RUN_QUAL}" == "1" ]]; then
  make_random_qual_index
  run_eval "${TRANSFER_QUAL_RUN_NAME}" "${QUAL_INDEX}" "${QUAL_ROOT}" "${QUAL_LIMIT_TEST_BATCHES}" true true
fi

if [[ "${RUN_SIDE_BY_SIDE}" == "1" ]]; then
  run_side_by_side
fi

print_metric_summary \
  "TRANSFER_ABLATION_SUMMARY" \
  "NoPo+TransferredRelFull=${FULL_ROOT}/${TRANSFER_RUN_NAME}" \
  "NoPo+TransferredRelQual=${QUAL_ROOT}/${TRANSFER_QUAL_RUN_NAME}"

cat <<EOF

Done.
Root: ${ROOT_DIR}
Merged checkpoint: ${MERGED_CKPT}
Full eval output: ${FULL_ROOT}/${TRANSFER_RUN_NAME}
Qual eval output: ${QUAL_ROOT}/${TRANSFER_QUAL_RUN_NAME}
Qual index: ${QUAL_INDEX}
Side-by-side panels: ${SIDE_BY_SIDE_DIR}/panels
EOF
