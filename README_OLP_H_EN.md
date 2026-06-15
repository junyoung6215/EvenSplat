# NoPoSplat OLP-H CUDA 11.8 Reproducible Experiment

This is an experiment layer on top of `cvg/NoPoSplat`. The OLP-H core logic is
kept in new modules, while existing NoPoSplat files contain only thin wiring
hooks for config propagation and the Gaussian-head call site.

Main references:

- Korean difference report: `docs/OLP_H_DIFF_REPORT_KO.md`
- English difference report: `docs/OLP_H_DIFF_REPORT_EN.md`
- GitHub experiment archive guide: `docs/GITHUB_EXPERIMENT_ARCHIVE_KO.md`
- CUDA 11.8 environment lock: `env/requirements-cuda118.lock`
- CUDA 11.8 environment recipe: `env/environment-cuda118.yml`

The default config keeps OLP-H disabled, so original NoPoSplat behavior is
preserved unless a run script explicitly enables it.
`scripts/run_re10k_original_eval.sh` keeps OLP-H off for official reproduction.
Baseline latent stats are collected by the fine-tuning control or
`scripts/run_re10k_collect_stats.sh` with `mix=0.0` identity mode.

## Minimal CUDA 11.8 Flow

```bash
scripts/create_cuda118_env.sh
conda activate noposplat-olp-cuda118

scripts/run_re10k_smoke.sh
scripts/run_re10k_original_eval.sh
scripts/run_re10k_ft_control.sh
scripts/run_re10k_olp_ft.sh
```

Use a slot-specific environment name on shared servers:

```bash
ENV_NAME=noposplat-olp-cuda118-slot0 \
bash scripts/create_cuda118_env.sh

conda activate noposplat-olp-cuda118-slot0
```

`scripts/create_cuda118_env.sh` reflects the server-side CUDA extension fixes:
install PyTorch `2.1.2+cu118` first, build `diff_gaussian_rasterization`
with CUDA 11.8 nvcc/header and gcc-11, then install the remaining locked
packages with `--no-deps`. This preserves the torch pin even though
`e3nn==0.6.0` advertises `torch>=2.2.0` in package metadata.

After fine-tuning:

```bash
BASELINE_CKPT=/path/to/re10k_ft_control.ckpt \
OLP_CKPT=/path/to/re10k_olp_ft.ckpt \
scripts/run_re10k_collect_stats.sh
```

To collect only GitHub-friendly analysis artifacts from `outputs/olp_re10k`:

```bash
python scripts/archive_re10k_experiment_artifacts.py \
  --name re10k_olp_h_main_2026 \
  --include-images \
  --max-images 12
```

The archive script copies small metric, config, command, fingerprint, and log
files into `experiment_records/`. Smoke/sanity outputs, checkpoints, and
datasets remain excluded by default.

## OLP-H Core Settings

`RandomizedHadamardPreconditioner` applies a fixed orthogonal transform
`Q = H D P / sqrt(C)` to the feature map immediately before the Gaussian
parameter head. `H` is a Walsh-Hadamard transform, `D` is a random sign
diagonal, and `P` is a channel permutation. The normalized Hadamard transform
preserves L2 norm when `C` is a power of two. NoPoSplat `dpt_gs` uses `C=256`,
so the assumption holds for the main experiment.

Recommended settings:

- Original reproduction: `enabled=false`
- Baseline latent stats: `enabled=true`, `mix=0.0`, `target_mix=0.0`,
  `warmup_steps=0`, `collect_stats=true`
- Zero-shot diagnostic: `enabled=true`, `mix=1.0`, `target_mix=1.0`,
  `warmup_steps=0`
- OLP-H fine-tuning: `enabled=true`, `mix=0.0`, `target_mix=1.0`,
  `warmup_steps=5000`, `seed=0`

References:

- Hadamard transform: https://en.wikipedia.org/wiki/Hadamard_transform
- Detailed diff report: `docs/OLP_H_DIFF_REPORT_EN.md`

## Tests

OLP-H unit tests live in `tests/test_olp_preconditioner.py`. They isolated-load
the OLP modules so they can run without the full NoPoSplat dataset dependency
tree.

```bash
PYTHONPYCACHEPREFIX=/private/tmp/noposplat_pycache \
python3 tests/test_olp_preconditioner.py
```

Covered behavior:

- disabled mode returns the original tensor object
- Hadamard preconditioning preserves L2 norm at `mix=1`
- warm-up schedule moves from `mix` to `target_mix`
- `collect_stats=true` stores latent stats
- the same `seed` reproduces the same rotation
- random sign/permutation buffers are excluded from checkpoint `state_dict`
- Gaussian tail stats include `floater_proxy`
