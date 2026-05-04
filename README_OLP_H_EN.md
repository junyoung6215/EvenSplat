# NoPoSplat OLP-H A100 Reproducible Experiment

This is an experiment layer on top of `cvg/NoPoSplat`. The OLP-H core logic is
kept in new modules, while existing NoPoSplat files contain only thin wiring
hooks for config propagation and the Gaussian-head call site.

Main references:

- Korean difference report: `docs/OLP_H_DIFF_REPORT_KO.md`
- English difference report: `docs/OLP_H_DIFF_REPORT_EN.md`
- A100 environment lock: `env/requirements-a100.lock`
- A100 environment recipe: `env/environment-a100.yml`

The default config keeps OLP-H disabled, so original NoPoSplat behavior is
preserved unless a run script explicitly enables it.
`scripts/run_re10k_original_eval.sh` keeps OLP-H off for official reproduction.
Baseline latent stats are collected by the fine-tuning control or
`scripts/run_re10k_collect_stats.sh` with `mix=0.0` identity mode.

## Minimal A100 Flow

```bash
scripts/create_a100_env.sh
conda activate noposplat-olp-a100

scripts/run_re10k_smoke.sh
scripts/run_re10k_original_eval.sh
scripts/run_re10k_ft_control.sh
scripts/run_re10k_olp_ft.sh
```

After fine-tuning:

```bash
BASELINE_CKPT=/path/to/re10k_ft_control.ckpt \
OLP_CKPT=/path/to/re10k_olp_ft.ckpt \
scripts/run_re10k_collect_stats.sh
```

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
