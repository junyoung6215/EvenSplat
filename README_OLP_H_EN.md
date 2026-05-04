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
