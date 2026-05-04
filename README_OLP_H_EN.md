# NoPoSplat OLP-H A100 Reproducible Experiment

This is an additive experiment layer on top of `cvg/NoPoSplat`.

Main references:

- Korean difference report: `docs/OLP_H_DIFF_REPORT_KO.md`
- English difference report: `docs/OLP_H_DIFF_REPORT_EN.md`
- A100 environment lock: `env/requirements-a100.lock`
- A100 environment recipe: `env/environment-a100.yml`

The default config keeps OLP-H disabled, so original NoPoSplat behavior is
preserved unless a run script explicitly enables it.

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
