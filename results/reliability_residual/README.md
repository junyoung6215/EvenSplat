# Reliability Residual Results

This directory contains lightweight, GitHub-friendly artifacts copied from the
full local experiment outputs. Model checkpoints and large raw dumps are
excluded so that weights can be published separately on Hugging Face.

Included artifacts:

- `test_metrics.json`
- `benchmark.json`
- `peak_memory.json`
- `olp_stats_summary.json`
- `olp_stats_summary_by_overlap.json`
- `command.sh`
- `run_config.env`
- `fingerprint/*`
- Hydra `config.yaml` and `overrides.yaml`
- training `pretrained_load_audit.json`

Excluded artifacts:

- `*.ckpt` and `*.torch` weights
- per-scene stats dumps
- stdout/main logs
- qualitative image folders
- smoke, sanity, and partial-run outputs

Full local outputs remain under ignored `outputs/` paths.

## Runs

| Run | Source output | PSNR | SSIM | LPIPS |
| --- | --- | ---: | ---: | ---: |
| OLP 300k baseline full | `outputs/reliability_residual_eval/20260603_094830/re10k_olp300k_baseline_full` | 23.802832 | 0.793890 | 0.181047 |
| Reliability residual 75k full | `outputs/reliability_residual_eval/20260603_094830/re10k_relctx_75k_full` | 24.842695 | 0.823976 | 0.163165 |
| Reliability residual 75k vis10 | `outputs/reliability_residual_eval/relctx_vis10_compare/re10k_relctx_75k_vis10` | 26.759022 | 0.879473 | 0.113874 |
| NoPoS full comparison | `outputs/reliability_residual_eval/nopos_olp_rel_compare/20260609_111408/full_eval/re10k_nopos_full` | 25.057571 | 0.839247 | 0.161374 |
| NoPoS transferred reliability head | `outputs/reliability_residual_eval/nopos_transferred_rel/20260611_073556/full_eval/re10k_nopos_transferred_rel` | 24.763098 | 0.836649 | 0.164011 |

## Weight Handling

Checkpoints are intentionally not committed here. Upload them to Hugging Face
and reference the model repo from the project README or release notes.
