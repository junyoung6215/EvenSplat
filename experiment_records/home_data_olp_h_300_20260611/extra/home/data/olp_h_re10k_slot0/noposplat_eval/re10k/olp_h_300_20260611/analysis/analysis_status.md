# OLP-H 300-scene analysis status

- exp_root: `/home/data/olp_h_re10k_slot0/noposplat_eval/re10k/olp_h_300_20260611`
- metric_index: `/home/data/olp_h_re10k_slot0/noposplat_eval/re10k/olp_h_300_20260611/evaluation_index_re10k_metric300.json`
- olp_csv: `/home/data/olp_h_re10k_slot0/noposplat_eval/re10k/re10k_eval_compare/compare_20260517_300k_final/test_outputs/re10k_olp_metric2000/olp_stats_per_scene.csv`
- baseline_csv: `/home/data/olp_h_re10k_slot0/noposplat_eval/re10k/olp_h_300_20260611/test_outputs/re10k_noposplat_ft_control_metric300/olp_stats_per_scene.csv`

## Status

- selected_scenes: 300
- olp_h_rows_in_subset: 300
- baseline_rows_in_subset: 300
- paired_rows: 300

## Not yet available in current CSV

- `latent_output_olp_score_mean`, `latent_output_olp_score_p95`, `latent_output_olp_score_p99`, `latent_output_top1_channel_mode`, `latent_output_top1_value_abs_p99`, `latent_output_topk_energy_ratio_mean`, `latent_output_topk_energy_ratio_p99`

These require per-Gaussian or per-pixel dump instrumentation. The current stats support opacity/scale/volume/latent aggregate analysis.

## Mean deltas, OLP-H minus baseline

- `psnr_ours`: -1.34356
- `ssim_ours`: -0.0471335
- `lpips_ours`: 0.020295
- `opacity_mean`: -0.0439275
- `opacity_p95`: -0.0274076
- `opacity_p99`: -0.00700423
- `opacity_tail_095`: 6.68589e-05
- `scale_max_mean`: 0.00164221
- `scale_max_p95`: -0.00209597
- `scale_max_p99`: -0.00329632
- `scale_tail`: 0.00116676
- `floater_proxy`: -0.000212962
- `volume_p95`: 9.91919e-05
- `volume_p99`: 0.000284434

## OLP-H latent/output correlations

- `latent_output_max_rms__vs__opacity_mean`: -0.2369
- `latent_output_max_rms__vs__opacity_p95`: -0.1087
- `latent_output_max_rms__vs__opacity_p99`: -0.0944
- `latent_output_max_rms__vs__opacity_tail_095`: -0.1445
- `latent_output_max_rms__vs__scale_max_p95`: 0.3635
- `latent_output_max_rms__vs__scale_max_p99`: 0.3103
- `latent_output_max_rms__vs__psnr_ours`: 0.4249
- `latent_output_max_rms__vs__lpips_ours`: -0.3234
- `latent_output_kurtosis__vs__opacity_p99`: -0.0471
- `latent_output_kurtosis__vs__scale_max_p99`: 0.3556

## Outputs

- `analysis/olp_h_metric300_subset.csv`
- `analysis/baseline_metric300_subset.csv` when baseline run is done
- `analysis/paired_metric300.csv` when baseline run is done
- `analysis/summary_metric300.json`
- `analysis/outliers_*.csv`

