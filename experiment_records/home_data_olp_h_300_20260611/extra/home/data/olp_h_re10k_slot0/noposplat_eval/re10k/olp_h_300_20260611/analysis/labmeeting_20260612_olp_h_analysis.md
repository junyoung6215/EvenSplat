# 2026-06-12 Lab Meeting: OLP-H Analysis Summary

## Main Claim

OLP-H should not be presented as simply improving image quality over NoPoSplat. The stronger current evidence is that OLP-H changes the Gaussian distribution: it lowers opacity/active Gaussian ratios and floater proxy, and it appears more robust when the render-time Gaussian budget is reduced. Channel outlier analysis shows strong channel concentration, but this concentration is not yet proven to be the direct cause of floater reduction.

## Terminology

- NoPoSplat control: baseline model evaluated on the same selected scenes.
- OLP-H: NoPoSplat with the OLP-H head / Hadamard preconditioner checkpoint.
- Active Gaussian ratio: fraction of generated Gaussians whose opacity is above a threshold, e.g. opacity > 0.10.
- Gaussian budget limit: render-time top-K selection by opacity. This is post-hoc render-time pruning, not retraining.
- Floater proxy: fraction of Gaussians that are both high-opacity and large-scale under the chosen tail thresholds. Lower value is interpreted as fewer likely floating artifacts, but it is still a proxy.
- Channel outlier: a spatial location where one OLP output channel dominates the channel vector.

## Experiment 1. Matched Gaussian Statistics

### Purpose

Check whether OLP-H changes Gaussian statistics on matched scenes before claiming any quality improvement.

### Method

- Use matched 300-scene subset.
- Compare Gaussian statistics between NoPoSplat control and OLP-H.
- Use quality numbers carefully because some diagnostic runs use different pose-alignment settings. Quality comparison should primarily use full-eval or cap-curve runs with matching settings.

### Result

| Metric | NoPoSplat | OLP-H | Direction |
| --- | ---: | ---: | --- |
| Opacity mean | 0.391 | 0.347 | OLP-H lower |
| active_ratio@0.10 | 0.897 | 0.842 | OLP-H lower |
| active_ratio@0.30 | 0.621 | 0.534 | OLP-H lower |
| active_ratio@0.50 | 0.328 | 0.269 | OLP-H lower |
| Floater proxy | 2.95e-4 | 8.16e-5 | OLP-H lower |

### Interpretation

OLP-H uses fewer high-opacity Gaussians and strongly reduces the floater proxy. This supports the direction that OLP-H affects Gaussian compactness / distribution, not just pixel-level quality.

## Experiment 2. Active Gaussian Threshold

### Purpose

Generated Gaussian count can be the same, but the number of effectively used Gaussians may differ. This experiment checks how many Gaussians remain active under several opacity thresholds.

### Method

- Thresholds: opacity > 0.05, 0.10, 0.30, 0.50, 0.70.
- Compute active ratio per scene and average over the matched subset.
- Focus on the trend across thresholds, not one arbitrary cutoff.

### Result

The clearest thresholds for presentation are 0.10, 0.30, and 0.50:

| Threshold | NoPoSplat | OLP-H | Delta |
| --- | ---: | ---: | ---: |
| active_ratio@0.10 | 0.897 | 0.842 | -0.055 |
| active_ratio@0.30 | 0.621 | 0.534 | -0.086 |
| active_ratio@0.50 | 0.328 | 0.269 | -0.060 |

### Interpretation

OLP-H does not merely lower extreme opacity tails; it lowers active Gaussian usage across practical opacity thresholds. This is a compactness signal.

## Experiment 3. Gaussian Budget Limit / Cap Curve

### Purpose

Test whether OLP-H remains usable when the renderer is allowed to use only a limited number of Gaussians. If OLP-H is genuinely more compact, quality should degrade more slowly under the same render-time budget.

### Method

- Sort Gaussians by opacity at render time.
- Keep top-K according to budget: 100%, 75%, 50%, 25%, 10%.
- Render with the retained Gaussians and compute PSNR / SSIM / LPIPS.
- Use no pose alignment for both models in this experiment, so the cap curve comparison is fair within this experiment.

### Result

| Budget | NoPoSplat PSNR | OLP-H PSNR | NoPo dPSNR | OLP-H dPSNR |
| --- | ---: | ---: | ---: | ---: |
| 100% | 23.61 | 22.56 | 0.00 | 0.00 |
| 75% | 16.71 | 18.03 | -6.90 | -4.52 |
| 50% | 11.39 | 12.03 | -12.22 | -10.53 |
| 25% | 7.82 | 8.11 | -15.79 | -14.45 |
| 10% | 6.15 | 6.32 | -17.46 | -16.23 |

### Interpretation

OLP-H has lower absolute no-align PSNR at 100%, but it degrades less under budget restriction. This supports the hypothesis that OLP-H makes the representation more pruning-friendly / budget-robust.

## Experiment 4. Channel Outlier Diagnostic

### Purpose

Understand whether OLP-H behavior is related to channel-wise concentration in the OLP output.

### Method

For each spatial location in the OLP output:

- `olp_score = max_c |y_c| / rms(y)`: dominance of the largest channel relative to RMS.
- `top1_channel`: index of the largest absolute-value channel.
- `top1_value`: signed value of that channel.
- `topk_energy_ratio`: energy concentration in the top-k channels.

Two outputs were saved:

- Scene-level CSV summary: `olp_stats_per_scene.csv`.
- Full-map raw dumps: 300 `.pt` files with per-pixel OLP maps and per-Gaussian opacity / scale_max / depth.

### Result

Scene-level channel concentration:

| Metric | Mean | p95 |
| --- | ---: | ---: |
| OLP score mean | 7.881 | 9.089 |
| OLP score p99 | 11.276 | 11.992 |
| top1 channel mode frequency | 0.736 | 0.919 |
| top-k energy ratio mean | 0.495 | 0.562 |

Dominant scene-level mode channels:

| Channel | Scene count | Ratio |
| --- | ---: | ---: |
| 200 | 159 / 300 | 53.0% |
| 91 | 74 / 300 | 24.7% |
| 120 | 56 / 300 | 18.7% |
| 229 | 11 / 300 | 3.7% |

Dense full-map top1 channel histogram:

| Module | Top channels | Interpretation |
| --- | --- | --- |
| module 1214 | ch244 53.3%, ch27 41.4% | almost all pixels dominated by two channels |
| module 1305 | ch156 65.5%, ch214 30.0% | almost all pixels dominated by two channels |

Correlation with Gaussian / quality metrics:

| Relation | Pearson r | Takeaway |
| --- | ---: | --- |
| OLP score mean vs opacity mean | -0.237 | weak-to-moderate negative |
| OLP score mean vs floater proxy | -0.001 | almost no direct relation |
| OLP score mean vs LPIPS | -0.323 | moderate negative, but not causal |
| top-k energy ratio mean vs floater proxy | -0.050 | weak relation |
| top-k energy ratio mean vs LPIPS | -0.339 | moderate negative, but not causal |

### Interpretation

Channel outliers clearly exist and are highly concentrated in a few channels. However, the scene-level correlation with floater proxy is weak. Therefore, the current evidence says channel concentration is a real OLP-H phenomenon, but it does not yet prove that these outlier channels directly cause floater reduction.

## Visual Materials

Use these figures for PPT / Notion.

| Figure | Purpose | Path |
| --- | --- | --- |
| Active Gaussian compactness | Show lower active ratios and floater proxy | `/home/data/olp_h_re10k_slot0/noposplat_eval/re10k/olp_h_300_20260611/analysis/figures/active_gaussian_compactness.png` |
| Gaussian cap curve | Show budget robustness | `/home/data/olp_h_re10k_slot0/noposplat_eval/re10k/olp_h_300_20260611/analysis/figures/gaussian_cap_curve_psnr_lpips.png` |
| Channel top1 histogram | Show dominant channels | `/home/data/olp_h_re10k_slot0/noposplat_eval/re10k/olp_h_300_20260611/analysis/figures/channel_top1_histogram.png` |
| Channel relation scatter | Show weak/direct relation limits | `/home/data/olp_h_re10k_slot0/noposplat_eval/re10k/olp_h_300_20260611/analysis/figures/channel_outlier_relationships.png` |
| Channel outlier map example | Visualize OLP score / top1 channel / energy map | `/home/data/olp_h_re10k_slot0/noposplat_eval/re10k/olp_h_300_20260611/analysis/figures/channel_outlier_example_maps.png` |

## Suggested Lab Meeting Narrative

1. Full-eval quality alone is not the selling point: OLP-H is lower than NoPoSplat in standard full-eval quality.
2. The meaningful signal is in Gaussian behavior: opacity mean, active Gaussian ratios, and floater proxy are lower.
3. The cap curve suggests OLP-H may be more robust under limited Gaussian budget, which is closer to a compactness claim.
4. Channel outlier analysis shows strong channel concentration, but current correlation does not prove causality.
5. Next step should test causality by perturbing or masking dominant channels and measuring Gaussian statistics / quality changes.

## Next Experiments

### 1. Dominant channel masking / ablation

- Mask or scale down dominant channels at inference time.
- Candidate full-map dominant channels: module 1214 channels 244 and 27; module 1305 channels 156 and 214.
- Measure opacity mean, active ratios, floater proxy, PSNR / SSIM / LPIPS.
- Goal: test whether channel concentration is causal or just a byproduct.

### 2. Outlier scene qualitative inspection

- Select top scenes by `latent_output_olp_score_p99`, `top1_channel_mode_freq`, and LPIPS.
- Compare rendered image, OLP score heatmap, top1 channel map, opacity map, and scale/depth map.
- Goal: identify whether outlier channels correspond to object boundaries, background, depth discontinuities, or failure regions.

### 3. Channel-conditioned Gaussian statistics

- Bin pixels/Gaussians by OLP score or top1 channel.
- Compare opacity, scale_max, depth, and floater proxy inside high-score vs normal regions.
- Goal: connect channel outlier maps to Gaussian geometry more directly than scene-level correlation.

### 4. Budget robustness on larger subset

- Repeat cap curve on more scenes if time allows.
- Keep pose alignment policy fixed across models.
- Goal: confirm that the budget robustness trend is stable beyond 300 scenes.

## Source Artifacts

- `summary_missing300.json`: `/home/data/olp_h_re10k_slot0/noposplat_eval/re10k/olp_h_300_20260611/analysis/summary_missing300.json`
- `gaussian_cap_curve_summary.csv`: `/home/data/olp_h_re10k_slot0/noposplat_eval/re10k/olp_h_300_20260611/analysis/gaussian_cap_curve_summary.csv`
- `channel_outlier_summary.md`: `/home/data/olp_h_re10k_slot0/noposplat_eval/re10k/olp_h_300_20260611/analysis/channel_outliers/channel_outlier_summary.md`
- `channel_outlier_summary.json`: `/home/data/olp_h_re10k_slot0/noposplat_eval/re10k/olp_h_300_20260611/analysis/channel_outliers/channel_outlier_summary.json`
- raw full maps: `/home/data/olp_h_re10k_slot0/noposplat_eval/re10k/olp_h_300_20260611/test_outputs/re10k_olp_h_rawfull_metric300_20260611/olp_raw_full_maps`
