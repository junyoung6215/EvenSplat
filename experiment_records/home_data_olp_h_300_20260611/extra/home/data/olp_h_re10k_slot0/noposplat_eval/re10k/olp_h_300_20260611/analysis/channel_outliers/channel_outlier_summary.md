# OLP-H channel outlier diagnostic

## What Was Measured

- `olp_score = max_c |y_c| / rms(y)`: how strongly one OLP output channel dominates at a spatial location.
- `top1_channel`: the channel index with the largest absolute value at that location.
- `top1_value`: the signed value of that dominant channel.
- `topk_energy_ratio`: how concentrated the channel energy is in the top-k channels.

## Scene-Level Summary

- `latent_output_olp_score_mean`: mean 7.881, median 7.847, p95 9.089
- `latent_output_olp_score_p95`: mean 10.61, median 10.69, p95 11.54
- `latent_output_olp_score_p99`: mean 11.28, median 11.35, p95 11.99
- `latent_output_top1_channel_mode_freq`: mean 0.736, median 0.7321, p95 0.9193
- `latent_output_top1_value_abs_p99`: mean 3.795, median 3.552, p95 5.593
- `latent_output_topk_energy_ratio_mean`: mean 0.4951, median 0.4969, p95 0.5621

## Dominant Channel Pattern

Scene-level mode channels:
- channel 200: 159/300 scenes (53.0%)
- channel 91: 74/300 scenes (24.7%)
- channel 120: 56/300 scenes (18.7%)
- channel 229: 11/300 scenes (3.7%)

Dense full-map top1 histogram:
- module 1214 rank 1: channel 244 (53.30% of pixels)
- module 1214 rank 2: channel 27 (41.41% of pixels)
- module 1214 rank 3: channel 55 (0.94% of pixels)
- module 1214 rank 4: channel 180 (0.89% of pixels)
- module 1214 rank 5: channel 239 (0.57% of pixels)
- module 1305 rank 1: channel 156 (65.54% of pixels)
- module 1305 rank 2: channel 214 (29.96% of pixels)
- module 1305 rank 3: channel 196 (1.54% of pixels)
- module 1305 rank 4: channel 120 (0.85% of pixels)
- module 1305 rank 5: channel 170 (0.73% of pixels)

## Relation With Gaussian / Quality Metrics

- `latent_output_olp_score_mean__vs__opacity_mean`: r=-0.237
- `latent_output_olp_score_mean__vs__floater_proxy`: r=-0.001
- `latent_output_olp_score_mean__vs__lpips_ours`: r=-0.323
- `latent_output_olp_score_p99__vs__opacity_p99`: r=-0.037
- `latent_output_topk_energy_ratio_mean__vs__floater_proxy`: r=-0.050

## Interpretation For Lab Meeting

- OLP-H does have measurable channel outliers, and the outlier is not uniformly spread across all channels.
- A small set of channels repeatedly appears as the dominant `top1_channel`; this suggests the OLP transform creates channel-wise concentration rather than random noise.
- The next question is whether these dominant channels are suppressing low-utility Gaussians or creating failure cases in specific scenes.
- For the next experiment, analyze top outlier scenes qualitatively and optionally mask/prune the most dominant channels at inference time to test causality.

