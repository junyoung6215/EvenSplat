# OLP-H channel diagnostic summary

- scene CSV rows: 300
- raw full map files: 300

## Scene-level channel summaries

- `latent_output_olp_score_mean` mean=7.88128, p50=7.84785, p95=9.08918
- `latent_output_olp_score_p95` mean=10.6148, p50=10.7029, p95=11.5399
- `latent_output_olp_score_p99` mean=11.2763, p50=11.3524, p95=11.9921
- `latent_output_top1_channel_mode` mean=159.46, p50=200, p95=200
- `latent_output_top1_channel_mode_freq` mean=0.73602, p50=0.732681, p95=0.919304
- `latent_output_top1_value_abs_mean` mean=1.71753, p50=1.64403, p95=2.32139
- `latent_output_top1_value_abs_p95` mean=2.91545, p50=2.7091, p95=4.41093
- `latent_output_top1_value_abs_p99` mean=3.79472, p50=3.55245, p95=5.59291
- `latent_output_topk_energy_ratio_mean` mean=0.495147, p50=0.497797, p95=0.562085
- `latent_output_topk_energy_ratio_p95` mean=0.637916, p50=0.644698, p95=0.689536
- `latent_output_topk_energy_ratio_p99` mean=0.673589, p50=0.678465, p95=0.716052

## Top scene-level mode channels

- channel 200: 159/300 scenes
- channel 91: 74/300 scenes
- channel 120: 56/300 scenes
- channel 229: 11/300 scenes

## Dense raw-map top1 channel histogram

- module 0 rank 1: channel 244 ratio=0.5330
- module 0 rank 2: channel 27 ratio=0.4141
- module 0 rank 3: channel 55 ratio=0.0094
- module 0 rank 4: channel 180 ratio=0.0089
- module 0 rank 5: channel 239 ratio=0.0057
- module 1 rank 1: channel 156 ratio=0.6554
- module 1 rank 2: channel 214 ratio=0.2996
- module 1 rank 3: channel 196 ratio=0.0154
- module 1 rank 4: channel 120 ratio=0.0085
- module 1 rank 5: channel 170 ratio=0.0073
