# Method

This repository builds a compact feed-forward 3DGS model on top of
[NoPoSplat](https://github.com/cvg/NoPoSplat). NoPoSplat predicts one Gaussian
per input pixel (2 views × 256 × 256 = 131K Gaussians). We add three pieces:

1. **OLP-H**: an orthogonal latent preconditioner in front of the Gaussian head.
2. **Reliability head r**: a per-Gaussian gate on opacity.
3. **Stratified retention with an r filter**, trained with pruning-aware
   fine-tuning, so the model renders with 25 to 75% of its Gaussians.

All three are disabled by default. With every switch off, the code runs the
original NoPoSplat model and reproduces its checkpoints exactly.

## 1. OLP-H (`src/model/encoder/heads/olp.py`)

`RandomizedHadamardPreconditioner` applies a fixed orthogonal transform to the
DPT feature map just before the Gaussian parameter head:

```text
Q x = H D P x / sqrt(C)
```

`P` is a fixed channel permutation, `D` a fixed random sign diagonal and `H`
the Walsh-Hadamard transform (`C = 256` channels, a power of two, so `Q` is
orthogonal). `P` and `D` come from `seed` and are not stored in checkpoints.
`mix` blends identity and `Q`, ramping to `target_mix` over `warmup_steps`.
The transform spreads outlier channels across all channels, which reduces
floaters, at a PSNR cost that the reliability head recovers.

```yaml
model.encoder.orthogonal_preconditioner:
  enabled: true
  mix: 0.0          # 1.0 once trained
  target_mix: 1.0
  warmup_steps: 1500
  seed: 0
```

Latent and Gaussian tail statistics for analysis live in
`src/model/encoder/heads/latent_stats.py` and `src/analysis/`
(`test.save_olp_stats=true`).

## 2. Reliability head (`src/model/encoder/heads/reliability.py`, `dpt_gs_head.py`)

A separate branch on the preconditioned feature predicts one gate per Gaussian:

```text
r_g = r_min + (1 - r_min) * sigmoid(a_g)        r_min = 0.05, r_init = 0.95
opacity'_g = opacity_g * r_g
```

It is trained with a weak floater signal (`src/loss/loss_reliability.py`):
the predicted Gaussians are rendered from the context cameras, the per-pixel
residual against the input images is sampled at each Gaussian's projection,
and

```text
L_rel = mean_g [ r_g * sg(opacity_g) * sg(residual_risk_g) ]
```

pushes r down where the context reconstruction is poor (weight 1e-4, linear
warm-up over steps 3K to 8K). Gradients go mainly to the new head, so the
existing opacity, scale and color predictions are preserved.

## 3. Stratified retention with an r filter (`src/model/retention/`)

Retention keeps `keep_ratio` of the predicted Gaussians before rendering.
Selection indices come from detached scores; gradients still flow through the
kept Gaussians.

- **Stratified top-k** (`cells.py`). A global top-k by any score removes whole
  regions of the image. Instead, the source pixel grid of each view is split
  into `cell_size × cell_size` cells and every cell keeps the same fraction of
  its Gaussians. With 2 × 2 cells and 50%, every cell keeps 2 of its 4 pixels.
- **r filter** (`scores.py`, `score_source: r_filter`). Inside a cell,
  Gaussians with `r < r_threshold` (0.35) are dropped first and the rest of the
  budget is filled at random. Ranking strictly by r was slightly worse.
- **Pruning-aware fine-tuning**. Training renders only the kept Gaussians, so
  the network learns to cover the image with the retained subset. This is the
  step that recovers quality (see the ablation below).

```yaml
model.gaussian_retention:
  enabled: true
  score_source: r_filter   # opacity | base_opacity | reliability | random | r_filter | grid
  keep_ratio: 0.5
  cell_size: 2             # 0 = one global top-k
  r_threshold: 0.35
```

Other score sources are kept for ablations: `grid` keeps fixed Bayer-ordered
pixel positions in each cell; `train_keep_ratio_min > 0` draws a new budget
every training step so one model serves several budgets; `budget_weight > 0`
adds a loss that trains r itself to rank for the budget (`budget.py`, the
"learned r" ablation). Learned r and random per-step budgets were below the
r filter. `grid` is above it on the full test set: 25.35 / 0.839 / 0.159 at
50% and 24.84 / 0.824 / 0.174 at 25% (r filter: 25.01 and 24.62 dB).

## Results (RE10K test, 5,590 evaluable scenes, PSNR / SSIM / LPIPS)

| Model | #Gaussians | PSNR | SSIM | LPIPS |
| --- | --- | --- | --- | --- |
| NoPoSplat (paper) | 131K | 25.03 | 0.838 | 0.160 |
| Base: OLP-H + r | 131K | 24.84 | 0.824 | 0.163 |
| Ours 75% | 98K | 25.26 | 0.837 | 0.161 |
| Ours 50% | 66K | 25.01 | 0.829 | 0.169 |
| Ours 25% | 33K | 24.62 | 0.817 | 0.184 |

Peak render memory at 50%: 167 MB → 87 MB.

Ablation at 50% (66K Gaussians):

| Configuration | PSNR | SSIM | LPIPS |
| --- | --- | --- | --- |
| r, global top-k, no fine-tuning | 14.39 | 0.529 | 0.390 |
| + 2 × 2 stratified | 19.16 | 0.625 | 0.354 |
| + pruning-aware FT, 2K steps | 24.78 | 0.824 | 0.175 |
| + r filter | 24.88 | 0.826 | 0.172 |
| + FT 10K steps (final) | 25.01 | 0.829 | 0.169 |
| Full-count FT 10K, then select (no pruning in training) | 19.13 | 0.635 | 0.348 |

First-semester modules, all Gaussians kept:

| Configuration | PSNR | SSIM | LPIPS |
| --- | --- | --- | --- |
| NoPoSplat (our evaluation) | 25.06 | 0.839 | 0.161 |
| + OLP-H (300K steps) | 23.80 | 0.794 | 0.181 |
| + OLP-H + reliability head (75K steps) | 24.84 | 0.824 | 0.163 |

The first-semester models have different numbers of training steps, so read
this table as cumulative stages rather than isolated module effects.

## Code map

| Path | Content |
| --- | --- |
| `src/model/encoder/heads/olp.py` | OLP-H config, Hadamard transform, preconditioner module |
| `src/model/encoder/heads/reliability.py` | Reliability head config and opacity gate |
| `src/loss/loss_reliability.py` | Context-residual reliability loss |
| `src/model/retention/` | Retention: `selector.py` (config, module), `cells.py`, `scores.py`, `budget.py` |
| `src/model/model_wrapper.py` | Calls retention in train, validation and test steps |
| `src/experiment_runner/retention.py` | Launcher behind `scripts/run_retention.py` |
| `src/experiment_runner/runner.py` | Launcher behind `scripts/run_re10k.py` (OLP-H presets) |
| `config/experiment/retention.yaml` | Hydra preset for every retention run |
| `config/experiment_runs/retention_suite.yaml` | Runner settings for the final method |
