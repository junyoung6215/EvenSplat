# Reproducing the results

All runs use one GPU. Paths below are examples; change them to your machine.

## 1. Environment and data

```bash
bash scripts/create_cuda118_env.sh          # PyTorch 2.1.2 + CUDA 11.8, builds the rasterizer
conda activate noposplat-olp-cuda118
python scripts/check_cuda118_env.py         # optional sanity check
bash scripts/download_re10k_checkpoint.sh   # NoPoSplat RE10K checkpoint -> pretrained_weights/re10k.ckpt
RE10K_SOURCE_DIR=/path/to/re10k bash scripts/setup_re10k_dataset.sh
```

`env/environment-cuda118.yml` and `env/requirements-cuda118.lock` pin the exact
package versions. Dataset formats are described in [DATASETS.md](../DATASETS.md).

Evaluation uses three RE10K index files:

| Index | Scenes | Use |
| --- | --- | --- |
| `assets/evaluation_index_re10k.json` | 5,601 | Official test index, all reported numbers (`--split full`) |
| `assets/evaluation_index_re10k_val64.json` | 64 | Validation during fine-tuning (`--split validation`) |
| `assets/evaluation_index_re10k_test500.json` | 500 | Quick held-out check (`--split test`) |

The subsets are the first 64 valid scenes of the official index and the next
500 (`scripts/make_eval_subset.py --count 64`, then `--start 64 --count 500`),
so `--split full` overlaps them by design.

## 2. Base model (OLP-H + reliability head)

The released base checkpoint can be used directly; these two stages rebuild it.

**Stage 1, OLP-H, 300K steps** from the NoPoSplat checkpoint. Set
`storage_root` and `data_root` in the preset first.

```bash
python scripts/run_re10k.py --preset re10k_olp_ft --run-id olp_h_300k
```

**Stage 2, reliability head, 75K steps** with the context-residual loss.

```bash
BASE_CKPT=/path/to/olp_h_300k/train/checkpoints/epoch_0-step_300000.ckpt \
DATA_ROOT=/path/to/re10k \
  bash scripts/train_olp_reliability.sh
```

## 3. Final method: stratified retention + r filter + pruning-aware FT

Copy `config/experiment_runs/retention_suite.yaml`, set `data_root`,
`initial_checkpoint` (the stage 2 checkpoint) and `output_root`. Its defaults are the final method: 2 × 2 cells, r threshold
0.35, 10K steps, batch 1 with 4-step gradient accumulation, lr 2e-5, 4 data workers.

Every command prints the planned jobs and stops; add `--execute` to run them.

```bash
C=/path/to/my_settings.yaml
for r in 0.25 0.5 0.75; do
  python scripts/run_retention.py --config $C train --prune --ratio $r --run-id ours_r$r --execute
done
```

Each run writes `<output_root>/<run-id>/pruned/checkpoints/last.ckpt`.

Evaluate on the full RE10K test set:

```bash
O=/path/to/output_root
python scripts/run_retention.py --config $C eval-grid --split full \
  --checkpoint $O/ours_r0.5/pruned/checkpoints/last.ckpt \
  --baselines none --scores r_filter --ratios 0.5 --run-id full_ours_r0.5 --execute
```

Metrics are written to `<output_root>/<run-id>/<job>/test_metrics.json`
(`psnr_ours`, `ssim_ours`, `lpips_ours`). Collect many runs into one CSV with

```bash
python scripts/run_retention.py summarize --root $O --output results.csv
```

Expected full-test PSNR / SSIM / LPIPS: 25% 24.62 / 0.817 / 0.184,
50% 25.01 / 0.829 / 0.169, 75% 25.26 / 0.837 / 0.161. A full-test run took
about 17 hours on our GPU with pose alignment on. `--split test` (500 scenes)
is a faster check: 25.24 PSNR at 50%.

Related evaluations:

- Base model with all Gaussians: `eval-grid --split full --baselines off --scores`
  (no `--checkpoint`, so the stage 2 checkpoint is evaluated).
- Original NoPoSplat against OLP-H: `python scripts/run_re10k.py --preset re10k_eval_compare`.
- Per-scene metrics, for example to find failure cases: set
  `evaluation.per_scene_stats: true`, which writes `<job>/renders/<job>/olp_stats_per_scene.csv`.
- Qualitative renders: set `evaluation.save_image: true`.

## 4. Ablations (50%)

| Row | Training | Evaluation |
| --- | --- | --- |
| r, global top-k, no FT | none (use the base checkpoint) | `--scores reliability --cell-size 0` |
| + 2 × 2 stratified | none | `--scores reliability` |
| + pruning-aware FT 2K | `train --prune --score reliability --steps 2000` | `--scores reliability` |
| + r filter | `train --prune --steps 2000` | `--scores r_filter` |
| + FT 10K (final) | `train --prune` | `--scores r_filter` |
| Full-count FT, then select | `train` (no `--prune`) | `--scores reliability` |

Further options of `train`:

- `--score grid`: fixed Bayer-ordered pixel positions in every cell.
- `--ratio 1 --ratio-min 0.25`: a new random budget in [0.25, 1] every step.
- Learned r: `--score reliability --cell-size 4 --steps 3000 --budget-weight 0.01
  --soft-steps 1000 --budget-warmup --budget-hard-scale 0`.

`train` without `--execute` prints the exact Hydra command, which can also be
run by hand with `python -m src.main +experiment=retention ...`.

## Tests

```bash
PYTHONPATH=tests python -m unittest discover -s tests
```

The tests need no GPU, dataset or checkpoint. Tests that import the CUDA
rasterizer skip themselves when `diff_gaussian_rasterization` is missing.
