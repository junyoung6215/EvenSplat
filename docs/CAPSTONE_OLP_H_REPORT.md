# Capstone OLP-H Final Change Report

## 1. Purpose

This repository adds an OLP-H experiment to NoPoSplat. The goal is not to apply
PolarQuant directly to 3D Gaussian Splatting. Instead, this experiment borrows
only the random orthogonal preconditioning idea and applies it before the
NoPoSplat `dpt_gs` Gaussian parameter head.

Main comparison:

```text
NoPoSplat + fine-tuning
vs.
NoPoSplat + OLP-H + fine-tuning
```

The expected signal is not necessarily higher average PSNR. The target signal is
reduced latent outliers, opacity/scale tails, and floater-like failure cases.

## 2. Version And Runtime

- Experiment version: `olp-h-mvp-a100-v0.1`
- Date prepared: `2026-05-03 KST`
- Branch: `codex/olp-h-a100-repro`
- Base NoPoSplat commit: `a097a78c5bdd0486493f74abb1165614d86ae952`
- Main implementation commit: `1ec448e906ef43c8a08138ebf2fac0784719d721`
- Target GPU: NVIDIA A100
- Environment manager: Miniconda
- Python: `3.10`
- CUDA: `11.8`
- PyTorch: `torch==2.1.2+cu118`
- TorchVision: `torchvision==0.16.2+cu118`
- TorchAudio: `torchaudio==2.1.2+cu118`
- Lightning: `lightning==2.6.1`, `pytorch-lightning==2.6.1`
- Gaussian rasterizer:
  `diff_gaussian_rasterization @ git+https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose.git@43e21bff91cd24986ee3dd52fe0bb06952e50ec7`

The full package lock is stored in `env/requirements-a100.lock`.

## 3. Implementation Summary

### Added Modules

- `src/model/encoder/heads/latent_preconditioner.py`
  - Adds `RandomizedHadamardPreconditioner`.
  - Input shape is `[B, C, H, W]`.
  - Default channel dimension is `C=256`.
  - Applies fixed `Q = H D P / sqrt(C)`.
  - Uses fixed random sign and permutation buffers with `persistent=False`.
  - Supports warm-up mixing through `set_step(global_step)`.
  - Stores optional latent stats in `latest_stats`.

- `src/analysis/olp_stats.py`
  - Computes latent MaxRMS and kurtosis.
  - Computes Gaussian opacity, scale, volume, and floater proxy statistics.

- `src/analysis/olp_eval_recorder.py`
  - Collects preconditioner stats and Gaussian tail stats during evaluation.
  - Writes:
    - `olp_stats_summary.json`
    - `olp_stats_per_scene.json`
    - `olp_stats_per_scene.csv`
  - Computes `floater_proxy` with either a fixed `tau_scale` or an adaptive
    per-batch scale p95 threshold.

- `src/misc/checkpoint_audit.py`
  - Loads fine-tuning pretrained encoder weights through one helper.
  - Prints and writes `pretrained_load_audit.json` with matched, missing, and
    unexpected key counts.

### Thin Wiring Changes

- `src/model/encoder/heads/dpt_gs_head.py`
  - Adds a preconditioner module in `DPTOutputAdapter_fix`.
  - Applies it only after `path_1 + direct_img_feat` and before `self.head`.

- `src/model/encoder/heads/__init__.py`
  - Passes `preconditioner_cfg` only to the `dpt_gs` Gaussian parameter head.

- `src/model/encoder/encoder_noposplat.py`
  - Adds `OrthogonalPreconditionerCfg`.
  - Passes the OLP config to `gaussian_param_head` and
    `gaussian_param_head2`.
  - Updates preconditioner step from `global_step`.

- `src/model/encoder/encoder_noposplat_multi.py`
  - Applies the same OLP config wiring for the multi-view encoder path.

- `src/model/model_wrapper.py`
  - Logs OLP latent stats during training when available.
  - Writes evaluation stats only when `test.save_olp_stats=true`.
  - Delegates threshold and serialization logic to `OlpEvalRecorder`.

- `src/main.py`
  - Enables `save_last=True` for Lightning checkpoints.

## 4. Config

Default behavior is disabled, so original NoPoSplat behavior is preserved unless
the experiment is explicitly enabled.

```yaml
orthogonal_preconditioner:
  enabled: false
  mode: hadamard
  dim: 256
  mix: 0.0
  target_mix: 1.0
  warmup_steps: 5000
  seed: 0
  collect_stats: false
```

OLP-H fine-tuning override:

```yaml
model.encoder.orthogonal_preconditioner.enabled=true
model.encoder.orthogonal_preconditioner.mode=hadamard
model.encoder.orthogonal_preconditioner.dim=256
model.encoder.orthogonal_preconditioner.mix=0.0
model.encoder.orthogonal_preconditioner.target_mix=1.0
model.encoder.orthogonal_preconditioner.warmup_steps=5000
model.encoder.orthogonal_preconditioner.seed=0
model.encoder.orthogonal_preconditioner.collect_stats=true
```

OLP stats thresholds:

```yaml
test.olp_tau_alpha: 0.95
test.olp_tau_scale: null
test.olp_tau_volume: null
test.olp_adaptive_tau_scale_quantile: 0.95
```

When `test.olp_tau_scale` is `null`, `floater_proxy` uses the per-batch scale
p95 threshold and records that threshold in the stats output.

## 5. Reproducible A100 Workflow

Create and verify the environment:

```bash
scripts/create_a100_env.sh
conda activate noposplat-olp-a100
```

Run a small smoke test before the full experiment:

```bash
scripts/run_re10k_smoke.sh
```

This defaults to `BATCH_SIZE=1`, `MAX_STEPS=2`, and
`CKPT_EVERY_N_STEPS=1`. It is intended to catch checkpoint loading, CUDA
extension, OLP-H memory, and checkpoint writing problems before a long A100 run.

Run the full experiment sequence:

```bash
scripts/run_re10k_original_eval.sh
scripts/run_re10k_olp_zeroshot.sh
scripts/run_re10k_ft_control.sh
scripts/run_re10k_olp_ft.sh
```

After fine-tuning, compare checkpoints:

```bash
BASELINE_CKPT=/path/to/re10k_ft_control.ckpt \
OLP_CKPT=/path/to/re10k_olp_ft.ckpt \
scripts/run_re10k_collect_stats.sh
```

All outputs are under `outputs/olp_re10k/`.

Baseline latent stats are collected by enabling the OLP-H module with
`mix=0.0` and `target_mix=0.0`. This is identity-equivalent for the output, but
it populates `latest_stats` so baseline and OLP latent outliers can be compared.

## 6. Saved Artifacts

Each run is designed to save:

- command line used for the run
- git commit hash
- `pip freeze`
- `conda list --explicit`
- `nvidia-smi`
- PyTorch and CUDA version fingerprint
- model checkpoints
- rendering metrics when available
- OLP latent stats when enabled
- Gaussian opacity/scale tail stats when enabled
- `olp_seed.txt`, because OLP-H random buffers are `persistent=False`
- `pretrained_load_audit.json` during fine-tuning runs

## 7. Validation Performed Locally

The following checks passed locally:

```bash
bash -n scripts/*.sh
PYTHONPYCACHEPREFIX=/private/tmp/noposplat_pycache python3 -m compileall src scripts tests
PYTHONPYCACHEPREFIX=/private/tmp/noposplat_pycache python3 tests/test_olp_preconditioner.py
git diff --check
```

The unit test covers:

- disabled preconditioner returns input exactly
- Hadamard preconditioner preserves L2 norm at `mix=1`
- identity-stats mode returns input and populates `latest_stats`
- warm-up schedule moves from `mix` to `target_mix`
- output shape equals input shape
- stats helper functions return scalar metrics
- Gaussian stats include `floater_proxy`

## 8. Not Yet Run

These require the A100 server, prepared RE10K dataset, and pretrained weights:

- A100 environment creation
- CUDA extension compile
- pretrained `re10k.ckpt` loading
- original RE10K reproduction
- baseline fine-tuning control
- OLP-H fine-tuning
- final checkpoint metric comparison

## 9. Maintenance Notes

The implementation keeps OLP-H modular:

- To disable the experiment, keep `orthogonal_preconditioner.enabled=false`.
- For baseline latent diagnostics, use `enabled=true`, `mix=0.0`, and
  `target_mix=0.0`.
- To replace Hadamard with another preconditioner, add a new module or mode in
  `build_latent_preconditioner`.
- To add more metrics, extend `src/analysis/olp_stats.py` and keep collection in
  `OlpEvalRecorder`.
- Use the same `OLP_SEED` for OLP fine-tuning and evaluation. The random
  permutation/sign buffers intentionally stay out of checkpoints.
- Set `OLP_TAU_SCALE` for fixed-threshold floater comparisons. If it is unset,
  `floater_proxy` uses adaptive scale p95.
- The original NoPoSplat README, dataset instructions, and citation are kept in
  the root `README.md`.
