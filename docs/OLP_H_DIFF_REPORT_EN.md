# OLP-H Difference Report

This repository is based on upstream `cvg/NoPoSplat` and adds a modular
experiment for Orthogonal Latent Preconditioning with a randomized Hadamard
transform.

## Base And Experiment Version

- Base repository: `cvg/NoPoSplat`
- Local base commit: `a097a78c5bdd0486493f74abb1165614d86ae952`
- Experiment version: `olp-h-mvp-cuda118-v0.2`
- Target environment: CUDA 11.8 + PyTorch 2.1.2 cu118
- Target CUDA/PyTorch stack: CUDA 11.8, `torch==2.1.2+cu118`

## What Is Different From NoPoSplat

### New OLP-H Modules

- `src/model/encoder/heads/latent_preconditioner.py`
  - Implements `RandomizedHadamardPreconditioner`.
  - Input shape is `[B, C, H, W]`.
  - Applies `Q = H D P / sqrt(C)` along the channel dimension.
  - Supports warm-up mixing and optional latent tail statistics.

- `src/model/encoder/heads/preconditioner_cfg.py`
  - Holds the OLP-H config dataclass and config field reader.

- `src/model/encoder/heads/preconditioner_math.py`
  - Holds power-of-two validation and the FWHT implementation.

- `src/model/encoder/heads/preconditioner_stats.py`
  - Holds latent max/RMS and kurtosis summaries.

- `src/model/encoder/heads/preconditioner_utils.py`
  - Propagates `global_step` to modules that expose `set_step()`.

### Thin NoPoSplat Hooks

Only small connector changes are made to upstream model files:

- `src/model/encoder/heads/dpt_gs_head.py`
  - Adds `self.preconditioner = nn.Identity()` by default.
  - Inserts `path_1 = self.preconditioner(path_1)` immediately before the
    Gaussian parameter output head.
  - Builds the OLP-H module only when config enables it.

- `src/model/encoder/heads/__init__.py`
  - Passes optional `preconditioner_cfg` only to the `dpt_gs` Gaussian parameter
    head path.

- `src/model/encoder/encoder_noposplat.py`
  - Adds an optional `orthogonal_preconditioner` config field.
  - Passes that config to both Gaussian parameter heads.
  - Calls step propagation at the beginning of forward.

- `src/model/encoder/encoder_noposplat_multi.py`
  - Applies the same minimal wiring for the multi-view encoder.

The center/pts3d branch is unchanged.

### Experiment Stats And Audit Modules

- `src/analysis/olp_stats.py`
  - Computes latent tail stats and Gaussian opacity/scale/floater proxy stats.

- `src/analysis/olp_eval_recorder.py`
  - Collects preconditioner stats and Gaussian tail stats during evaluation.
  - Saves `olp_stats_summary.json`, `olp_stats_per_scene.json`, and CSV.

- `src/misc/checkpoint_audit.py`
  - Loads pretrained encoder weights and writes a load audit.
  - Prints matched, missing, and unexpected key counts before long runs.

### Config Additions

- `config/model/encoder/noposplat.yaml`
  - Adds `orthogonal_preconditioner`.
  - Default is `enabled: false`, so original NoPoSplat behavior is preserved.

- `config/main.yaml`
  - Adds optional test fields for OLP statistics.
  - Default is `save_olp_stats: false`.

### Reproducibility Additions

- `env/environment-cuda118.yml`
- `env/requirements-cuda118.lock`
- `scripts/create_cuda118_env.sh`
- `scripts/check_cuda118_env.py`
- RE10K launch scripts under `scripts/run_re10k_*.sh`
- `tests/test_olp_preconditioner.py`

## Important Reproducibility Notes

- `RandomizedHadamardPreconditioner` uses `Q = H D P / sqrt(C)`. The channel
  dimension must be a power of two; NoPoSplat `dpt_gs` uses `C=256`, which
  satisfies this assumption. Background:
  https://en.wikipedia.org/wiki/Hadamard_transform
- For baseline latent stats, keep the module enabled but set `mix=0.0`,
  `target_mix=0.0`, and `collect_stats=true`.
- `scripts/run_re10k_original_eval.sh` keeps OLP-H off for official
  reproduction. Original reproduction and latent-stat collection are separated.
- OLP-H buffers are `persistent=False`; use the same `OLP_SEED` for train and
  eval.
- `floater_proxy` is emitted when a fixed `tau_scale` is provided or when the
  adaptive scale quantile is enabled.
- Start GPU validation with `BATCH_SIZE=1` and `MAX_STEPS=2`.
- Unit tests live in `tests/test_olp_preconditioner.py` and cover disabled
  mode, norm preservation, warm-up scheduling, stats collection, seed
  reproducibility, and checkpoint buffer exclusion.
