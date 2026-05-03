# OLP-H MVP Change Map

This branch keeps OLP-H modular. The existing NoPoSplat path is changed only by
thin wiring points; the experiment logic lives in added modules.

## Added Modules

- `src/model/encoder/heads/latent_preconditioner.py`
  - Defines `RandomizedHadamardPreconditioner`.
  - Applies fixed `Q = H D P / sqrt(C)` on `[B, C, H, W]`.
  - Supports `enabled`, warm-up `mix`, `set_step()`, and optional `latest_stats`.

- `src/analysis/olp_stats.py`
  - Standalone latent MaxRMS/kurtosis and Gaussian opacity/scale tail metrics.

- `src/analysis/olp_eval_recorder.py`
  - Small recorder used by `ModelWrapper` when `test.save_olp_stats=true`.
  - Writes `olp_stats_summary.json`, `olp_stats_per_scene.json`, and
    `olp_stats_per_scene.csv`.

- `env/environment-a100.yml` and `env/requirements-a100.lock`
  - A100 Miniconda environment lock.
  - The non-portable `packaging @ file:///...` entry is replaced by
    `packaging==25.0`.

- `scripts/check_a100_env.py`
  - Verifies A100, `torch==2.1.2+cu118`, `torch.version.cuda == 11.8`, and
    pinned package versions.

- `scripts/run_re10k_*.sh`
  - Reproducible entrypoints for original eval, OLP zero-shot, baseline FT,
    OLP FT, and checkpoint stats collection.

## Thin Wiring Changes

- `src/model/encoder/heads/dpt_gs_head.py`
  - Adds `self.preconditioner = nn.Identity()` by default.
  - Calls `path_1 = self.preconditioner(path_1)` immediately before the
    Gaussian output head.
  - Accepts `preconditioner_cfg` from the factory.

- `src/model/encoder/heads/__init__.py`
  - Passes `preconditioner_cfg` only for `head_type == "dpt_gs"` and
    `output_mode == "gs_params"`.

- `src/model/encoder/encoder_noposplat.py`
  - Adds `OrthogonalPreconditionerCfg`.
  - Passes the config only to `gaussian_param_head` and `gaussian_param_head2`.
  - Calls `set_step(global_step)` on modules that support it.

- `src/model/encoder/encoder_noposplat_multi.py`
  - Applies the same wiring for the multi-view encoder.

- `src/model/model_wrapper.py`
  - Adds only a thin hook to log OLP latent stats during training and write
    OLP/Gaussian tail stats during test when `test.save_olp_stats=true`.

- `src/main.py`
  - Enables `save_last=True` for Lightning checkpoints.

## Default Behavior

The default config is:

```yaml
orthogonal_preconditioner:
  enabled: false
```

So the inserted DPT-GS module resolves to `nn.Identity()` and pretrained
checkpoint loading remains compatible with original NoPoSplat.

## Main Commands

```bash
scripts/create_a100_env.sh
scripts/run_re10k_original_eval.sh
scripts/run_re10k_olp_zeroshot.sh
scripts/run_re10k_ft_control.sh
scripts/run_re10k_olp_ft.sh
BASELINE_CKPT=/path/to/control.ckpt OLP_CKPT=/path/to/olp.ckpt \
  scripts/run_re10k_collect_stats.sh
```
