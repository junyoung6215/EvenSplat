# VGGT-AnchorGeo Usage

VGGT-AnchorGeo is a compact, optional research extension merged from NoPoSplat
into this GCSplat/GCSpalt codebase. It keeps the baseline architecture intact
and adds two switchable pieces:

1. Adaptive anchor selection reorders context views so a chosen reference view is
   placed at context index 0 before the existing model runs. When reordering is
   enabled, all context and target camera-to-world extrinsics are also
   renormalized into the selected-anchor frame.
2. A single reference-centered canonical geometry loss can be enabled for
   fine-tuning when real per-view canonical 3D point maps are exposed by the
   encoder.

All new behavior is disabled by default. With the default config, context order,
checkpoint loading, dataset loading, training, and testing follow the original
NoPoSplat path.

## Config Keys

Anchor selection lives under `model.anchor_selection`:

```yaml
model:
  anchor_selection:
    enabled: false
    method: none        # none | middle | sharpness | vggt_cache
    apply_to: test      # test | train | both
    renormalize_cameras: true
    allow_unsafe_reorder: false
    debug_save_scores: false
    vggt_cache_root: null
    score_weights:
      overlap: 1.0
      confidence: 0.5
      blur: 0.1
```

The geometry loss lives under `loss.geo`:

```yaml
loss:
  geo:
    enabled: false
    weight: 0.0
    robust: charbonnier
    charbonnier_eps: 1.0e-3
    visibility_depth_threshold: 0.05
    use_confidence_weight: true
    stop_gradient_anchor: true
    min_valid_pixels: 128
    debug: false
    coordinate_frame: context0
    allow_ambiguous_frame: false
```

Debug output is off by default:

```yaml
debug:
  anchor:
    enabled: false
    output_dir: debug_anchor
    save_first_n_batches: 4
  geo:
    enabled: false
    output_dir: debug_geo
    save_first_n_batches: 4
```

## Methods

`none` preserves the original context order.

`middle` selects context index `V // 2` as the anchor, renormalizes all cameras
to that selected-anchor frame, and moves it to context index 0.

`sharpness` computes a variance-of-Laplacian score for each context image and
selects the sharpest view. This is CPU-safe and does not require gradients.

`vggt_cache` expects cached confidence aligned to the context views, either
attached to `batch["context"]` as `vggt_confidence`/`confidence` or loaded from
`model.anchor_selection.vggt_cache_root`. If robust overlap is not available, it
falls back to `confidence - blur` scoring and emits a warning once. It does not
run VGGT inference.

## Camera Frame Contract

NoPoSplat uses camera-to-world extrinsics. RE10k-style datasets normally apply
`relative_pose: true`, making original `context[0]` the world reference. Anchor
selection changes which view becomes `context[0]`, so plain context reordering is
not enough.

By default, VGGT-AnchorGeo uses:

```text
E'_i = inverse(E_anchor) @ E_i
```

for every context and target camera-to-world matrix. The selected anchor then has
identity extrinsics after reordering, context-aligned fields are reordered, and
target fields keep their original order while their extrinsics move into the same
selected-anchor frame.

Setting `model.anchor_selection.renormalize_cameras=false` is blocked unless
`model.anchor_selection.allow_unsafe_reorder=true` is also set. That unsafe mode
keeps cameras in the old reference frame and is diagnostic only.

## Commands

Baseline unchanged from the project root:

```bash
python -m src.main +experiment=re10k mode=test wandb.name=baseline_test dataset/view_sampler@dataset.re10k.view_sampler=evaluation checkpointing.load=./pretrained_weights/re10k.ckpt test.save_image=true test.save_compare=true dataset.re10k.roots=[datasets/re10k]
```

Anchor-only inference with the middle view:

```bash
python -m src.main +experiment=re10k mode=test wandb.name=anchor_middle dataset/view_sampler@dataset.re10k.view_sampler=evaluation checkpointing.load=./pretrained_weights/re10k.ckpt test.save_image=true test.save_compare=true dataset.re10k.roots=[datasets/re10k] model.anchor_selection.enabled=true model.anchor_selection.method=middle
```

Anchor-only inference with sharpness:

```bash
python -m src.main +experiment=re10k mode=test wandb.name=anchor_sharpness dataset/view_sampler@dataset.re10k.view_sampler=evaluation checkpointing.load=./pretrained_weights/re10k.ckpt test.save_image=true test.save_compare=true dataset.re10k.roots=[datasets/re10k] model.anchor_selection.enabled=true model.anchor_selection.method=sharpness
```

Fine-tuning with the geometry loss:

```bash
python -m src.main +experiment=re10k mode=train wandb.name=anchorgeo_ft checkpointing.load=./pretrained_weights/re10k.ckpt dataset.re10k.roots=[datasets/re10k] model.anchor_selection.enabled=true model.anchor_selection.method=sharpness model.anchor_selection.apply_to=train loss.geo.enabled=true loss.geo.weight=0.01
```

Research-only unsafe diagnostic reordering:

```bash
python -m src.main +experiment=re10k mode=test wandb.name=anchor_middle_unsafe dataset/view_sampler@dataset.re10k.view_sampler=evaluation checkpointing.load=./pretrained_weights/re10k.ckpt test.save_image=false test.save_compare=false dataset.re10k.roots=[datasets/re10k] model.anchor_selection.enabled=true model.anchor_selection.method=middle model.anchor_selection.renormalize_cameras=false model.anchor_selection.allow_unsafe_reorder=true trainer.limit_test_batches=1
```

CPU-only validation:

```bash
python -c "from src.utils.anchor_selection import compute_sharpness_scores, select_anchor_indices, reorder_context_fields; print('anchor_selection import OK')"
python -c "from src.losses.canonical_geo_loss import compute_reference_centered_geo_loss; print('geo_loss import OK')"
python scripts/check_anchorgeo_config.py
python scripts/smoke_test_anchorgeo_units.py
```

Optional lightweight real-data sanity run:

```bash
python -m src.main +experiment=re10k mode=test wandb.name=anchor_middle_sanity dataset/view_sampler@dataset.re10k.view_sampler=evaluation checkpointing.load=./pretrained_weights/re10k.ckpt test.save_image=false test.save_compare=false dataset.re10k.roots=[datasets/re10k] model.anchor_selection.enabled=true model.anchor_selection.method=middle trainer.limit_test_batches=1
```

This optional command may fail on machines without enough GPU memory, without the
checkpoint or dataset, or without CUDA extension support. That is not required for
the limited-GPU validation target.

## Limitations

The geometry loss is a fine-tuning objective. It does not improve pure inference
unless the model has been trained or fine-tuned with it.

The loss uses real per-view 3D point maps from `visualization_dump["means"]` in
the NoPoSplat encoders only when the wrapper has marked them as view-aligned and
expressed in the current `context0` frame. With anchor selection and camera
renormalization enabled, `context0` is the selected-anchor frame. If that tensor
is missing, has more than one Gaussian center per pixel, does not map one point
to one context pixel, or has ambiguous frame metadata, enabling `loss.geo` raises
a clear `RuntimeError`.

`loss.geo.allow_ambiguous_frame=true` bypasses the metadata guard for diagnostic
experiments only. It should not be used for reported experiments unless the point
map coordinate frame has been independently verified.

The implementation does not substitute RGB images, rendered colors, or 2D depth
maps as fake canonical 3D points.

`model.anchor_selection.vggt_cache_root` currently auto-loads confidence for
anchor scoring only. VGGT depth is used by the geometry loss only if a
context-aligned `vggt_depth`, `visibility_depth`, or `anchor_visibility_depth`
tensor is already attached to the batch. It is used only as a visibility mask,
not as direct target supervision. If depth is unavailable, the loss falls back to
the in-image projection mask and emits a warning.
