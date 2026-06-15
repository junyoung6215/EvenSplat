from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor


AnchorMethod = Literal["none", "middle", "sharpness", "vggt_cache"]
AnchorApplyTo = Literal["test", "train", "both"]

_warned_vggt_overlap_fallback = False
_warned_unsafe_reorder = False


@dataclass
class AnchorScoreWeightsCfg:
    overlap: float = 1.0
    confidence: float = 0.5
    blur: float = 0.1


@dataclass
class AnchorSelectionCfg:
    enabled: bool = False
    method: AnchorMethod = "none"
    apply_to: AnchorApplyTo = "test"
    renormalize_cameras: bool = True
    allow_unsafe_reorder: bool = False
    debug_save_scores: bool = False
    vggt_cache_root: str | None = None
    score_weights: AnchorScoreWeightsCfg = field(
        default_factory=AnchorScoreWeightsCfg
    )


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _as_device_tensor(value: Any, device: torch.device) -> Tensor | None:
    if value is None:
        return None
    if isinstance(value, Tensor):
        return value.to(device=device)
    return torch.as_tensor(value, device=device)


def get_context_images(batch: dict[str, Any]) -> Tensor:
    try:
        images = batch["context"]["image"]
    except KeyError as exc:
        raise KeyError(
            "Anchor selection requires batch['context']['image'] with shape "
            "[B, V, C, H, W]."
        ) from exc

    if not isinstance(images, Tensor) or images.ndim != 5:
        raise ValueError(
            "Anchor selection requires context images with shape [B, V, C, H, W], "
            f"got {getattr(images, 'shape', None)}."
        )
    return images


def anchor_selection_applies_to_stage(cfg: Any, stage: str) -> bool:
    enabled = bool(_cfg_get(cfg, "enabled", False))
    method = _cfg_get(cfg, "method", "none")
    if not enabled or method == "none":
        return False

    apply_to = _cfg_get(cfg, "apply_to", "test")
    if apply_to == "both":
        return True
    if stage == "train":
        return apply_to == "train"
    return apply_to == "test"


def require_safe_anchor_reorder(cfg: Any) -> bool:
    """Return whether cameras should be renormalized before context reordering.

    NoPoSplat stores extrinsics as camera-to-world matrices. The pose-free model
    and renderer require predicted Gaussian means and target cameras to share the
    same world frame. Reordering context views without changing that frame is only
    a diagnostic/unsafe mode, so it must be explicitly enabled.
    """
    global _warned_unsafe_reorder

    renormalize = bool(_cfg_get(cfg, "renormalize_cameras", True))
    allow_unsafe = bool(_cfg_get(cfg, "allow_unsafe_reorder", False))
    if renormalize:
        return True
    if allow_unsafe:
        if not _warned_unsafe_reorder:
            warnings.warn(
                "Unsafe anchor reordering is enabled. Cameras remain in the "
                "original reference frame; results may be theoretically invalid.",
                RuntimeWarning,
                stacklevel=2,
            )
            _warned_unsafe_reorder = True
        return False
    raise RuntimeError(
        "Anchor reordering without camera renormalization is unsafe for "
        "NoPoSplat's coordinate-frame contract. The dataset/model expect "
        "Gaussian means and context/target camera extrinsics to share one frame. "
        "Set model.anchor_selection.renormalize_cameras=true, or explicitly set "
        "model.anchor_selection.allow_unsafe_reorder=true for diagnostic-only "
        "experiments."
    )


def compute_sharpness_scores(images: Tensor) -> Tensor:
    """Compute variance-of-Laplacian sharpness scores for [B, V, C, H, W]."""
    if images.ndim != 5:
        raise ValueError(
            f"Expected images with shape [B, V, C, H, W], got {images.shape}."
        )

    with torch.no_grad():
        b, v, c, h, w = images.shape
        images_f = images.detach().float()
        if c == 1:
            gray = images_f[:, :, :1]
        elif c >= 3:
            weights = torch.tensor(
                [0.2989, 0.5870, 0.1140],
                dtype=images_f.dtype,
                device=images_f.device,
            )
            gray = (images_f[:, :, :3] * weights.view(1, 1, 3, 1, 1)).sum(
                dim=2, keepdim=True
            )
        else:
            gray = images_f.mean(dim=2, keepdim=True)

        flat = gray.reshape(b * v, 1, h, w)
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=flat.dtype,
            device=flat.device,
        ).view(1, 1, 3, 3)
        padded = F.pad(flat, (1, 1, 1, 1), mode="replicate")
        laplacian = F.conv2d(padded, kernel)
        scores = laplacian.flatten(1).var(dim=1, unbiased=False)
        return scores.view(b, v)


def _normalize_per_sample(values: Tensor, eps: float = 1e-8) -> Tensor:
    min_v = values.amin(dim=1, keepdim=True)
    max_v = values.amax(dim=1, keepdim=True)
    return (values - min_v) / (max_v - min_v + eps)


def _find_context_tensor(context: dict[str, Any], names: tuple[str, ...]) -> Tensor | None:
    for name in names:
        value = context.get(name)
        if isinstance(value, Tensor):
            return value
    return None


def _mean_per_view(value: Tensor, b: int, v: int) -> Tensor:
    if value.ndim < 2 or value.shape[0] != b or value.shape[1] != v:
        raise ValueError(
            "Expected a context-aligned VGGT tensor with leading shape [B, V], "
            f"got {value.shape}."
        )
    if value.ndim == 2:
        return value.float()
    reduce_dims = tuple(range(2, value.ndim))
    return value.float().mean(dim=reduce_dims)


def _candidate_cache_paths(root: Path, key: str) -> list[Path]:
    safe_key = key.replace("\\", "/")
    return [
        root / f"{safe_key}.pt",
        root / f"{safe_key}.pth",
        root / f"{safe_key}.torch",
        root / f"{safe_key}.npz",
        root / safe_key / "vggt_cache.pt",
        root / safe_key / "cache.pt",
    ]


def _load_cache_file(path: Path) -> dict[str, Any]:
    if path.suffix == ".npz":
        import numpy as np

        with np.load(path) as data:
            return {key: data[key] for key in data.files}
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(f"VGGT cache file {path} must contain a dictionary.")
    return data


def _extract_cached_context_tensor(
    cache: dict[str, Any],
    names: tuple[str, ...],
    context_indices: Tensor,
    device: torch.device,
) -> Tensor | None:
    for name in names:
        if name not in cache:
            continue
        tensor = _as_device_tensor(cache[name], device)
        if tensor is None:
            continue
        if tensor.ndim == 0:
            continue

        indices = context_indices.to(device=tensor.device, dtype=torch.long)
        if tensor.shape[0] == indices.numel():
            return tensor
        if tensor.shape[0] > int(indices.max().item()):
            return tensor.index_select(0, indices)
        raise ValueError(
            f"Cached tensor '{name}' has first dimension {tensor.shape[0]}, "
            f"but context indices require max index {int(indices.max().item())}."
        )
    return None


def _load_vggt_cache_for_batch(
    batch: dict[str, Any],
    cache_root: str | None,
    names: tuple[str, ...],
    device: torch.device,
) -> Tensor | None:
    if not cache_root:
        return None

    root = Path(cache_root)
    scenes = batch.get("scene")
    indices = batch.get("context", {}).get("index")
    if scenes is None or indices is None:
        return None
    if isinstance(scenes, str):
        scenes = [scenes]
    if not isinstance(indices, Tensor):
        return None

    per_sample = []
    for scene, sample_indices in zip(scenes, indices):
        cache = None
        for candidate in _candidate_cache_paths(root, str(scene)):
            if candidate.exists():
                cache = _load_cache_file(candidate)
                break
        if cache is None:
            return None
        tensor = _extract_cached_context_tensor(cache, names, sample_indices, device)
        if tensor is None:
            return None
        per_sample.append(tensor)

    return torch.stack(per_sample, dim=0)


def _get_vggt_confidence(batch: dict[str, Any], images: Tensor, cfg: Any) -> Tensor | None:
    context = batch.get("context", {})
    confidence = _find_context_tensor(
        context,
        (
            "vggt_confidence",
            "vggt_conf",
            "confidence",
            "conf",
        ),
    )
    if confidence is None:
        confidence = _load_vggt_cache_for_batch(
            batch,
            _cfg_get(cfg, "vggt_cache_root"),
            ("confidence", "conf", "vggt_confidence", "vggt_conf"),
            images.device,
        )
    return confidence


def _get_overlap_scores(batch: dict[str, Any], b: int, v: int, device: torch.device) -> Tensor | None:
    context = batch.get("context", {})
    overlap = _find_context_tensor(context, ("vggt_overlap", "overlap"))
    if overlap is None:
        return None
    overlap = overlap.to(device=device)
    if overlap.ndim == 2 and overlap.shape[:2] == (b, v):
        return overlap.float()
    if overlap.ndim >= 3 and overlap.shape[0] == b and overlap.shape[1] == v:
        return _mean_per_view(overlap, b, v)
    return None


def _vggt_cache_scores(batch: dict[str, Any], images: Tensor, cfg: Any) -> Tensor:
    global _warned_vggt_overlap_fallback

    b, v = images.shape[:2]
    confidence = _get_vggt_confidence(batch, images, cfg)
    if confidence is None:
        raise RuntimeError(
            "model.anchor_selection.method='vggt_cache' requires cached VGGT "
            "confidence aligned to context views. Attach one of "
            "batch['context']['vggt_confidence'], batch['context']['confidence'], "
            "or provide model.anchor_selection.vggt_cache_root containing per-scene "
            "cache files with a confidence/conf tensor."
        )
    confidence = _mean_per_view(confidence, b, v)

    sharpness = compute_sharpness_scores(images)
    blur_penalty = 1.0 - _normalize_per_sample(sharpness)

    weights = _cfg_get(cfg, "score_weights", AnchorScoreWeightsCfg())
    alpha = float(_cfg_get(weights, "overlap", 1.0))
    beta = float(_cfg_get(weights, "confidence", 0.5))
    gamma = float(_cfg_get(weights, "blur", 0.1))

    scores = beta * confidence - gamma * blur_penalty
    overlap = _get_overlap_scores(batch, b, v, images.device)
    if overlap is None:
        if not _warned_vggt_overlap_fallback:
            warnings.warn(
                "VGGT anchor scoring could not compute robust overlap from cached "
                "depth/camera fields; using confidence minus blur fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            _warned_vggt_overlap_fallback = True
    else:
        scores = scores + alpha * overlap.float()

    return scores


def select_anchor_indices(
    batch: dict[str, Any],
    cfg: Any,
    *,
    return_scores: bool = False,
) -> Tensor | tuple[Tensor, Tensor | None]:
    images = get_context_images(batch)
    b, v = images.shape[:2]
    device = images.device

    enabled = bool(_cfg_get(cfg, "enabled", False))
    method = _cfg_get(cfg, "method", "none")

    scores: Tensor | None = None
    if not enabled or method == "none":
        anchors = torch.zeros(b, dtype=torch.long, device=device)
    elif method == "middle":
        anchors = torch.full((b,), v // 2, dtype=torch.long, device=device)
    elif method == "sharpness":
        scores = compute_sharpness_scores(images)
        anchors = scores.argmax(dim=1).long()
    elif method == "vggt_cache":
        scores = _vggt_cache_scores(batch, images, cfg)
        anchors = scores.argmax(dim=1).long()
    else:
        raise ValueError(
            "Unsupported anchor selection method "
            f"{method!r}. Expected one of none, middle, sharpness, vggt_cache."
        )

    if return_scores:
        return anchors, scores
    return anchors


def compute_reorder_indices(anchor_indices: Tensor, num_views: int) -> Tensor:
    if anchor_indices.ndim != 1:
        raise ValueError(
            f"Expected anchor_indices with shape [B], got {anchor_indices.shape}."
        )

    device = anchor_indices.device
    base = torch.arange(num_views, device=device).view(1, num_views)
    orders = []
    for anchor in anchor_indices.tolist():
        if anchor < 0 or anchor >= num_views:
            raise ValueError(
                f"Anchor index {anchor} is out of range for {num_views} views."
            )
        anchor_tensor = torch.tensor([anchor], device=device, dtype=base.dtype)
        remaining = base[0][base[0] != anchor]
        orders.append(torch.cat([anchor_tensor, remaining], dim=0))
    return torch.stack(orders, dim=0)


def _reorder_tensor(value: Tensor, order: Tensor, b: int, v: int) -> Tensor:
    if value.ndim < 2 or value.shape[0] != b or value.shape[1] != v:
        return value
    gather_index = order.to(value.device)
    view_shape = [b, v] + [1] * (value.ndim - 2)
    gather_index = gather_index.view(*view_shape).expand(b, v, *value.shape[2:])
    return value.gather(dim=1, index=gather_index)


def _reorder_tree(value: Any, order: Tensor, b: int, v: int) -> Any:
    if isinstance(value, Tensor):
        return _reorder_tensor(value, order, b, v)
    if isinstance(value, dict):
        return {key: _reorder_tree(child, order, b, v) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_reorder_tree(child, order, b, v) for child in value)
    if isinstance(value, list):
        return [_reorder_tree(child, order, b, v) for child in value]
    return value


def reorder_context_fields(
    batch: dict[str, Any],
    anchor_indices: Tensor,
) -> dict[str, Any]:
    images = get_context_images(batch)
    b, v = images.shape[:2]
    order = compute_reorder_indices(anchor_indices, v)

    reordered = dict(batch)
    reordered["context"] = _reorder_tree(batch["context"], order, b, v)
    return reordered


def _validate_extrinsics(
    value: Any,
    name: str,
    b: int,
    expected_views: int | None = None,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise RuntimeError(
            f"Anchor camera renormalization requires batch['{name}']['extrinsics'] "
            "as camera-to-world matrices with shape [B, V, 4, 4]."
        )
    if value.ndim != 4 or value.shape[0] != b or value.shape[-2:] != (4, 4):
        raise RuntimeError(
            f"batch['{name}']['extrinsics'] must have shape [B, V, 4, 4], "
            f"got {tuple(value.shape)}."
        )
    if expected_views is not None and value.shape[1] != expected_views:
        raise RuntimeError(
            f"batch['{name}']['extrinsics'] must have {expected_views} views "
            f"aligned with context images, got {value.shape[1]}."
        )
    return value


def _gather_extrinsic_view(extrinsics: Tensor, indices: Tensor) -> Tensor:
    b = extrinsics.shape[0]
    gather_index = indices.to(extrinsics.device).view(b, 1, 1, 1)
    gather_index = gather_index.expand(b, 1, 4, 4)
    return extrinsics.gather(dim=1, index=gather_index).squeeze(1)


def _renormalize_extrinsics_to_anchor(
    extrinsics: Tensor,
    anchor_c2w: Tensor,
) -> Tensor:
    world_to_anchor = torch.linalg.inv(anchor_c2w)
    return world_to_anchor[:, None] @ extrinsics


def renormalize_cameras_to_selected_anchor(
    batch: dict[str, Any],
    anchor_indices: Tensor,
) -> dict[str, Any]:
    """Reorder context and express all cameras in the selected-anchor frame.

    Repository convention: dataset extrinsics are camera-to-world transforms.
    Given selected anchor c2w matrix E_a, every camera E_i is rewritten as
    E'_i = inverse(E_a) @ E_i. After this transform, the selected anchor camera is
    identity. Context-aligned fields are then reordered so the selected anchor is
    context index 0. Target views are never reordered; only their extrinsics are
    transformed into the same selected-anchor world frame.
    """
    images = get_context_images(batch)
    b, v = images.shape[:2]
    if anchor_indices.shape != (b,):
        raise ValueError(
            f"Expected anchor_indices with shape [B], got {anchor_indices.shape}."
        )

    context = batch.get("context", {})
    target = batch.get("target", {})
    context_extrinsics = _validate_extrinsics(
        context.get("extrinsics"),
        "context",
        b,
        expected_views=v,
    )
    anchor_c2w = _gather_extrinsic_view(context_extrinsics, anchor_indices)

    transformed = dict(batch)
    transformed_context = dict(context)
    transformed_context["extrinsics"] = _renormalize_extrinsics_to_anchor(
        context_extrinsics,
        anchor_c2w,
    )
    transformed["context"] = transformed_context

    if target:
        transformed_target = dict(target)
        if "extrinsics" not in target:
            raise RuntimeError(
                "Anchor camera renormalization requires target extrinsics so "
                "renderer cameras stay in the selected-anchor frame."
            )
        target_extrinsics = _validate_extrinsics(target["extrinsics"], "target", b)
        transformed_target["extrinsics"] = _renormalize_extrinsics_to_anchor(
            target_extrinsics,
            anchor_c2w,
        )
        transformed["target"] = transformed_target

    return reorder_context_fields(transformed, anchor_indices)


def save_anchor_debug(
    output_dir: str | Path,
    *,
    stage: str,
    batch_idx: int,
    global_step: int,
    selected_anchor_indices: Tensor,
    original_context_indices: Tensor | None,
    reordered_context_indices: Tensor | None,
    scores: Tensor | None,
    scenes: Any = None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / f"{stage}_batch_{batch_idx:06d}_step_{global_step:06d}.json"

    if isinstance(scenes, str):
        scenes = [scenes]
    payload: dict[str, Any] = {
        "stage": stage,
        "batch_idx": batch_idx,
        "global_step": global_step,
        "scene": scenes,
        "selected_anchor_indices": selected_anchor_indices.detach().cpu().tolist(),
    }
    if original_context_indices is not None:
        payload["original_context_indices"] = (
            original_context_indices.detach().cpu().tolist()
        )
    if reordered_context_indices is not None:
        payload["reordered_context_indices"] = (
            reordered_context_indices.detach().cpu().tolist()
        )
    if scores is not None:
        payload["scores"] = scores.detach().cpu().tolist()

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path
