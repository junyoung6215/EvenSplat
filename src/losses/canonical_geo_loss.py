from __future__ import annotations

import warnings
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor


RobustType = Literal["charbonnier", "huber", "l1"]

_warned_missing_depth = False


def charbonnier(delta: Tensor, eps: float = 1e-3) -> Tensor:
    return torch.sqrt(delta.square().sum(dim=-1) + eps * eps)


def robust_penalty(
    delta: Tensor,
    robust: RobustType = "charbonnier",
    charbonnier_eps: float = 1e-3,
    huber_delta: float = 1.0,
) -> Tensor:
    if robust == "charbonnier":
        return charbonnier(delta, charbonnier_eps)
    if robust == "huber":
        norm = torch.linalg.norm(delta, dim=-1)
        quadratic = torch.minimum(norm, torch.as_tensor(huber_delta, device=norm.device))
        linear = norm - quadratic
        return 0.5 * quadratic.square() + huber_delta * linear
    if robust == "l1":
        return delta.abs().sum(dim=-1)
    raise ValueError(
        f"Unsupported robust penalty {robust!r}. Expected charbonnier, huber, or l1."
    )


def _canonical_points_to_map(
    canonical_points: Tensor,
    image_shape: tuple[int, int] | None,
) -> Tensor:
    if canonical_points.ndim == 5:
        if canonical_points.shape[-1] != 3:
            raise ValueError(
                "canonical_points must end with xyz dimension 3, got "
                f"{canonical_points.shape}."
            )
        return canonical_points

    if canonical_points.ndim == 4:
        if image_shape is None:
            raise RuntimeError(
                "Flattened canonical point maps require image_shape=(H, W) to "
                "guarantee the pixel mapping."
            )
        h, w = image_shape
        b, v, n, xyz = canonical_points.shape
        if xyz != 3 or n != h * w:
            raise RuntimeError(
                "Flattened canonical point maps can only be reshaped when their "
                f"third dimension equals H*W. Got shape {canonical_points.shape} "
                f"and image_shape={image_shape}."
            )
        return canonical_points.view(b, v, h, w, 3)

    raise ValueError(
        "Expected canonical_points with shape [B, V, H, W, 3] or "
        f"[B, V, H*W, 3], got {canonical_points.shape}."
    )


def _gather_anchor_view(value: Tensor, anchor_indices: Tensor) -> Tensor:
    b = value.shape[0]
    index_shape = [b, 1] + [1] * (value.ndim - 2)
    gather_index = anchor_indices.to(value.device).view(*index_shape)
    gather_index = gather_index.expand(b, 1, *value.shape[2:])
    return value.gather(dim=1, index=gather_index).squeeze(1)


def _as_bvhw_map(value: Tensor, name: str) -> Tensor:
    if value.ndim == 5 and value.shape[2] == 1:
        value = value.squeeze(2)
    if value.ndim == 5 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    if value.ndim != 4:
        raise ValueError(
            f"{name} must have shape [B, V, H, W], [B, V, 1, H, W], "
            f"or [B, V, H, W, 1], got {value.shape}."
        )
    return value


def _sample_anchor_map(anchor_map: Tensor, grid: Tensor) -> Tensor:
    """Sample [B, C, H, W] anchor map for each [B, V, H, W, 2] grid."""
    b, v, h, w = grid.shape[:4]
    channels = anchor_map.shape[1]
    anchor_expanded = (
        anchor_map[:, None]
        .expand(b, v, channels, anchor_map.shape[-2], anchor_map.shape[-1])
        .reshape(b * v, channels, anchor_map.shape[-2], anchor_map.shape[-1])
    )
    sampled = F.grid_sample(
        anchor_expanded,
        grid.reshape(b * v, h, w, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.reshape(b, v, channels, h, w)


def _zero_nonfinite(value: Tensor) -> Tensor:
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def project_points_to_anchor(
    canonical_points: Tensor,
    anchor_intrinsics: Tensor,
    image_shape: tuple[int, int],
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Project canonical points into the anchor image.

    Returns pixel coordinates, normalized grid_sample coordinates, z, and in-bounds
    mask. Intrinsics are assumed to be normalized like the dataset intrinsics used
    by NoPoSplat, so fx/cx are scaled by image width and fy/cy by image height.
    """
    if canonical_points.ndim != 5 or canonical_points.shape[-1] != 3:
        raise ValueError(
            "project_points_to_anchor expects points with shape [B, V, H, W, 3], "
            f"got {canonical_points.shape}."
        )
    if anchor_intrinsics.ndim != 3 or anchor_intrinsics.shape[-2:] != (3, 3):
        raise ValueError(
            "anchor_intrinsics must have shape [B, 3, 3], got "
            f"{anchor_intrinsics.shape}."
        )

    h, w = image_shape
    points = canonical_points
    x, y, z = points.unbind(dim=-1)
    z_safe = z.clamp_min(eps)

    fx = anchor_intrinsics[:, 0, 0].view(-1, 1, 1, 1) * w
    fy = anchor_intrinsics[:, 1, 1].view(-1, 1, 1, 1) * h
    cx = anchor_intrinsics[:, 0, 2].view(-1, 1, 1, 1) * w
    cy = anchor_intrinsics[:, 1, 2].view(-1, 1, 1, 1) * h

    u = fx * (x / z_safe) + cx
    v_coord = fy * (y / z_safe) + cy

    in_bounds = (
        torch.isfinite(points).all(dim=-1)
        & (z > eps)
        & (u >= 0)
        & (u <= max(w - 1, 0))
        & (v_coord >= 0)
        & (v_coord <= max(h - 1, 0))
    )

    if w > 1:
        grid_x = 2.0 * u / (w - 1) - 1.0
    else:
        grid_x = torch.zeros_like(u)
    if h > 1:
        grid_y = 2.0 * v_coord / (h - 1) - 1.0
    else:
        grid_y = torch.zeros_like(v_coord)

    pixel_coords = torch.stack([u, v_coord], dim=-1)
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return pixel_coords, grid, z, in_bounds


def compute_reference_centered_geo_loss(
    canonical_points: Tensor,
    intrinsics: Tensor,
    *,
    anchor_indices: Tensor | None = None,
    image_shape: tuple[int, int] | None = None,
    depth: Tensor | None = None,
    confidence: Tensor | None = None,
    robust: RobustType = "charbonnier",
    charbonnier_eps: float = 1e-3,
    visibility_depth_threshold: float = 0.05,
    use_confidence_weight: bool = True,
    stop_gradient_anchor: bool = True,
    min_valid_pixels: int = 128,
    warn_missing_depth: bool = True,
    return_debug_tensors: bool = False,
) -> tuple[Tensor, dict[str, Tensor | float | int]]:
    global _warned_missing_depth

    points = _canonical_points_to_map(canonical_points, image_shape)
    b, v, h, w, _ = points.shape
    if intrinsics.shape[:2] != (b, v) or intrinsics.shape[-2:] != (3, 3):
        raise ValueError(
            "intrinsics must have shape [B, V, 3, 3] aligned with canonical_points, "
            f"got {intrinsics.shape} for points {points.shape}."
        )
    if anchor_indices is None:
        anchor_indices = torch.zeros(b, dtype=torch.long, device=points.device)
    else:
        anchor_indices = anchor_indices.to(device=points.device, dtype=torch.long)
    if anchor_indices.shape != (b,):
        raise ValueError(
            f"anchor_indices must have shape [B], got {anchor_indices.shape}."
        )

    anchor_points = _gather_anchor_view(points, anchor_indices)
    if stop_gradient_anchor:
        anchor_points = anchor_points.detach()
    anchor_intrinsics = _gather_anchor_view(intrinsics, anchor_indices)

    _, grid, projected_z, in_bounds = project_points_to_anchor(
        points,
        anchor_intrinsics,
        (h, w),
    )
    grid_finite = torch.isfinite(grid).all(dim=-1)
    sampleable = in_bounds & grid_finite
    safe_grid = torch.where(
        sampleable[..., None],
        grid.clamp(min=-2.0, max=2.0),
        torch.zeros_like(grid),
    )

    anchor_points_finite = torch.isfinite(anchor_points).all(dim=-1)
    anchor_points_chw = _zero_nonfinite(anchor_points).permute(0, 3, 1, 2)
    sampled_anchor_points = _sample_anchor_map(anchor_points_chw, safe_grid)
    sampled_anchor_points = sampled_anchor_points.permute(0, 1, 3, 4, 2)
    sampled_anchor_valid = _sample_anchor_map(
        anchor_points_finite.float()[:, None],
        safe_grid,
    ).squeeze(2) > 0.999
    sampled_anchor_finite = torch.isfinite(sampled_anchor_points).all(dim=-1)

    valid = sampleable & sampled_anchor_valid & sampled_anchor_finite
    if depth is None:
        if warn_missing_depth and not _warned_missing_depth:
            warnings.warn(
                "loss.geo.enabled=True has no cached depth/visibility map; using "
                "in-image projection mask only for canonical geometry loss.",
                RuntimeWarning,
                stacklevel=2,
            )
            _warned_missing_depth = True
    else:
        depth_map = _as_bvhw_map(depth, "depth")
        if depth_map.shape[:2] != (b, v) or depth_map.shape[-2:] != (h, w):
            raise ValueError(
                f"depth must align with points as [B, V, H, W], got {depth_map.shape}."
            )
        depth_finite = torch.isfinite(depth_map)
        anchor_depth = _gather_anchor_view(_zero_nonfinite(depth_map), anchor_indices)
        sampled_depth = _sample_anchor_map(anchor_depth[:, None], safe_grid).squeeze(2)
        sampled_depth_valid = _sample_anchor_map(
            _gather_anchor_view(depth_finite.float(), anchor_indices)[:, None],
            safe_grid,
        ).squeeze(2) > 0.999
        visible = (
            torch.isfinite(projected_z)
            & torch.isfinite(sampled_depth)
            & sampled_depth_valid
            & ((projected_z - sampled_depth).abs() < visibility_depth_threshold)
        )
        valid = valid & visible

    non_anchor = (
        torch.arange(v, device=points.device).view(1, v, 1, 1)
        != anchor_indices.view(b, 1, 1, 1)
    )
    projected_valid = in_bounds & non_anchor
    valid = valid & non_anchor

    weights = torch.ones_like(projected_z)
    if use_confidence_weight and confidence is not None:
        confidence_map = _as_bvhw_map(confidence, "confidence")
        if confidence_map.shape[:2] != (b, v) or confidence_map.shape[-2:] != (h, w):
            raise ValueError(
                "confidence must align with points as [B, V, H, W], got "
                f"{confidence_map.shape}."
            )
        confidence_map = _zero_nonfinite(confidence_map.float()).clamp_min(0)
        source_conf = confidence_map
        anchor_conf = _gather_anchor_view(confidence_map, anchor_indices)
        sampled_anchor_conf = _sample_anchor_map(
            anchor_conf[:, None],
            safe_grid,
        ).squeeze(2)
        weights = torch.sqrt(source_conf * sampled_anchor_conf.clamp_min(0))
        weights = _zero_nonfinite(weights).clamp(max=1.0e6)

    valid_counts = valid.flatten(2).sum(dim=2)
    pair_keep = valid_counts >= int(min_valid_pixels)
    valid = valid & pair_keep[:, :, None, None]

    delta = points - sampled_anchor_points
    delta_finite = torch.isfinite(delta).all(dim=-1)
    valid = valid & delta_finite
    safe_delta = torch.where(
        valid[..., None],
        delta,
        torch.zeros_like(delta),
    )
    safe_delta = _zero_nonfinite(safe_delta)
    raw_penalty = robust_penalty(safe_delta, robust, charbonnier_eps)
    penalty_finite = torch.isfinite(raw_penalty)
    penalty = _zero_nonfinite(raw_penalty)
    valid = valid & penalty_finite & torch.isfinite(weights)
    weighted_valid = valid.float() * weights
    weighted_valid = _zero_nonfinite(weighted_valid).clamp(max=1.0e6)
    raw_weighted_penalty = penalty * weighted_valid
    weighted_penalty_finite = torch.isfinite(raw_weighted_penalty)
    valid = valid & weighted_penalty_finite
    weighted_valid = torch.where(
        weighted_penalty_finite,
        weighted_valid,
        torch.zeros_like(weighted_valid),
    )
    denom = weighted_valid.sum().clamp_min(1.0)
    weighted_penalty = _zero_nonfinite(raw_weighted_penalty)
    loss = weighted_penalty.sum() / denom
    loss = _zero_nonfinite(loss)
    valid_ratio = valid.float().mean()

    info: dict[str, Tensor | float | int] = {
        "valid_ratio": valid_ratio.detach(),
        "projected_valid_ratio": projected_valid.float().mean().detach(),
        "loss_unweighted": loss.detach(),
        "valid_pixel_count": valid.sum().detach(),
        "denominator": denom.detach(),
        "used_depth_visibility": int(depth is not None),
    }
    if return_debug_tensors:
        info["valid_mask"] = valid.detach()
        info["weights"] = weights.detach()
        info["grid"] = grid.detach()

    return loss, info
