from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor


OPACITY_PRUNING_THRESHOLDS = (0.001, 0.01, 0.03, 0.05, 0.1, 0.2)


def _to_float_tensor(x: Tensor) -> Tensor:
    return x.detach().float()


def _quantile(x: Tensor, q: float) -> float:
    return float(torch.quantile(x.flatten(), q).cpu())


def _threshold_key(threshold: float) -> str:
    return f"{threshold:g}".replace(".", "p")


def _safe_compression_factor(total_count: int, active_count: Tensor) -> float:
    active = int(active_count.detach().cpu())
    if active <= 0:
        return float(total_count)
    return float(total_count / active)


def opacity_pruning_stats(
    opacities: Tensor,
    thresholds: Iterable[float] = OPACITY_PRUNING_THRESHOLDS,
) -> dict[str, float]:
    """Estimate active Gaussian counts and compression under opacity pruning."""
    opacity = _to_float_tensor(opacities).flatten()
    count = opacity.numel()
    stats: dict[str, float] = {
        "gaussian_count": float(count),
        "opacity_nonzero_frac": float((opacity > 0).float().mean().cpu()),
    }
    for threshold in thresholds:
        key = _threshold_key(float(threshold))
        active = opacity > float(threshold)
        active_count = active.sum()
        active_frac = active.float().mean()
        stats[f"opacity_active_frac_gt_{key}"] = float(active_frac.cpu())
        stats[f"opacity_active_count_gt_{key}"] = float(active_count.cpu())
        stats[f"opacity_prunable_frac_le_{key}"] = float((1.0 - active_frac).cpu())
        stats[f"opacity_compression_factor_gt_{key}"] = _safe_compression_factor(
            count,
            active_count,
        )
    return stats


def opacity_reliability_effect_stats(
    opacities_before: Tensor,
    opacities_after: Tensor,
    thresholds: Iterable[float] = OPACITY_PRUNING_THRESHOLDS,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Compare opacity distributions before and after reliability modulation."""
    before = _to_float_tensor(opacities_before).flatten()
    after = _to_float_tensor(opacities_after).flatten()
    if before.numel() != after.numel():
        raise ValueError(
            "opacities_before_reliability and opacities must contain the same "
            f"number of values, got {before.numel()} and {after.numel()}."
        )

    ratio = after / before.clamp_min(eps)
    reduction = (before - after).clamp_min(0)
    relative_reduction = reduction / before.clamp_min(eps)
    count = before.numel()
    stats: dict[str, float] = {
        "opacity_before_mean": float(before.mean().cpu()),
        "opacity_before_p95": _quantile(before, 0.95),
        "opacity_before_p99": _quantile(before, 0.99),
        "opacity_after_over_before_mean": float(ratio.mean().cpu()),
        "opacity_after_over_before_p05": _quantile(ratio, 0.05),
        "opacity_after_over_before_p50": _quantile(ratio, 0.50),
        "opacity_after_over_before_p95": _quantile(ratio, 0.95),
        "opacity_reduction_mean": float(reduction.mean().cpu()),
        "opacity_relative_reduction_mean": float(relative_reduction.mean().cpu()),
        "opacity_relative_reduction_p95": _quantile(relative_reduction, 0.95),
    }

    for threshold in thresholds:
        key = _threshold_key(float(threshold))
        before_active = before > float(threshold)
        after_active = after > float(threshold)
        before_count = before_active.sum()
        after_count = after_active.sum()
        before_frac = before_active.float().mean()
        after_frac = after_active.float().mean()
        gate_pruned = before_active & ~after_active
        gate_pruned_count = gate_pruned.sum()
        stats[f"opacity_before_active_frac_gt_{key}"] = float(before_frac.cpu())
        stats[f"opacity_before_active_count_gt_{key}"] = float(before_count.cpu())
        stats[f"opacity_after_active_frac_gt_{key}"] = float(after_frac.cpu())
        stats[f"opacity_after_active_count_gt_{key}"] = float(after_count.cpu())
        stats[f"opacity_active_frac_delta_gt_{key}"] = float(
            (after_frac - before_frac).cpu()
        )
        stats[f"opacity_gate_pruned_frac_gt_{key}"] = float(
            gate_pruned.float().mean().cpu()
        )
        if int(before_count.detach().cpu()) > 0:
            stats[f"opacity_gate_pruned_cond_frac_gt_{key}"] = float(
                (gate_pruned_count.float() / before_count.float()).cpu()
            )
        else:
            stats[f"opacity_gate_pruned_cond_frac_gt_{key}"] = 0.0
        stats[f"opacity_before_compression_factor_gt_{key}"] = (
            _safe_compression_factor(count, before_count)
        )
        stats[f"opacity_after_compression_factor_gt_{key}"] = (
            _safe_compression_factor(count, after_count)
        )

    return stats


def latent_tail_stats(
    x: Tensor,
    channel_dim: int = 1,
    eps: float = 1e-6,
) -> dict[str, float]:
    """
    Compute coordinate tail statistics for a latent feature map.

    Args:
        x: Tensor with channel dimension, typically [B, C, H, W].
        channel_dim: Dimension used as the channel axis.
        eps: Numerical stability epsilon.
    """
    x = _to_float_tensor(x)
    max_abs = x.abs().amax(dim=channel_dim)
    rms = torch.sqrt(x.square().mean(dim=channel_dim) + eps)
    max_rms = max_abs / rms

    mean = x.mean(dim=channel_dim, keepdim=True)
    centered = x - mean
    var = centered.square().mean(dim=channel_dim) + eps
    kurtosis = centered.pow(4).mean(dim=channel_dim) / var.square()

    return {
        "mean_max_rms": float(max_rms.mean().cpu()),
        "p95_max_rms": _quantile(max_rms, 0.95),
        "p99_max_rms": _quantile(max_rms, 0.99),
        "mean_kurtosis": float(kurtosis.mean().cpu()),
        "p95_kurtosis": _quantile(kurtosis, 0.95),
        "p99_kurtosis": _quantile(kurtosis, 0.99),
    }


def gaussian_tail_stats(
    opacities: Tensor,
    scales: Tensor,
    tau_alpha: float = 0.95,
    tau_scale: float | None = None,
    tau_volume: float | None = None,
    adaptive_tau_scale_quantile: float | None = 0.95,
) -> dict[str, float]:
    """
    Compute opacity/scale tail statistics after GaussianAdapter.

    Args:
        opacities: Tensor with arbitrary shape, optionally ending in singleton dim.
        scales: Tensor whose last dimension is xyz scale, [..., 3].
        tau_alpha: Opacity tail threshold.
        tau_scale: Optional max-scale tail threshold.
        tau_volume: Optional volume tail threshold.
        adaptive_tau_scale_quantile: If tau_scale is not provided, use this
            scale quantile as the threshold for scale_tail and floater_proxy.
    """
    opacity = _to_float_tensor(opacities).flatten()
    scale = _to_float_tensor(scales).reshape(-1, 3)
    scale_max = scale.amax(dim=-1)
    volume = scale.prod(dim=-1)

    stats = {
        "opacity_mean": float(opacity.mean().cpu()),
        "opacity_p95": _quantile(opacity, 0.95),
        "opacity_p99": _quantile(opacity, 0.99),
        "opacity_tail_095": float((opacity > tau_alpha).float().mean().cpu()),
        "active_ratio@0.05": float((opacity > 0.05).float().mean().cpu()),
        "active_ratio@0.10": float((opacity > 0.10).float().mean().cpu()),
        "active_ratio@0.30": float((opacity > 0.30).float().mean().cpu()),
        "active_ratio@0.50": float((opacity > 0.50).float().mean().cpu()),
        "active_ratio@0.70": float((opacity > 0.70).float().mean().cpu()),
        "scale_max_mean": float(scale_max.mean().cpu()),
        "scale_max_p95": _quantile(scale_max, 0.95),
        "scale_max_p99": _quantile(scale_max, 0.99),
        "volume_p95": _quantile(volume, 0.95),
        "volume_p99": _quantile(volume, 0.99),
    }

    if tau_scale is None and adaptive_tau_scale_quantile is not None:
        tau_scale = _quantile(scale_max, adaptive_tau_scale_quantile)
        stats["scale_tail_threshold_source_quantile"] = float(
            adaptive_tau_scale_quantile
        )

    if tau_scale is not None:
        stats["scale_tail_threshold"] = float(tau_scale)
        stats["scale_tail"] = float((scale_max > tau_scale).float().mean().cpu())
        stats["floater_proxy"] = float(
            ((opacity > tau_alpha) & (scale_max > tau_scale)).float().mean().cpu()
        )

    if tau_volume is not None:
        stats["volume_tail"] = float((volume > tau_volume).float().mean().cpu())

    stats.update(opacity_pruning_stats(opacities))
    return stats


def reliability_stats(reliability: Tensor) -> dict[str, float]:
    """Summarize Gaussian reliability gates dumped by the encoder."""
    r = _to_float_tensor(reliability).flatten()
    return {
        "reliability_mean": float(r.mean().cpu()),
        "reliability_min": float(r.min().cpu()),
        "reliability_p05": _quantile(r, 0.05),
        "reliability_p50": _quantile(r, 0.50),
        "reliability_p95": _quantile(r, 0.95),
        "reliability_max": float(r.max().cpu()),
        "reliability_below_050": float((r < 0.50).float().mean().cpu()),
        "reliability_below_080": float((r < 0.80).float().mean().cpu()),
        "reliability_below_090": float((r < 0.90).float().mean().cpu()),
        "reliability_below_095": float((r < 0.95).float().mean().cpu()),
    }


def merge_stats(list_of_dicts: Iterable[dict[str, float]]) -> dict[str, float]:
    stats = list(list_of_dicts)
    if not stats:
        return {}

    keys = sorted({key for item in stats for key in item})
    merged = {}
    for key in keys:
        values = [item[key] for item in stats if key in item]
        merged[key] = float(sum(values) / len(values))
    return merged


def save_stats_json(
    stats: dict[str, float] | dict[str, dict[str, float]],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
