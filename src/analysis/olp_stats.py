from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor


def _to_float_tensor(x: Tensor) -> Tensor:
    return x.detach().float()


def _quantile(x: Tensor, q: float) -> float:
    return float(torch.quantile(x.flatten(), q).cpu())


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

    return stats


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


def save_stats_json(stats: dict[str, float], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
