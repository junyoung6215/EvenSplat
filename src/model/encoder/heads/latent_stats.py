from __future__ import annotations

import torch
from torch import Tensor


def _quantile(x: Tensor, q: float) -> float:
    return float(torch.quantile(x.flatten(), q).cpu())


def max_rms(x: Tensor, channel_dim: int = 1, eps: float = 1e-6) -> Tensor:
    max_abs = x.abs().amax(dim=channel_dim)
    rms = torch.sqrt(x.square().mean(dim=channel_dim) + eps)
    return max_abs / rms


def kurtosis(x: Tensor, channel_dim: int = 1, eps: float = 1e-6) -> Tensor:
    mean = x.mean(dim=channel_dim, keepdim=True)
    centered = x - mean
    var = centered.square().mean(dim=channel_dim) + eps
    return centered.pow(4).mean(dim=channel_dim) / var.square()


def channel_outlier_stats(
    x: Tensor,
    prefix: str,
    channel_dim: int = 1,
    topk: int = 8,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Summarize per-location channel outliers without dumping full maps.

    For each spatial location, this computes:
    - olp_score: max_c |x_c| / rms(x)
    - top1_channel: argmax_c |x_c|
    - top1_value: signed value at top1_channel
    - topk_energy_ratio: top-k channel energy / total channel energy
    """
    if x.ndim < 2:
        return {}

    x_float = x.detach().float()
    channel_count = int(x_float.shape[channel_dim])
    k = min(int(topk), channel_count)

    abs_x = x_float.abs()
    max_abs, top1_channel = abs_x.max(dim=channel_dim)
    top1_value = x_float.gather(channel_dim, top1_channel.unsqueeze(channel_dim)).squeeze(
        channel_dim
    )
    rms = torch.sqrt(x_float.square().mean(dim=channel_dim) + eps)
    olp_score = max_abs / rms

    energy = x_float.square()
    total_energy = energy.sum(dim=channel_dim) + eps
    topk_energy = energy.topk(k, dim=channel_dim).values.sum(dim=channel_dim)
    topk_energy_ratio = topk_energy / total_energy

    top1_flat = top1_channel.flatten().detach().cpu()
    counts = torch.bincount(top1_flat, minlength=channel_count).float()
    mode_count, mode_channel = counts.max(dim=0)
    mode_freq = mode_count / max(1, int(top1_flat.numel()))

    top1_abs = top1_value.abs()
    return {
        f"{prefix}_olp_score_mean": float(olp_score.mean().cpu()),
        f"{prefix}_olp_score_p95": _quantile(olp_score, 0.95),
        f"{prefix}_olp_score_p99": _quantile(olp_score, 0.99),
        f"{prefix}_olp_score_max": float(olp_score.max().cpu()),
        f"{prefix}_top1_channel_mode": float(mode_channel.cpu()),
        f"{prefix}_top1_channel_mode_freq": float(mode_freq.cpu()),
        f"{prefix}_top1_value_mean": float(top1_value.mean().cpu()),
        f"{prefix}_top1_value_abs_mean": float(top1_abs.mean().cpu()),
        f"{prefix}_top1_value_abs_p95": _quantile(top1_abs, 0.95),
        f"{prefix}_top1_value_abs_p99": _quantile(top1_abs, 0.99),
        f"{prefix}_top1_value_abs_max": float(top1_abs.max().cpu()),
        f"{prefix}_topk_energy_ratio_mean": float(topk_energy_ratio.mean().cpu()),
        f"{prefix}_topk_energy_ratio_p95": _quantile(topk_energy_ratio, 0.95),
        f"{prefix}_topk_energy_ratio_p99": _quantile(topk_energy_ratio, 0.99),
        f"{prefix}_topk_energy_ratio_max": float(topk_energy_ratio.max().cpu()),
    }


def channel_outlier_maps(
    x: Tensor,
    prefix: str,
    channel_dim: int = 1,
    topk: int = 8,
    eps: float = 1e-6,
) -> dict[str, Tensor | int | list[int] | str]:
    """Return dense per-location OLP diagnostic maps without storing full y."""
    if x.ndim < 2:
        return {}

    x_float = x.detach().float()
    channel_count = int(x_float.shape[channel_dim])
    k = min(int(topk), channel_count)

    abs_x = x_float.abs()
    max_abs, top1_channel = abs_x.max(dim=channel_dim)
    top1_value = x_float.gather(channel_dim, top1_channel.unsqueeze(channel_dim)).squeeze(
        channel_dim
    )
    rms = torch.sqrt(x_float.square().mean(dim=channel_dim) + eps)
    olp_score = max_abs / rms

    energy = x_float.square()
    total_energy = energy.sum(dim=channel_dim) + eps
    topk_energy = energy.topk(k, dim=channel_dim).values.sum(dim=channel_dim)
    topk_energy_ratio = topk_energy / total_energy

    return {
        "prefix": prefix,
        "shape": list(x_float.shape),
        "channel_dim": int(channel_dim),
        "channel_count": channel_count,
        "topk": k,
        f"{prefix}_olp_score": olp_score.detach().cpu(),
        f"{prefix}_top1_channel": top1_channel.detach().to(torch.int16).cpu(),
        f"{prefix}_top1_value": top1_value.detach().cpu(),
        f"{prefix}_topk_energy_ratio": topk_energy_ratio.detach().cpu(),
    }


def channel_outlier_top_locations(
    x: Tensor,
    prefix: str,
    channel_dim: int = 1,
    topk: int = 8,
    max_locations: int = 256,
    eps: float = 1e-6,
) -> dict[str, object]:
    """Return compact raw outlier records for the highest-score locations."""
    if x.ndim < 2 or max_locations <= 0:
        return {}

    x_float = x.detach().float()
    channel_count = int(x_float.shape[channel_dim])
    k = min(int(topk), channel_count)

    abs_x = x_float.abs()
    max_abs, top1_channel = abs_x.max(dim=channel_dim)
    top1_value = x_float.gather(channel_dim, top1_channel.unsqueeze(channel_dim)).squeeze(
        channel_dim
    )
    rms = torch.sqrt(x_float.square().mean(dim=channel_dim) + eps)
    olp_score = max_abs / rms

    energy = x_float.square()
    total_energy = energy.sum(dim=channel_dim) + eps
    topk_energy = energy.topk(k, dim=channel_dim).values.sum(dim=channel_dim)
    topk_energy_ratio = topk_energy / total_energy

    flat_score = olp_score.flatten()
    n = min(int(max_locations), int(flat_score.numel()))
    if n <= 0:
        return {}
    values, flat_indices = torch.topk(flat_score, n)

    spatial_shape = list(olp_score.shape)
    flat_top1_channel = top1_channel.flatten()
    flat_top1_value = top1_value.flatten()
    flat_topk_ratio = topk_energy_ratio.flatten()

    entries = []
    for rank, (score, flat_index) in enumerate(zip(values.cpu(), flat_indices.cpu())):
        flat = int(flat_index.item())
        rem = flat
        coords = []
        for size in reversed(spatial_shape):
            coords.append(rem % int(size))
            rem //= int(size)
        coords.reverse()
        entries.append(
            {
                "rank": rank,
                "flat_index": flat,
                "coord": coords,
                f"{prefix}_olp_score": float(score.item()),
                f"{prefix}_top1_channel": int(flat_top1_channel[flat].cpu().item()),
                f"{prefix}_top1_value": float(flat_top1_value[flat].cpu().item()),
                f"{prefix}_topk_energy_ratio": float(flat_topk_ratio[flat].cpu().item()),
            }
        )

    return {
        "prefix": prefix,
        "shape": list(x_float.shape),
        "channel_dim": int(channel_dim),
        "channel_count": channel_count,
        "topk": k,
        "max_locations": n,
        "entries": entries,
    }


def summarize_latent_pair(x: Tensor, y: Tensor, mix: float) -> dict[str, float]:
    with torch.no_grad():
        x_float = x.detach().float()
        y_float = y.detach().float()
        stats = {
            "input_max_rms": float(max_rms(x_float).mean().cpu()),
            "output_max_rms": float(max_rms(y_float).mean().cpu()),
            "input_kurtosis": float(kurtosis(x_float).mean().cpu()),
            "output_kurtosis": float(kurtosis(y_float).mean().cpu()),
            "effective_mix": float(mix),
        }
        stats.update(channel_outlier_stats(y_float, "output"))
        return stats
