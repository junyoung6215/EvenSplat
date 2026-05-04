from __future__ import annotations

import torch
from torch import Tensor


def max_rms(x: Tensor, channel_dim: int = 1, eps: float = 1e-6) -> Tensor:
    max_abs = x.abs().amax(dim=channel_dim)
    rms = torch.sqrt(x.square().mean(dim=channel_dim) + eps)
    return max_abs / rms


def kurtosis(x: Tensor, channel_dim: int = 1, eps: float = 1e-6) -> Tensor:
    mean = x.mean(dim=channel_dim, keepdim=True)
    centered = x - mean
    var = centered.square().mean(dim=channel_dim) + eps
    return centered.pow(4).mean(dim=channel_dim) / var.square()


def summarize_latent_pair(x: Tensor, y: Tensor, mix: float) -> dict[str, float]:
    with torch.no_grad():
        x_float = x.detach().float()
        y_float = y.detach().float()
        return {
            "input_max_rms": float(max_rms(x_float).mean().cpu()),
            "output_max_rms": float(max_rms(y_float).mean().cpu()),
            "input_kurtosis": float(kurtosis(x_float).mean().cpu()),
            "output_kurtosis": float(kurtosis(y_float).mean().cpu()),
            "effective_mix": float(mix),
        }
