from __future__ import annotations

import math

import torch
from torch import Tensor

from .prior import validate_prior_maps


def prior_reliability_loss(reliability: Tensor, target: Tensor, valid: Tensor,
                           weight: Tensor, r_min: float) -> tuple[Tensor, dict[str, Tensor]]:
    """Weighted two-sided supervision of normalized reliability, over all candidates.

    target/weight/valid describe the source pixels, not NoPo canonical XYZ.
    One source-pixel target is broadcast across its surface gates.
    """
    if not math.isfinite(r_min) or not 0 <= r_min < 1:
        raise ValueError("r_min must be finite and in [0,1).")
    if reliability.ndim != 6 or reliability.shape[-1] != 1 or min(reliability.shape) < 1:
        raise ValueError("Reliability must have shape [B,V,H,W,S,1].")
    if not bool(torch.isfinite(reliability).all()):
        raise ValueError("Reliability contains NaN or Inf.")
    tolerance = 2e-3 if reliability.dtype in (torch.float16, torch.bfloat16) else 1e-6
    if bool(((reliability < r_min - tolerance) | (reliability > 1 + tolerance)).any()):
        raise ValueError("Reliability lies outside the encoder's configured range.")
    shape = tuple(reliability.shape[:4])
    validate_prior_maps(target, valid, weight, shape)
    target = target.detach().to(device=reliability.device, dtype=torch.float32)[..., None, None]
    weights = (weight.detach().to(device=reliability.device, dtype=torch.float32)
               * valid.detach().to(device=reliability.device, dtype=torch.float32))
    weights = weights[..., None, None].expand_as(reliability)
    normalized = (reliability.float() - r_min) / (1 - r_min)
    denominator = weights.sum().clamp_min(torch.finfo(torch.float32).eps)
    loss = ((normalized - target).square() * weights).sum() / denominator
    info = {"raw_loss": loss.detach(), "valid_fraction": valid.float().mean().detach(),
            "effective_weight": weights.sum().detach(),
            "target_mean": ((target * weights).sum() / denominator).detach(),
            "reliability_mean": ((normalized * weights).sum() / denominator).detach()}
    return loss, info
