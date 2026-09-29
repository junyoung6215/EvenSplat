from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def image_detail_demand(images: Tensor, strength: float = 1.0) -> Tensor:
    """Return positive [B,V,H,W] demand from RGB [0,1], jointly scaled over views.

    This is an image-detail heuristic, not a learned geometry/capacity estimate.
    strength=0 and constant images yield ones, preserving global score ordering.
    """
    if images.ndim != 5 or images.shape[2] != 3 or min(images.shape[-2:]) < 2:
        raise ValueError("Demand images must be [B,V,3,H,W] with H,W >= 2.")
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("Demand strength must be finite and non-negative.")
    if not bool(torch.isfinite(images).all()) or bool(((images < -1e-5) | (images > 1+1e-5)).any()):
        raise ValueError("Image detail demand requires finite RGB values in [0,1].")
    rgb = images.detach().float()
    dx = (rgb[..., 1:] - rgb[..., :-1]).abs().mean(dim=2)
    dy = (rgb[..., 1:, :] - rgb[..., :-1, :]).abs().mean(dim=2)
    detail = F.pad(dx, (0, 1, 0, 0)) + F.pad(dy, (0, 0, 0, 1))
    scale = detail.flatten(1).amax(dim=1).clamp_min(1e-8).reshape(-1, 1, 1, 1)
    return 1.0 + strength * detail / scale
