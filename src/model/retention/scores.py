"""Per-Gaussian ranking scores for retention."""
from __future__ import annotations

import torch
from torch import Tensor


def flatten_gaussian_scores(value: Tensor, batch: int, gaussians: int, name: str) -> Tensor:
    """Align a dense encoder map (``[B,V,H,W(,S)(,1)]`` or ``[B,G(,1)]``) with ``[B,G]``.

    When the adapter emits several samples per surface, each source score is
    repeated consecutively to match view / pixel / surface / sample order.
    """

    if value.ndim in (3, 6) and value.shape[-1] == 1:
        value = value.squeeze(-1)
    if value.ndim not in (2, 4, 5):
        raise ValueError(f"Retention score '{name}' has unsupported shape {tuple(value.shape)}.")
    scores = value.reshape(value.shape[0], -1)
    if scores.shape[0] != batch:
        raise ValueError(f"Retention score '{name}' batch mismatch: {scores.shape[0]} != {batch}.")
    count = int(scores.shape[1])
    if count != gaussians:
        if count <= 0 or gaussians % count:
            raise ValueError(
                f"Retention score '{name}' ({count}) cannot align with {gaussians} Gaussians."
            )
        scores = scores.repeat_interleave(gaussians // count, dim=1)
    return scores


def random_scores(reference: Tensor, seed: int, selection_id: int) -> Tensor:
    """Reproducible uniform scores that leave PyTorch's global RNG untouched."""

    mixed_seed = (int(seed) * 1_000_003 + int(selection_id) * 97_409 + 17) % (2**63 - 1)
    generator = torch.Generator(device=reference.device)
    generator.manual_seed(mixed_seed)
    scores = torch.rand(
        reference.shape, generator=generator, device=reference.device, dtype=torch.float32
    )
    return scores.to(dtype=reference.dtype)


def r_filter_scores(reliability: Tensor, random: Tensor, threshold: float) -> Tensor:
    """Rank Gaussians with reliability below ``threshold`` last, the rest at random.

    Random scores lie in [0, 1), so the +1 offset keeps every Gaussian at or
    above the threshold ahead of every Gaussian below it.
    """

    return random + (reliability >= threshold).to(random.dtype)
