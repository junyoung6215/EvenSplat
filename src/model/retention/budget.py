"""Optional budget loss that trains the reliability head to rank for pruning.

Not part of the final method (r filter + pruning-aware fine-tuning); kept for
the learned-r ablation.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .cells import stratified_scores


def budget_weight(weight: float, step: int, soft_steps: int, warmup: bool, hard_scale: float) -> float:
    """Budget loss weight at a training step: optional ramp, then hard scale."""

    if step >= soft_steps:
        return weight * hard_scale
    if warmup:
        return weight * step / soft_steps
    return weight


def budget_loss(
    reliability: Tensor,
    cell_ids: Tensor,
    cells: int,
    keep_ratio: float,
    r_min: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Pull the Gaussians each cell would drop to r_min and the kept ones to 1.

    The split follows the stratified top-k of the detached reliability. Both
    sides are needed: reliability multiplies opacity, so a one-sided push lets
    every gate shrink together while base opacity compensates, and the gap
    between kept and dropped never opens. Also returns detached kept/dropped
    reliability means for monitoring.
    """

    count = reliability.shape[1]
    keys = stratified_scores(reliability.detach(), cell_ids, cells)
    kept = max(1, min(count, int(round(count * float(keep_ratio)))))
    dropped = torch.ones_like(reliability, dtype=torch.bool)
    dropped.scatter_(1, torch.topk(keys, kept, dim=1).indices, False)
    detached = reliability.detach()
    stats = {"kept_mean": detached.masked_select(~dropped).mean()}
    if not bool(dropped.any()):
        return (1.0 - reliability).mean(), stats
    stats["dropped_mean"] = detached.masked_select(dropped).mean()
    loss = (reliability - float(r_min)).masked_select(dropped).mean()
    loss = loss + (1.0 - reliability).masked_select(~dropped).mean()
    return loss, stats
