from __future__ import annotations

import torch
from torch import Tensor


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def fwht_last_dim(x: Tensor) -> Tensor:
    """Unnormalized Fast Walsh-Hadamard Transform on the last dimension."""
    n = x.shape[-1]
    if not is_power_of_two(n):
        raise ValueError(f"FWHT requires power-of-two dimension, got {n}.")

    y = x
    prefix = x.shape[:-1]
    h = 1
    while h < n:
        y = y.reshape(*prefix, -1, h * 2)
        left = y[..., :h]
        right = y[..., h : 2 * h]
        y = torch.cat((left + right, left - right), dim=-1)
        h *= 2
    return y.reshape_as(x)
