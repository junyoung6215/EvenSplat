from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from .preconditioner_cfg import cfg_get
from .preconditioner_math import fwht_last_dim, is_power_of_two
from .preconditioner_stats import summarize_latent_pair


class RandomizedHadamardPreconditioner(nn.Module):
    """Fixed randomized Hadamard preconditioner for BCHW latent feature maps."""

    def __init__(
        self,
        dim: int = 256,
        enabled: bool = True,
        mix: float = 0.0,
        target_mix: float = 1.0,
        warmup_steps: int = 5000,
        seed: int = 0,
        collect_stats: bool = False,
    ) -> None:
        super().__init__()
        if not is_power_of_two(dim):
            raise ValueError(f"Hadamard dimension must be a power of two, got {dim}.")

        generator = torch.Generator()
        generator.manual_seed(seed)
        permutation = torch.randperm(dim, generator=generator)
        sign_bits = torch.randint(0, 2, (dim,), generator=generator)
        signs = sign_bits.float().mul(2.0).sub(1.0)

        self.dim = dim
        self.enabled = enabled
        self.mix = float(mix)
        self.target_mix = float(target_mix)
        self.warmup_steps = int(warmup_steps)
        self.collect_stats = collect_stats
        self.current_step = 0
        self.latest_stats: dict[str, float] = {}

        self.register_buffer("permutation", permutation, persistent=False)
        self.register_buffer("signs", signs, persistent=False)

    def set_step(self, global_step: int) -> None:
        self.current_step = int(global_step)

    @property
    def effective_mix(self) -> float:
        if self.warmup_steps <= 0:
            alpha = 1.0
        else:
            alpha = min(1.0, max(0.0, self.current_step / self.warmup_steps))
        return self.mix + alpha * (self.target_mix - self.mix)

    def _rotate(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [B, C, H, W], got {tuple(x.shape)}.")
        if x.shape[1] != self.dim:
            raise ValueError(f"Expected channel dim {self.dim}, got {x.shape[1]}.")

        original_dtype = x.dtype
        x_float = x.float()
        y = x_float[:, self.permutation, :, :]
        y = y * self.signs.view(1, self.dim, 1, 1)
        y = y.permute(0, 2, 3, 1).contiguous()
        y = fwht_last_dim(y) / math.sqrt(self.dim)
        y = y.permute(0, 3, 1, 2).contiguous()
        return y.to(dtype=original_dtype)

    def _collect_stats(self, x: Tensor, y: Tensor, mix: float) -> None:
        self.latest_stats = summarize_latent_pair(x, y, mix)

    def forward(self, x: Tensor) -> Tensor:
        if not self.enabled:
            if self.collect_stats:
                self._collect_stats(x, x, 0.0)
            return x

        mix = self.effective_mix
        if mix == 0.0:
            y = x
        else:
            rotated = self._rotate(x)
            if mix == 1.0:
                y = rotated
            else:
                y = (1.0 - mix) * x + mix * rotated

        if self.collect_stats:
            self._collect_stats(x, y, mix)
        return y


def build_latent_preconditioner(cfg: Any, dim: int = 256) -> nn.Module:
    if cfg is None or not bool(cfg_get(cfg, "enabled", False)):
        return nn.Identity()

    mode = cfg_get(cfg, "mode", "hadamard")
    if mode != "hadamard":
        raise ValueError(f"Unsupported orthogonal preconditioner mode: {mode}")

    return RandomizedHadamardPreconditioner(
        dim=int(cfg_get(cfg, "dim", dim)),
        enabled=True,
        mix=float(cfg_get(cfg, "mix", 0.0)),
        target_mix=float(cfg_get(cfg, "target_mix", 1.0)),
        warmup_steps=int(cfg_get(cfg, "warmup_steps", 5000)),
        seed=int(cfg_get(cfg, "seed", 0)),
        collect_stats=bool(cfg_get(cfg, "collect_stats", False)),
    )
