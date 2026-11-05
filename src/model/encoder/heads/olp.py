"""OLP-H: a fixed randomized Hadamard preconditioner on the Gaussian-head latent.

Contains the config, the fast Walsh-Hadamard transform, the module and the
helper that forwards the training step to it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .latent_stats import channel_outlier_maps, channel_outlier_top_locations, summarize_latent_pair


@dataclass
class OrthogonalPreconditionerCfg:
    enabled: bool = False
    mode: str = "hadamard"
    dim: int = 256
    mix: float = 0.0
    target_mix: float = 1.0
    warmup_steps: int = 5000
    seed: int = 0
    collect_stats: bool = False


def cfg_get(cfg: Any, name: str, default: Any) -> Any:
    """Read config fields from dataclasses, OmegaConf nodes, or plain dicts."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


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


class RandomizedHadamardPreconditioner(nn.Module):
    """Fixed randomized Hadamard preconditioner for BCHW latent feature maps.

    For each pixel feature vector x in R^C, this module applies

        Qx = H D P x / sqrt(C)

    where P is a fixed channel permutation, D is a fixed random sign diagonal,
    and H is the Walsh-Hadamard transform. The normalized transform is
    orthogonal when C is a power of two, so it preserves L2 norm at mix=1. This
    experiment uses the transform only as a Gaussian-head latent preconditioner:
    no polar transform, quantization, clipping, or learnable rotation is used.

    Recommended experiment settings:
        - Disabled/default: enabled=False.
        - Baseline latent stats: enabled=True, mix=0.0, target_mix=0.0,
          warmup_steps=0, collect_stats=True.
        - Zero-shot diagnostic: enabled=True, mix=1.0, target_mix=1.0,
          warmup_steps=0.
        - Fine-tuning: enabled=True, mix=0.0, target_mix=1.0,
          warmup_steps=5000, seed=0.

    The random permutation and sign buffers are persistent=False, so they are
    intentionally not saved in checkpoints. Reuse the same seed for training
    and evaluation when OLP-H is enabled.

    Background:
        https://en.wikipedia.org/wiki/Hadamard_transform
    """

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
        self.latest_outlier_dump: dict[str, object] = {}
        self.latest_outlier_maps: dict[str, object] = {}

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
        self.latest_outlier_dump = channel_outlier_top_locations(y, "output")
        self.latest_outlier_maps = channel_outlier_maps(y, "output")

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


def set_preconditioner_step(root: nn.Module, global_step: int) -> None:
    """Propagate a training/eval step to modules that expose set_step()."""
    for module in root.modules():
        if module is root:
            continue
        set_step = getattr(module, "set_step", None)
        if callable(set_step):
            set_step(global_step)
