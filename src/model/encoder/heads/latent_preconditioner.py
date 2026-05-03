from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn


def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _fwht_last_dim(x: Tensor) -> Tensor:
    """Unnormalized Fast Walsh-Hadamard Transform on the last dimension."""
    n = x.shape[-1]
    if not _is_power_of_two(n):
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


def _max_rms(x: Tensor, channel_dim: int = 1, eps: float = 1e-6) -> Tensor:
    max_abs = x.abs().amax(dim=channel_dim)
    rms = torch.sqrt(x.square().mean(dim=channel_dim) + eps)
    return max_abs / rms


def _kurtosis(x: Tensor, channel_dim: int = 1, eps: float = 1e-6) -> Tensor:
    mean = x.mean(dim=channel_dim, keepdim=True)
    centered = x - mean
    var = centered.square().mean(dim=channel_dim) + eps
    return centered.pow(4).mean(dim=channel_dim) / var.square()


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
        if not _is_power_of_two(dim):
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
        y = _fwht_last_dim(y) / math.sqrt(self.dim)
        y = y.permute(0, 3, 1, 2).contiguous()
        return y.to(dtype=original_dtype)

    def _collect_stats(self, x: Tensor, y: Tensor, mix: float) -> None:
        with torch.no_grad():
            x_float = x.detach().float()
            y_float = y.detach().float()
            input_max_rms = _max_rms(x_float).mean()
            output_max_rms = _max_rms(y_float).mean()
            input_kurtosis = _kurtosis(x_float).mean()
            output_kurtosis = _kurtosis(y_float).mean()
            self.latest_stats = {
                "input_max_rms": float(input_max_rms.cpu()),
                "output_max_rms": float(output_max_rms.cpu()),
                "input_kurtosis": float(input_kurtosis.cpu()),
                "output_kurtosis": float(output_kurtosis.cpu()),
                "effective_mix": float(mix),
            }

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
    if cfg is None or not bool(_cfg_get(cfg, "enabled", False)):
        return nn.Identity()

    mode = _cfg_get(cfg, "mode", "hadamard")
    if mode != "hadamard":
        raise ValueError(f"Unsupported orthogonal preconditioner mode: {mode}")

    return RandomizedHadamardPreconditioner(
        dim=int(_cfg_get(cfg, "dim", dim)),
        enabled=True,
        mix=float(_cfg_get(cfg, "mix", 0.0)),
        target_mix=float(_cfg_get(cfg, "target_mix", 1.0)),
        warmup_steps=int(_cfg_get(cfg, "warmup_steps", 5000)),
        seed=int(_cfg_get(cfg, "seed", 0)),
        collect_stats=bool(_cfg_get(cfg, "collect_stats", False)),
    )
