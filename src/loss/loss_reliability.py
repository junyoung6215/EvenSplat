from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable

import torch
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossReliabilityCfg:
    enabled: bool
    weight: float
    warmup_start: int
    warmup_end: int


@dataclass
class LossReliabilityCfgWrapper:
    reliability: LossReliabilityCfg


def validate_reliability_loss_encoder(
    encoder: object,
    losses: Iterable[object],
) -> None:
    """Fail fast when an enabled reliability loss has no trainable head."""

    loss_enabled = any(
        getattr(loss_fn, "name", None) == "reliability"
        and bool(getattr(getattr(loss_fn, "cfg", None), "enabled", False))
        and float(getattr(getattr(loss_fn, "cfg", None), "weight", 0.0)) != 0.0
        for loss_fn in losses
    )
    if not loss_enabled:
        return

    reliability_cfg = getattr(encoder, "gaussian_reliability_cfg", None)
    if not bool(getattr(reliability_cfg, "enabled", False)):
        raise ValueError(
            "An enabled reliability loss requires "
            "model.encoder.gaussian_reliability.enabled=true."
        )
    if bool(getattr(reliability_cfg, "detach_gate", False)):
        warnings.warn(
            "The current reliability auxiliary loss is one-sided: with "
            "gaussian_reliability.detach_gate=true, it has no reconstruction "
            "counter-gradient and may drive reliability gates toward r_min. "
            "Use this setting only as a diagnostic ablation or with an "
            "additional balancing objective.",
            UserWarning,
            stacklevel=2,
        )


class LossReliability(Loss[LossReliabilityCfg, LossReliabilityCfgWrapper]):
    last_info: dict[str, Tensor | float] | None

    def __init__(self, cfg: LossReliabilityCfgWrapper) -> None:
        super().__init__(cfg)
        self.last_info = None

    def _warmup_factor(self, global_step: int) -> float:
        start = int(self.cfg.warmup_start)
        end = int(self.cfg.warmup_end)
        if global_step < start:
            return 0.0
        if end <= start:
            return 1.0
        return min(max((global_step - start) / float(end - start), 0.0), 1.0)

    def is_active(self, global_step: int) -> bool:
        """Whether this step needs metadata and the extra context rendering."""

        return (
            bool(self.cfg.enabled)
            and float(self.cfg.weight) != 0.0
            and self._warmup_factor(global_step) > 0.0
        )

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        del gaussians
        zero = prediction.color.new_zeros(())
        self.last_info = None
        warmup = self._warmup_factor(global_step)
        if not self.is_active(global_step):
            return zero

        metadata = batch.get("_reliability")
        if metadata is None:
            raise RuntimeError(
                "reliability loss is enabled, but ModelWrapper did not attach "
                "context residual reliability intermediates. Ensure the encoder "
                "has gaussian_reliability.enabled=true and the loss is enabled "
                "only during training."
            )

        reliability = metadata["reliability"]
        opacity = metadata["opacity_before_reliability"].detach()
        residual_risk = metadata["residual_risk"].detach()
        valid_mask = metadata["valid_mask"].detach().to(dtype=reliability.dtype)

        if not torch.isfinite(residual_risk).all():
            raise RuntimeError("reliability residual_risk contains non-finite values.")
        if not torch.isfinite(reliability).all():
            raise RuntimeError("reliability gate contains non-finite values.")

        valid_count = valid_mask.sum().clamp_min(1.0)
        weighted_risk = reliability * opacity * residual_risk * valid_mask
        raw_loss = weighted_risk.sum() / valid_count
        loss = raw_loss * float(self.cfg.weight) * warmup

        self.last_info = {
            "raw_loss": raw_loss.detach(),
            "weight": float(self.cfg.weight) * warmup,
            "warmup_factor": warmup,
            "valid_fraction": valid_mask.mean().detach(),
            "risk_mean": (
                (residual_risk * valid_mask).sum().detach()
                / valid_count.detach()
            ),
            "reliability_mean": (
                (reliability.detach() * valid_mask).sum()
                / valid_count.detach()
            ),
        }
        context_residual_mean = metadata.get("context_residual_mean")
        if isinstance(context_residual_mean, Tensor):
            self.last_info["context_residual_mean"] = context_residual_mean.detach()
        return loss
