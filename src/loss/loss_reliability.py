from __future__ import annotations

from dataclasses import dataclass

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
        if not bool(self.cfg.enabled) or float(self.cfg.weight) == 0.0:
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
        raw_loss = (reliability * opacity * residual_risk * valid_mask).sum() / valid_count
        warmup = self._warmup_factor(global_step)
        loss = raw_loss * float(self.cfg.weight) * warmup

        self.last_info = {
            "raw_loss": raw_loss.detach(),
            "weight": float(self.cfg.weight) * warmup,
            "warmup_factor": warmup,
            "valid_fraction": valid_mask.mean().detach(),
            "risk_mean": (residual_risk * valid_mask).sum().detach() / valid_count.detach(),
            "reliability_mean": (reliability.detach() * valid_mask).sum() / valid_count.detach(),
        }
        context_residual_mean = metadata.get("context_residual_mean")
        if isinstance(context_residual_mean, Tensor):
            self.last_info["context_residual_mean"] = context_residual_mean.detach()
        return loss
