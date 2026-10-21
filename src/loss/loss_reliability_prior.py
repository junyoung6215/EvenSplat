"""Opt-in teacher supervision; the legacy residual loss is intentionally unchanged."""
import math
from dataclasses import dataclass

from ..foundation.supervision import prior_reliability_loss
from .loss import Loss


@dataclass
class LossReliabilityPriorCfg:
    enabled: bool = False
    weight: float = 1e-3
    warmup_start: int = 500
    warmup_end: int = 1500
    allow_mock: bool = False


@dataclass
class LossReliabilityPriorCfgWrapper:
    reliability_prior: LossReliabilityPriorCfg


def validate_prior_loss_encoder(encoder, losses, anchor_selection_cfg):
    active = {loss.name for loss in losses
              if getattr(loss.cfg, "enabled", False) and getattr(loss.cfg, "weight", 0) != 0}
    if "reliability_prior" not in active:
        return
    if "reliability" in active:
        raise ValueError("Choose legacy reliability OR reliability_prior, not both, for a clean ablation.")
    if not getattr(getattr(encoder, "gaussian_reliability_cfg", None), "enabled", False):
        raise ValueError("reliability_prior requires gaussian_reliability.enabled=true.")
    if anchor_selection_cfg.enabled:
        raise ValueError("Prior MVP preserves source order and first-view canonical: disable anchor_selection.")


class LossReliabilityPrior(Loss[LossReliabilityPriorCfg, LossReliabilityPriorCfgWrapper]):
    def __init__(self, cfg):
        super().__init__(cfg)
        if not math.isfinite(self.cfg.weight) or self.cfg.weight < 0:
            raise ValueError("reliability_prior weight must be finite and nonnegative.")
        if self.cfg.warmup_start < 0 or self.cfg.warmup_end < self.cfg.warmup_start:
            raise ValueError("Require 0 <= warmup_start <= warmup_end.")
        self.last_info = None

    def _warmup_factor(self, step):
        start, end = self.cfg.warmup_start, self.cfg.warmup_end
        if step < start:
            return 0.0
        return 1.0 if end == start else min((step - start) / (end - start), 1.0)

    def is_active(self, step):
        return self.cfg.enabled and self.cfg.weight > 0 and self._warmup_factor(step) > 0

    def forward(self, prediction, batch, gaussians, global_step):
        self.last_info = None
        if not self.is_active(global_step):
            return prediction.color.new_zeros(())
        metadata, prior = batch.get("_reliability_prior"), batch.get("_foundation")
        if metadata is None or prior is None:
            raise RuntimeError("Active reliability_prior needs dense encoder gates and training prior maps.")
        if "is_mock" not in prior:
            raise ValueError("Prior provenance flag is_mock is required.")
        if bool(prior["is_mock"].any()) and not self.cfg.allow_mock:
            raise ValueError("Mock labels must not be used in real teacher experiments.")
        raw, info = prior_reliability_loss(
            metadata["reliability"], prior["teacher_target"], prior["teacher_valid"],
            prior["teacher_weight"], metadata["r_min"])
        weight = self.cfg.weight * self._warmup_factor(global_step)
        self.last_info = {**info, "weight": weight,
                          "mock_fraction": prior["is_mock"].float().mean().detach()}
        return raw * weight
