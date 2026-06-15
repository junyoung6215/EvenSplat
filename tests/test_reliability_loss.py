from __future__ import annotations

import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from jaxtyping import install_import_hook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.loss.loss_reliability import (
        LossReliability,
        LossReliabilityCfg,
        LossReliabilityCfgWrapper,
    )
    from src.model.decoder.decoder import DecoderOutput
    from src.model.types import Gaussians


def make_prediction() -> DecoderOutput:
    return DecoderOutput(torch.zeros(1, 1, 3, 2, 2), None)


def make_gaussians() -> Gaussians:
    return Gaussians(
        means=torch.zeros(1, 2, 3),
        covariances=torch.eye(3).view(1, 1, 3, 3).expand(1, 2, 3, 3),
        harmonics=torch.zeros(1, 2, 3, 1),
        opacities=torch.zeros(1, 2),
    )


def make_loss(weight=2.0, warmup_start=0, warmup_end=0) -> LossReliability:
    cfg = LossReliabilityCfgWrapper(
        reliability=LossReliabilityCfg(
            enabled=True,
            weight=weight,
            warmup_start=warmup_start,
            warmup_end=warmup_end,
        )
    )
    return LossReliability(cfg)


def test_reliability_loss_detaches_risk_and_opacity(tmp_path):
    del tmp_path
    reliability = torch.tensor([[[0.5], [0.25]]], requires_grad=True)
    opacity = torch.tensor([[[0.2], [0.8]]], requires_grad=True)
    risk = torch.tensor([[[0.4], [0.1]]], requires_grad=True)
    valid = torch.tensor([[[1.0], [0.0]]])
    batch = {
        "_reliability": {
            "reliability": reliability,
            "opacity_before_reliability": opacity,
            "residual_risk": risk,
            "valid_mask": valid,
        }
    }

    loss = make_loss(weight=2.0).forward(make_prediction(), batch, make_gaussians(), 0)
    loss.backward()

    assert math.isclose(float(loss.detach()), 0.08, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(float(reliability.grad[0, 0, 0]), 0.16, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(float(reliability.grad[0, 1, 0]), 0.0, rel_tol=1e-6, abs_tol=1e-6)
    assert opacity.grad is None
    assert risk.grad is None


def test_reliability_loss_warmup(tmp_path):
    del tmp_path
    batch = {
        "_reliability": {
            "reliability": torch.ones(1, 1, 1),
            "opacity_before_reliability": torch.ones(1, 1, 1),
            "residual_risk": torch.ones(1, 1, 1),
            "valid_mask": torch.ones(1, 1, 1),
        }
    }
    loss_fn = make_loss(weight=1.0, warmup_start=10, warmup_end=20)

    before = loss_fn.forward(make_prediction(), batch, make_gaussians(), 5)
    middle = loss_fn.forward(make_prediction(), batch, make_gaussians(), 15)
    after = loss_fn.forward(make_prediction(), batch, make_gaussians(), 20)

    assert math.isclose(float(before), 0.0, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(float(middle), 0.5, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(float(after), 1.0, rel_tol=1e-6, abs_tol=1e-6)


if __name__ == "__main__":
    with TemporaryDirectory() as tmp:
        test_reliability_loss_detaches_risk_and_opacity(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_reliability_loss_warmup(Path(tmp))
