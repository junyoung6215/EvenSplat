from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _test_support import add_function_tests, require_test_dependencies

require_test_dependencies("jaxtyping", "beartype", direct_run=__name__ == "__main__")
from jaxtyping import install_import_hook

with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.loss.loss_reliability import (
        LossReliability,
        LossReliabilityCfg,
        LossReliabilityCfgWrapper,
        validate_reliability_loss_encoder,
    )
    from src.model.decoder.decoder import DecoderOutput
    from src.model.encoder.heads.reliability_cfg import opacity_reliability_gate
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
    assert math.isclose(
        float(reliability.grad[0, 0, 0]),
        0.16,
        rel_tol=1e-6,
        abs_tol=1e-6,
    )
    assert math.isclose(
        float(reliability.grad[0, 1, 0]),
        0.0,
        rel_tol=1e-6,
        abs_tol=1e-6,
    )
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


def test_zero_warmup_factor_does_not_require_metadata(tmp_path):
    del tmp_path
    loss_fn = make_loss(weight=1.0, warmup_start=10, warmup_end=20)

    before = loss_fn.forward(make_prediction(), {}, make_gaussians(), 5)

    assert math.isclose(float(before), 0.0, rel_tol=1e-6, abs_tol=1e-6)
    assert not loss_fn.is_active(5)
    assert loss_fn.last_info is None


def test_reliability_loss_requires_enabled_encoder():
    class DummyEncoder:
        gaussian_reliability_cfg = type("Cfg", (), {"enabled": False})()

    class EnabledEncoder:
        gaussian_reliability_cfg = type("Cfg", (), {"enabled": True})()

    class DetachedEncoder:
        gaussian_reliability_cfg = type(
            "Cfg",
            (),
            {"enabled": True, "detach_gate": True},
        )()

    loss_fn = make_loss()
    try:
        validate_reliability_loss_encoder(DummyEncoder(), [loss_fn])
    except ValueError as exc:
        assert "gaussian_reliability.enabled=true" in str(exc)
    else:
        raise AssertionError("Expected a disabled reliability head to fail fast.")

    validate_reliability_loss_encoder(EnabledEncoder(), [loss_fn])
    validate_reliability_loss_encoder(DummyEncoder(), [make_loss(weight=0.0)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_reliability_loss_encoder(DetachedEncoder(), [loss_fn])
    assert len(caught) == 1
    assert "may drive reliability gates toward r_min" in str(caught[0].message)


def test_detach_gate_only_detaches_opacity_path():
    reliability = torch.tensor([0.5], requires_grad=True)
    opacity = torch.tensor([2.0], requires_grad=True)
    opacity_gate = opacity_reliability_gate(reliability, detach_gate=True)

    assert not opacity_gate.requires_grad
    reconstruction_loss = (opacity * opacity_gate).sum()
    auxiliary_loss = 3.0 * reliability.sum()
    (reconstruction_loss + auxiliary_loss).backward()
    torch.testing.assert_close(reliability.grad, torch.tensor([3.0]))
    torch.testing.assert_close(opacity.grad, torch.tensor([0.5]))

    attached_reliability = torch.tensor([0.5], requires_grad=True)
    attached_opacity = torch.tensor([2.0], requires_grad=True)
    attached_gate = opacity_reliability_gate(
        attached_reliability,
        detach_gate=False,
    )
    assert attached_gate is attached_reliability
    attached_loss = (attached_opacity * attached_gate).sum()
    attached_loss = attached_loss + 3.0 * attached_reliability.sum()
    attached_loss.backward()
    torch.testing.assert_close(attached_reliability.grad, torch.tensor([5.0]))
    torch.testing.assert_close(attached_opacity.grad, torch.tensor([0.5]))


def load_tests(loader, suite, pattern):
    return add_function_tests(suite, globals())


if __name__ == "__main__":
    with TemporaryDirectory() as tmp:
        test_reliability_loss_detaches_risk_and_opacity(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_reliability_loss_warmup(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_zero_warmup_factor_does_not_require_metadata(Path(tmp))
    test_reliability_loss_requires_enabled_encoder()
    test_detach_gate_only_detaches_opacity_path()
