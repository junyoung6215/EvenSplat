import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

olp_stats_spec = importlib.util.spec_from_file_location(
    "olp_stats",
    ROOT / "src/analysis/olp_stats.py",
)
olp_stats = importlib.util.module_from_spec(olp_stats_spec)
assert olp_stats_spec.loader is not None
olp_stats_spec.loader.exec_module(olp_stats)

preconditioner_spec = importlib.util.spec_from_file_location(
    "latent_preconditioner",
    ROOT / "src/model/encoder/heads/latent_preconditioner.py",
)
preconditioner = importlib.util.module_from_spec(preconditioner_spec)
assert preconditioner_spec.loader is not None
preconditioner_spec.loader.exec_module(preconditioner)

gaussian_tail_stats = olp_stats.gaussian_tail_stats
latent_tail_stats = olp_stats.latent_tail_stats
RandomizedHadamardPreconditioner = preconditioner.RandomizedHadamardPreconditioner


def test_disabled_returns_input_exactly():
    x = torch.randn(2, 256, 4, 4)
    module = RandomizedHadamardPreconditioner(dim=256, enabled=False)
    y = module(x)
    assert y is x
    torch.testing.assert_close(y, x, rtol=0.0, atol=0.0)


def test_norm_is_preserved_for_mix_one():
    torch.manual_seed(0)
    x = torch.randn(2, 256, 4, 4)
    module = RandomizedHadamardPreconditioner(
        dim=256,
        enabled=True,
        mix=1.0,
        target_mix=1.0,
        warmup_steps=0,
        seed=0,
    )
    y = module(x)
    torch.testing.assert_close(
        y.flatten(1).norm(dim=1),
        x.flatten(1).norm(dim=1),
        rtol=1e-5,
        atol=1e-5,
    )


def test_warmup_mix_schedule():
    module = RandomizedHadamardPreconditioner(
        dim=256,
        enabled=True,
        mix=0.2,
        target_mix=0.8,
        warmup_steps=10,
    )
    module.set_step(0)
    assert abs(module.effective_mix - 0.2) < 1e-12
    module.set_step(5)
    assert abs(module.effective_mix - 0.5) < 1e-12
    module.set_step(10)
    assert abs(module.effective_mix - 0.8) < 1e-12
    module.set_step(20)
    assert abs(module.effective_mix - 0.8) < 1e-12


def test_output_shape_equals_input_shape():
    x = torch.randn(2, 256, 7, 5)
    module = RandomizedHadamardPreconditioner(
        dim=256,
        enabled=True,
        mix=1.0,
        target_mix=1.0,
        warmup_steps=0,
    )
    assert module(x).shape == x.shape


def test_stats_helpers_return_scalars():
    x = torch.randn(2, 256, 4, 4)
    latent = latent_tail_stats(x)
    assert "mean_max_rms" in latent
    assert all(isinstance(value, float) for value in latent.values())

    opacities = torch.rand(2, 32, 1)
    scales = torch.rand(2, 32, 3)
    gaussian = gaussian_tail_stats(opacities, scales, tau_scale=0.5)
    assert "floater_proxy" in gaussian
    assert all(isinstance(value, float) for value in gaussian.values())


if __name__ == "__main__":
    test_disabled_returns_input_exactly()
    test_norm_is_preserved_for_mix_one()
    test_warmup_mix_schedule()
    test_output_shape_equals_input_shape()
    test_stats_helpers_return_scalars()
