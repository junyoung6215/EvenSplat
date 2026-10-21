from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _test_support import add_function_tests

HEADS = ROOT / "src/model/encoder/heads"


def _ensure_package(name: str, path: Path | None = None) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [] if path is None else [str(path)]
    sys.modules[name] = module


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# Avoid replacing real src packages during collection of unrelated tests.
_ensure_package("_olp_test_heads", HEADS)

_load_module(
    "_olp_test_heads.preconditioner_cfg",
    HEADS / "preconditioner_cfg.py",
)
_load_module(
    "_olp_test_heads.preconditioner_math",
    HEADS / "preconditioner_math.py",
)
_load_module(
    "_olp_test_heads.preconditioner_stats",
    HEADS / "preconditioner_stats.py",
)
preconditioner = _load_module(
    "_olp_test_heads.latent_preconditioner",
    HEADS / "latent_preconditioner.py",
)
olp_stats = _load_module("_olp_test_stats", ROOT / "src/analysis/olp_stats.py")

gaussian_tail_stats = olp_stats.gaussian_tail_stats
latent_tail_stats = olp_stats.latent_tail_stats
RandomizedHadamardPreconditioner = preconditioner.RandomizedHadamardPreconditioner
build_latent_preconditioner = preconditioner.build_latent_preconditioner


def test_disabled_returns_input_exactly():
    x = torch.randn(2, 256, 4, 4)
    module = RandomizedHadamardPreconditioner(dim=256, enabled=False)
    y = module(x)
    assert y is x
    torch.testing.assert_close(y, x, rtol=0.0, atol=0.0)


def test_disabled_can_collect_identity_stats():
    x = torch.randn(2, 256, 4, 4)
    module = RandomizedHadamardPreconditioner(
        dim=256,
        enabled=False,
        collect_stats=True,
    )
    y = module(x)
    assert y is x
    assert module.latest_stats["effective_mix"] == 0.0
    assert module.latest_stats["input_max_rms"] == module.latest_stats["output_max_rms"]
    assert module.latest_stats["input_kurtosis"] == module.latest_stats["output_kurtosis"]


def test_build_disabled_returns_identity_module():
    module = build_latent_preconditioner({"enabled": False}, dim=256)
    assert isinstance(module, torch.nn.Identity)


def test_invalid_hadamard_dim_raises():
    try:
        RandomizedHadamardPreconditioner(dim=255)
    except ValueError as exc:
        assert "power of two" in str(exc)
    else:
        raise AssertionError("Expected non-power-of-two dim to raise ValueError")


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


def test_same_seed_reproduces_rotation():
    x = torch.randn(2, 256, 4, 4)
    module_a = RandomizedHadamardPreconditioner(
        dim=256,
        enabled=True,
        mix=1.0,
        target_mix=1.0,
        warmup_steps=0,
        seed=123,
    )
    module_b = RandomizedHadamardPreconditioner(
        dim=256,
        enabled=True,
        mix=1.0,
        target_mix=1.0,
        warmup_steps=0,
        seed=123,
    )
    torch.testing.assert_close(module_a(x), module_b(x), rtol=0.0, atol=0.0)


def test_random_buffers_are_not_checkpoint_persistent():
    module = RandomizedHadamardPreconditioner(
        dim=256,
        enabled=True,
        mix=1.0,
        target_mix=1.0,
        warmup_steps=0,
    )
    assert "permutation" not in module.state_dict()
    assert "signs" not in module.state_dict()


def test_identity_stats_mode_collects_stats():
    x = torch.randn(2, 256, 4, 4)
    module = RandomizedHadamardPreconditioner(
        dim=256,
        enabled=True,
        mix=0.0,
        target_mix=0.0,
        warmup_steps=0,
        collect_stats=True,
    )
    y = module(x)
    assert y is x
    assert module.latest_stats["effective_mix"] == 0.0
    assert "input_max_rms" in module.latest_stats
    assert "output_max_rms" in module.latest_stats


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
    gaussian = gaussian_tail_stats(opacities, scales)
    assert "floater_proxy" in gaussian
    assert "scale_tail_threshold" in gaussian
    assert all(isinstance(value, float) for value in gaussian.values())


def load_tests(loader, suite, pattern):
    return add_function_tests(suite, globals())


if __name__ == "__main__":
    test_disabled_returns_input_exactly()
    test_disabled_can_collect_identity_stats()
    test_build_disabled_returns_identity_module()
    test_invalid_hadamard_dim_raises()
    test_norm_is_preserved_for_mix_one()
    test_same_seed_reproduces_rotation()
    test_random_buffers_are_not_checkpoint_persistent()
    test_identity_stats_mode_collects_stats()
    test_warmup_mix_schedule()
    test_output_shape_equals_input_shape()
    test_stats_helpers_return_scalars()
