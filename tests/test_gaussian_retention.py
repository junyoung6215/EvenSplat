from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _test_support import add_function_tests, annotation_only_jaxtyping

with annotation_only_jaxtyping():
    from src.model.gaussian_retention import (
        GaussianRetention,
        GaussianRetentionCfg,
        deterministic_random_scores,
        flatten_gaussian_scores,
        gather_gaussians,
    )
    from src.model.types import Gaussians


def make_gaussians(batch: int = 1, count: int = 5) -> Gaussians:
    values = torch.arange(batch * count, dtype=torch.float32).reshape(batch, count)
    return Gaussians(
        means=values[..., None].expand(batch, count, 3).clone(),
        covariances=values[..., None, None].expand(batch, count, 3, 3).clone(),
        harmonics=values[..., None, None].expand(batch, count, 3, 2).clone(),
        opacities=values.clone(),
    )


def test_disabled_module_is_exact_identity() -> None:
    gaussians = make_gaussians()
    module = GaussianRetention(GaussianRetentionCfg(enabled=False))
    result = module(gaussians, torch.randn(1, 5))

    assert result.gaussians is gaussians
    assert result.indices is None
    assert not result.applied
    assert result.input_count == result.output_count == 5
    assert not list(module.parameters())
    assert not module.state_dict()


def test_enabled_full_ratio_is_exact_identity() -> None:
    gaussians = make_gaussians()
    module = GaussianRetention(
        GaussianRetentionCfg(enabled=True, mode="fixed_ratio", keep_ratio=1.0)
    )
    result = module(gaussians, torch.randn(1, 5))

    assert result.gaussians is gaussians
    assert result.indices is None
    assert not result.applied
    assert result.input_count == result.output_count == 5


def test_stage_switches_and_start_step_are_independent() -> None:
    module = GaussianRetention(
        GaussianRetentionCfg(
            enabled=True,
            apply_during_train=True,
            apply_during_eval=False,
            start_step=10,
        )
    )

    assert not module.applies("train", 9)
    assert module.applies("train", 10)
    assert not module.applies("eval", 100)


def test_deterministic_random_scores_do_not_touch_global_rng() -> None:
    reference = torch.zeros(2, 8)
    first = deterministic_random_scores(reference, seed=7, selection_id=11)
    second = deterministic_random_scores(reference, seed=7, selection_id=11)
    different = deterministic_random_scores(reference, seed=7, selection_id=12)
    torch.testing.assert_close(first, second)
    assert not torch.equal(first, different)

    torch.manual_seed(123)
    expected = torch.rand(4)
    torch.manual_seed(123)
    deterministic_random_scores(reference, seed=1, selection_id=2)
    actual = torch.rand(4)
    torch.testing.assert_close(actual, expected)


def test_fixed_ratio_selects_same_indices_for_every_field() -> None:
    gaussians = make_gaussians(batch=2, count=5)
    scores = torch.tensor(
        [
            [0.1, 0.8, 0.2, 0.9, 0.3],
            [0.8, 0.1, 0.9, 0.2, 0.3],
        ]
    )
    module = GaussianRetention(
        GaussianRetentionCfg(enabled=True, mode="fixed_ratio", keep_ratio=0.4)
    )
    result = module(gaussians, scores)

    expected_indices = torch.tensor([[1, 3], [0, 2]])
    torch.testing.assert_close(result.indices, expected_indices)
    expected = gather_gaussians(gaussians, expected_indices)
    torch.testing.assert_close(result.gaussians.means, expected.means)
    torch.testing.assert_close(result.gaussians.covariances, expected.covariances)
    torch.testing.assert_close(result.gaussians.harmonics, expected.harmonics)
    torch.testing.assert_close(result.gaussians.opacities, expected.opacities)
    assert result.output_count == 2
    assert result.kept_fraction == 0.4


def test_elastic_all_selected_is_exact_identity() -> None:
    gaussians = make_gaussians(count=4)
    module = GaussianRetention(
        GaussianRetentionCfg(enabled=True, mode="elastic", threshold=0.0)
    )
    result = module(gaussians, torch.tensor([[0.4, 0.1, 0.3, 0.2]]))

    assert result.gaussians is gaussians
    assert result.indices is None
    assert not result.applied


def test_dense_score_map_matches_surface_and_sample_order() -> None:
    score_map = torch.tensor([[[[[[0.1], [0.2]]], [[[0.3], [0.4]]]]]])
    scores = flatten_gaussian_scores(
        score_map,
        expected_batch=1,
        expected_gaussians=8,
        name="reliability",
    )

    torch.testing.assert_close(
        scores,
        torch.tensor([[0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4]]),
    )


def test_per_pixel_score_map_repeats_for_surface_and_sample_axes() -> None:
    score_map = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]]])
    scores = flatten_gaussian_scores(
        score_map,
        expected_batch=1,
        expected_gaussians=8,
        name="external",
    )
    torch.testing.assert_close(
        scores,
        torch.tensor([[0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4]]),
    )


def test_selection_preserves_gradients_to_selected_gaussians_only() -> None:
    means = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3)
    means.requires_grad_()
    gaussians = Gaussians(
        means=means,
        covariances=torch.zeros(1, 5, 3, 3),
        harmonics=torch.zeros(1, 5, 3, 1),
        opacities=torch.ones(1, 5),
    )
    scores = torch.tensor([[0.1, 0.9, 0.2, 0.8, 0.3]], requires_grad=True)
    module = GaussianRetention(
        GaussianRetentionCfg(enabled=True, mode="fixed_count", keep_count=2)
    )
    result = module(gaussians, scores)
    result.gaussians.means.sum().backward()

    assert means.grad is not None
    assert torch.count_nonzero(means.grad[0, 1]) == 3
    assert torch.count_nonzero(means.grad[0, 3]) == 3
    assert torch.count_nonzero(means.grad[0, 0]) == 0
    assert torch.count_nonzero(means.grad[0, 2]) == 0
    assert torch.count_nonzero(means.grad[0, 4]) == 0
    assert scores.grad is None


def test_opacity_scoring_keeps_selected_opacity_gradients() -> None:
    gaussians = make_gaussians(count=5)
    opacities = gaussians.opacities.detach().clone().requires_grad_()
    gaussians.opacities = opacities
    module = GaussianRetention(
        GaussianRetentionCfg(enabled=True, mode="fixed_count", keep_count=2)
    )
    result = module(gaussians, opacities)
    result.gaussians.opacities.sum().backward()

    torch.testing.assert_close(
        opacities.grad,
        torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0]]),
    )


def test_elastic_threshold_respects_global_cap() -> None:
    gaussians = make_gaussians(count=5)
    scores = torch.tensor([[0.1, 0.7, 0.6, 0.95, 0.8]])
    module = GaussianRetention(
        GaussianRetentionCfg(
            enabled=True,
            mode="elastic",
            threshold=0.5,
            max_gaussians=2,
        )
    )
    result = module(gaussians, scores)

    torch.testing.assert_close(result.indices, torch.tensor([[3, 4]]))
    assert result.output_count == 2


def test_elastic_threshold_keeps_best_when_nothing_passes() -> None:
    gaussians = make_gaussians(count=4)
    scores = torch.tensor([[0.1, 0.2, 0.4, 0.3]])
    module = GaussianRetention(
        GaussianRetentionCfg(enabled=True, mode="elastic", threshold=0.9)
    )
    result = module(gaussians, scores)

    torch.testing.assert_close(result.indices, torch.tensor([[2]]))
    assert result.output_count == 1


def test_elastic_threshold_rejects_variable_length_batch() -> None:
    gaussians = make_gaussians(batch=2, count=4)
    module = GaussianRetention(
        GaussianRetentionCfg(enabled=True, mode="elastic", threshold=0.5)
    )
    try:
        module(gaussians, torch.rand(2, 4))
    except ValueError as exc:
        assert "batch size 1" in str(exc)
    else:
        raise AssertionError("Expected elastic retention with B>1 to fail.")


def test_non_finite_scores_fail_fast() -> None:
    gaussians = make_gaussians(count=3)
    module = GaussianRetention(GaussianRetentionCfg(enabled=True))
    try:
        module(gaussians, torch.tensor([[0.1, float("nan"), 0.2]]))
    except ValueError as exc:
        assert "NaN or Inf" in str(exc)
    else:
        raise AssertionError("Expected non-finite retention scores to fail.")


def test_fixed_count_requires_an_explicit_budget() -> None:
    try:
        GaussianRetention(
            GaussianRetentionCfg(enabled=True, mode="fixed_count")
        )
    except ValueError as exc:
        assert "keep_count is required" in str(exc)
    else:
        raise AssertionError("Expected fixed_count without keep_count to fail.")


def load_tests(loader, suite, pattern):
    return add_function_tests(suite, globals())


if __name__ == "__main__":
    test_disabled_module_is_exact_identity()
    test_enabled_full_ratio_is_exact_identity()
    test_stage_switches_and_start_step_are_independent()
    test_deterministic_random_scores_do_not_touch_global_rng()
    test_fixed_ratio_selects_same_indices_for_every_field()
    test_elastic_all_selected_is_exact_identity()
    test_dense_score_map_matches_surface_and_sample_order()
    test_per_pixel_score_map_repeats_for_surface_and_sample_axes()
    test_selection_preserves_gradients_to_selected_gaussians_only()
    test_opacity_scoring_keeps_selected_opacity_gradients()
    test_elastic_threshold_respects_global_cap()
    test_elastic_threshold_keeps_best_when_nothing_passes()
    test_elastic_threshold_rejects_variable_length_batch()
    test_non_finite_scores_fail_fast()
    test_fixed_count_requires_an_explicit_budget()
