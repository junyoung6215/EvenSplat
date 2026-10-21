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
        retention_selection_mask,
    )
    from src.model.types import Gaussians
from src.model.local_cell_retention import (  # noqa: E402
    GaussianGridLayout,
    LocalCellAllocator,
    LocalCellRetentionCfg,
    gaussian_cell_ids,
    pool_cell_demand,
    retention_distribution_statistics,
)


def make_gaussians(count: int, batch: int = 1) -> Gaussians:
    values = torch.arange(batch * count, dtype=torch.float32).reshape(batch, count)
    return Gaussians(
        means=values[..., None].expand(batch, count, 3).clone(),
        covariances=values[..., None, None].expand(batch, count, 3, 3).clone(),
        harmonics=values[..., None, None].expand(batch, count, 3, 2).clone(),
        opacities=values.clone(),
    )


def test_cell_ids_follow_view_pixel_surface_sample_order() -> None:
    layout = GaussianGridLayout(
        views=2,
        height=2,
        width=4,
        surfaces=1,
        samples_per_surface=2,
    )
    ids = gaussian_cell_ids(layout, cell_size=2, device=torch.device("cpu"))
    torch.testing.assert_close(
        ids,
        torch.tensor(
            [
                0, 0, 0, 0, 1, 1, 1, 1,
                0, 0, 0, 0, 1, 1, 1, 1,
                2, 2, 2, 2, 3, 3, 3, 3,
                2, 2, 2, 2, 3, 3, 3, 3,
            ]
        ),
    )


def test_full_resolution_demand_is_average_pooled() -> None:
    layout = GaussianGridLayout(1, 4, 4, 1, 1)
    demand = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    pooled = pool_cell_demand(demand, layout, cell_size=2)
    torch.testing.assert_close(
        pooled,
        torch.tensor([[[[2.5, 4.5], [10.5, 12.5]]]]),
    )


def test_unit_demand_reproduces_global_scores() -> None:
    layout = GaussianGridLayout(views=1, height=2, width=2, surfaces=1, samples_per_surface=2)
    scores = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
    allocator = LocalCellAllocator(
        LocalCellRetentionCfg(enabled=True, cell_size=1)
    )
    allocation = allocator(scores, torch.ones(1, 1, 2, 2), layout)
    torch.testing.assert_close(allocation.scores, scores)
    assert allocation.required_mask is None


def test_cell_demand_is_broadcast_to_each_gaussian_in_the_cell() -> None:
    layout = GaussianGridLayout(views=1, height=1, width=4, surfaces=1, samples_per_surface=1)
    scores = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    allocator = LocalCellAllocator(
        LocalCellRetentionCfg(enabled=True, cell_size=2)
    )
    allocation = allocator(scores, torch.tensor([[[[0.1, 2.0]]]]), layout)
    torch.testing.assert_close(
        allocation.scores,
        torch.tensor([[0.1, 0.2, 6.0, 8.0]]),
    )


def test_minimum_cell_coverage_is_combined_with_global_budget() -> None:
    layout = GaussianGridLayout(views=1, height=1, width=4, surfaces=1, samples_per_surface=1)
    scores = torch.tensor([[10.0, 9.0, 2.0, 1.0]])
    allocator = LocalCellAllocator(
        LocalCellRetentionCfg(
            enabled=True,
            cell_size=2,
            min_per_cell=1,
        )
    )
    allocation = allocator(scores, torch.ones(1, 1, 1, 2), layout)
    selector = GaussianRetention(
        GaussianRetentionCfg(enabled=True, mode="fixed_count", keep_count=2)
    )
    result = selector(
        make_gaussians(4),
        allocation.scores,
        required_mask=allocation.required_mask,
    )
    torch.testing.assert_close(result.indices, torch.tensor([[0, 2]]))


def test_minimum_cell_coverage_rejects_impossible_budget() -> None:
    layout = GaussianGridLayout(views=1, height=1, width=4, surfaces=1, samples_per_surface=1)
    allocator = LocalCellAllocator(
        LocalCellRetentionCfg(enabled=True, cell_size=2, min_per_cell=1)
    )
    scores = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    allocation = allocator(scores, torch.ones(1, 1, 1, 2), layout)
    selector = GaussianRetention(
        GaussianRetentionCfg(enabled=True, mode="fixed_count", keep_count=1)
    )
    try:
        selector(
            make_gaussians(4),
            allocation.scores,
            required_mask=allocation.required_mask,
        )
    except ValueError as exc:
        assert "exceeds the global budget" in str(exc)
    else:
        raise AssertionError("Expected an impossible minimum cell budget to fail.")


def test_elastic_selection_unions_threshold_and_minimum_coverage() -> None:
    layout = GaussianGridLayout(views=1, height=1, width=4, surfaces=1, samples_per_surface=1)
    scores = torch.tensor([[10.0, 9.0, 2.0, 1.0]])
    allocator = LocalCellAllocator(
        LocalCellRetentionCfg(enabled=True, cell_size=2, min_per_cell=1)
    )
    allocation = allocator(scores, torch.ones(1, 1, 1, 2), layout)
    selector = GaussianRetention(
        GaussianRetentionCfg(
            enabled=True,
            mode="elastic",
            threshold=5.0,
            max_gaussians=2,
        )
    )
    result = selector(
        make_gaussians(4),
        allocation.scores,
        required_mask=allocation.required_mask,
    )
    torch.testing.assert_close(result.indices, torch.tensor([[0, 2]]))


def test_minimum_cell_coverage_supports_batched_fixed_budgets() -> None:
    layout = GaussianGridLayout(views=1, height=1, width=4, surfaces=1, samples_per_surface=1)
    scores = torch.tensor(
        [
            [10.0, 9.0, 2.0, 1.0],
            [1.0, 2.0, 9.0, 10.0],
        ]
    )
    allocator = LocalCellAllocator(
        LocalCellRetentionCfg(enabled=True, cell_size=2, min_per_cell=1)
    )
    allocation = allocator(scores, torch.ones(2, 1, 1, 2), layout)
    selector = GaussianRetention(
        GaussianRetentionCfg(enabled=True, mode="fixed_count", keep_count=2)
    )
    result = selector(
        make_gaussians(4, batch=2),
        allocation.scores,
        required_mask=allocation.required_mask,
    )
    torch.testing.assert_close(result.indices, torch.tensor([[0, 2], [1, 3]]))


def test_distribution_statistics_detect_empty_cells() -> None:
    layout = GaussianGridLayout(views=1, height=1, width=4, surfaces=1, samples_per_surface=1)
    scores = torch.ones(1, 4)
    mask = retention_selection_mask(scores, torch.tensor([[0, 1]]))
    ids = gaussian_cell_ids(layout, cell_size=2, device=torch.device("cpu"))
    stats = retention_distribution_statistics(mask, layout, ids)
    torch.testing.assert_close(stats["empty_cell_fraction"], torch.tensor(0.5))
    torch.testing.assert_close(stats["cell_kept_fraction_min"], torch.tensor(0.0))
    torch.testing.assert_close(stats["cell_kept_fraction_max"], torch.tensor(1.0))


def test_pruned_candidates_still_receive_prior_gradients() -> None:
    from src.foundation.supervision import prior_reliability_loss
    from src.model.image_demand import image_detail_demand

    reliability = torch.full((1, 2, 4, 4, 1, 1), 0.6, requires_grad=True)
    gaussians = make_gaussians(32)
    gaussians.opacities = reliability.reshape(1, 32)
    gaussians.means.requires_grad_()
    layout = GaussianGridLayout(2, 4, 4, 1, 1)
    allocator = LocalCellAllocator(LocalCellRetentionCfg(enabled=True, cell_size=2))
    allocation = allocator(reliability.reshape(1, 32),
                           image_detail_demand(torch.zeros(1, 2, 3, 4, 4)), layout)
    result = GaussianRetention(GaussianRetentionCfg(enabled=True, keep_ratio=0.5))(
        gaussians, allocation.scores)
    prior_loss, _ = prior_reliability_loss(reliability, torch.zeros(1, 2, 4, 4),
        torch.ones(1, 2, 4, 4, dtype=torch.bool), torch.ones(1, 2, 4, 4), .05)
    (result.gaussians.opacities.sum() + result.gaussians.means.sum() + prior_loss).backward()
    mask = retention_selection_mask(allocation.scores, result.indices)
    assert result.output_count == 16
    # Real gather graph: geometry gradients only on selected candidates.
    assert torch.all(gaussians.means.grad[~mask] == 0)
    assert torch.all(gaussians.means.grad[mask] != 0)
    # Dense auxiliary graph: discarded gates still receive a correction.
    assert torch.all(reliability.grad.reshape(1, 32)[~mask] > 0)


def load_tests(loader, suite, pattern):
    return add_function_tests(suite, globals())


if __name__ == "__main__":
    test_cell_ids_follow_view_pixel_surface_sample_order()
    test_full_resolution_demand_is_average_pooled()
    test_unit_demand_reproduces_global_scores()
    test_cell_demand_is_broadcast_to_each_gaussian_in_the_cell()
    test_minimum_cell_coverage_is_combined_with_global_budget()
    test_minimum_cell_coverage_rejects_impossible_budget()
    test_elastic_selection_unions_threshold_and_minimum_coverage()
    test_minimum_cell_coverage_supports_batched_fixed_budgets()
    test_distribution_statistics_detect_empty_cells()
    test_pruned_candidates_still_receive_prior_gradients()
