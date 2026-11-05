"""Gaussian retention unit tests: CPU tensors only, no model or dataset."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _test_support import annotation_only_jaxtyping

with annotation_only_jaxtyping():
    from src.model.retention import (
        GaussianGridLayout,
        GaussianRetention,
        GaussianRetentionCfg,
        budget_loss,
        budget_weight,
        distribution_statistics,
        flatten_gaussian_scores,
        gaussian_cell_ids,
        grid_scores,
        r_filter_scores,
        random_scores,
        selection_mask,
        stratified_scores,
    )
    from src.model.types import Gaussians

CPU = torch.device("cpu")


def gaussians(batch=1, count=16, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return Gaussians(
        means=torch.randn(batch, count, 3, generator=generator),
        covariances=torch.randn(batch, count, 3, 3, generator=generator),
        harmonics=torch.randn(batch, count, 3, 4, generator=generator),
        opacities=torch.rand(batch, count, generator=generator),
    )


def retention(**kwargs):
    return GaussianRetention(GaussianRetentionCfg(enabled=True, **kwargs))


class CellTests(unittest.TestCase):
    def test_cell_ids_follow_view_pixel_order(self):
        layout = GaussianGridLayout(views=2, height=2, width=4)
        ids = gaussian_cell_ids(layout, 2, CPU)
        self.assertEqual(ids.tolist(), [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3])
        self.assertEqual(gaussian_cell_ids(layout, 0, CPU).unique().tolist(), [0])

    def test_cell_ids_repeat_for_surfaces_and_samples(self):
        layout = GaussianGridLayout(views=1, height=2, width=2, surfaces=1, samples_per_surface=3)
        self.assertEqual(gaussian_cell_ids(layout, 1, CPU).tolist(), [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

    def test_stratified_top_k_keeps_same_share_of_every_cell(self):
        layout = GaussianGridLayout(views=1, height=4, width=4)
        ids = gaussian_cell_ids(layout, 2, CPU)
        scores = torch.arange(16.0)[None]  # global top-k would take one corner only
        keys = stratified_scores(scores, ids, layout.cell_count(2))
        kept = torch.topk(keys, 8, dim=1).indices[0]
        self.assertEqual(torch.bincount(ids[kept], minlength=4).tolist(), [2, 2, 2, 2])
        # Inside each cell the two best scores win.
        for cell in range(4):
            members = (ids == cell).nonzero().flatten()
            best = members[scores[0, members].argsort(descending=True)[:2]]
            self.assertEqual(set(best.tolist()), set(kept[ids[kept] == cell].tolist()))

    def test_grid_scores_are_a_bayer_pattern_per_cell(self):
        layout = GaussianGridLayout(views=1, height=2, width=4)
        scores = grid_scores(layout, 2, CPU).reshape(2, 4)
        self.assertEqual(scores.tolist(), [[0, -2, 0, -2], [-3, -1, -3, -1]])
        with self.assertRaises(ValueError):
            grid_scores(layout, 3, CPU)

    def test_distribution_statistics(self):
        layout = GaussianGridLayout(views=2, height=2, width=2)
        mask = torch.tensor([[True, False, True, False, True, True, True, True]])
        stats = distribution_statistics(mask, layout, gaussian_cell_ids(layout, 1, CPU))
        self.assertAlmostEqual(float(stats["view_kept_fraction_min"]), 0.5)
        self.assertAlmostEqual(float(stats["empty_cell_fraction"]), 0.25)


class ScoreTests(unittest.TestCase):
    def test_flatten_repeats_samples_per_surface(self):
        value = torch.arange(4.0).reshape(1, 1, 2, 2, 1, 1)
        self.assertEqual(flatten_gaussian_scores(value, 1, 8, "r").tolist(), [[0, 0, 1, 1, 2, 2, 3, 3]])
        with self.assertRaises(ValueError):
            flatten_gaussian_scores(value, 1, 6, "r")

    def test_random_scores_are_reproducible_and_leave_global_rng(self):
        reference = torch.zeros(2, 8)
        state = torch.random.get_rng_state()
        first = random_scores(reference, 3, 5)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(first, random_scores(reference, 3, 5)))
        self.assertFalse(torch.equal(first, random_scores(reference, 3, 6)))

    def test_r_filter_ranks_low_reliability_last(self):
        reliability = torch.tensor([[0.9, 0.1, 0.5, 0.2]])
        scores = r_filter_scores(reliability, torch.tensor([[0.0, 0.99, 0.3, 0.98]]), 0.35)
        self.assertEqual(scores.argsort(descending=True)[0, :2].tolist(), [2, 0])


class SelectorTests(unittest.TestCase):
    def test_disabled_is_a_literal_pass_through(self):
        g = gaussians()
        result = GaussianRetention()(g, GaussianGridLayout(1, 4, 4))
        self.assertIs(result.gaussians, g)
        self.assertIsNone(result.indices)
        self.assertFalse(result.applied)

    def test_keep_ratio_one_returns_identity(self):
        g = gaussians()
        result = retention(keep_ratio=1.0)(g, GaussianGridLayout(1, 4, 4))
        self.assertIs(result.gaussians, g)

    def test_global_opacity_top_k_in_original_order(self):
        g = gaussians(batch=2)
        result = retention(keep_ratio=0.25)(g, GaussianGridLayout(1, 4, 4))
        expected = torch.topk(g.opacities, 4, dim=1).indices.sort(dim=1).values
        self.assertTrue(torch.equal(result.indices, expected))
        self.assertEqual((result.input_count, result.output_count), (16, 4))
        for name in ("means", "covariances", "harmonics", "opacities"):
            field = getattr(g, name)
            index = expected.reshape(*expected.shape, *[1] * (field.ndim - 2)).expand(2, 4, *field.shape[2:])
            self.assertTrue(torch.equal(getattr(result.gaussians, name), field.gather(1, index)))

    def test_stratified_r_filter_keeps_half_of_every_cell(self):
        layout = GaussianGridLayout(views=2, height=4, width=4)
        g = gaussians(count=layout.gaussian_count)
        reliability = torch.rand(1, 2, 4, 4, 1, 1, generator=torch.Generator().manual_seed(1))
        result = retention(score_source="r_filter", cell_size=2, r_threshold=0.35)(
            g, layout, encoder_map=reliability, selection_id=3)
        ids = gaussian_cell_ids(layout, 2, CPU)
        self.assertEqual(torch.bincount(ids[result.indices[0]]).tolist(), [2] * 8)
        # Within a cell, a Gaussian below the threshold is never kept over one above it.
        flat = reliability.reshape(-1)
        kept = selection_mask(flat[None], result.indices)[0]
        for cell in range(8):
            members = ids == cell
            above = (flat >= 0.35) & members
            if int(above.sum()) >= 2:
                self.assertTrue(bool((kept & members & ~above).sum() == 0))

    def test_reliability_scores_need_the_encoder_map(self):
        with self.assertRaisesRegex(RuntimeError, "encoder map"):
            retention(score_source="reliability")(gaussians(), GaussianGridLayout(1, 4, 4))

    def test_grid_needs_power_of_two_cells(self):
        with self.assertRaises(ValueError):
            retention(score_source="grid", cell_size=0)
        with self.assertRaises(ValueError):
            retention(score_source="grid", cell_size=3)

    def test_gradients_flow_through_kept_gaussians(self):
        g = gaussians()
        g.means.requires_grad_(True)
        result = retention(keep_ratio=0.5)(g, GaussianGridLayout(1, 4, 4))
        result.gaussians.means.sum().backward()
        self.assertEqual(int((g.means.grad.abs().sum(-1) > 0).sum()), 8)

    def test_train_keep_ratio_is_reproducible_and_in_range(self):
        module = retention(keep_ratio=1.0, train_keep_ratio_min=0.25)
        ratios = [module.train_keep_ratio(step) for step in range(50)]
        self.assertTrue(all(0.25 <= r <= 1.0 for r in ratios))
        self.assertEqual(ratios, [module.train_keep_ratio(step) for step in range(50)])
        self.assertIsNone(retention().train_keep_ratio(0))

    def test_invalid_configs(self):
        for kwargs in ({"keep_ratio": 0.0}, {"score_source": "external"}, {"cell_size": -1},
                       {"train_keep_ratio_min": 0.8, "keep_ratio": 0.5}, {"budget_weight": -1.0},
                       {"budget_warmup": True}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                retention(**kwargs)

    def test_training_stage_waits_for_start_step(self):
        module = retention(start_step=10, apply_during_eval=False)
        self.assertFalse(module.applies("train", 9))
        self.assertTrue(module.applies("train", 10))
        self.assertFalse(module.applies("eval", 100))


class BudgetTests(unittest.TestCase):
    def test_budget_loss_pushes_dropped_to_r_min_and_kept_to_one(self):
        reliability = torch.tensor([[0.9, 0.2, 0.8, 0.1]], requires_grad=True)
        loss, stats = budget_loss(reliability, torch.zeros(4, dtype=torch.long), 1, 0.5, 0.05)
        loss.backward()
        self.assertAlmostEqual(float(stats["kept_mean"]), 0.85, places=5)
        self.assertAlmostEqual(float(stats["dropped_mean"]), 0.15, places=5)
        self.assertEqual(torch.sign(reliability.grad).tolist(), [[-1, 1, -1, 1]])

    def test_budget_weight_schedule(self):
        self.assertEqual(budget_weight(0.01, 50, 100, True, 0.1), 0.005)
        self.assertEqual(budget_weight(0.01, 50, 100, False, 0.1), 0.01)
        self.assertAlmostEqual(budget_weight(0.01, 100, 100, True, 0.1), 0.001)


if __name__ == "__main__":
    unittest.main()
