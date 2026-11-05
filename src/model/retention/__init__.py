"""Gaussian retention: stratified per-cell selection with an r filter."""
from .budget import budget_loss, budget_weight
from .cells import (
    GaussianGridLayout,
    distribution_statistics,
    gaussian_cell_ids,
    grid_scores,
    stratified_scores,
)
from .scores import flatten_gaussian_scores, r_filter_scores, random_scores
from .selector import (
    SCORE_SOURCES,
    GaussianRetention,
    GaussianRetentionCfg,
    RetentionResult,
    gather_gaussians,
    score_statistics,
    selection_mask,
)
