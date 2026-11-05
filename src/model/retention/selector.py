"""Pre-render Gaussian retention: keep a fixed fraction of predicted Gaussians."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from ..types import Gaussians
from .cells import GaussianGridLayout, gaussian_cell_ids, grid_scores, stratified_scores
from .scores import flatten_gaussian_scores, r_filter_scores, random_scores

ScoreSource = Literal["opacity", "base_opacity", "reliability", "random", "r_filter", "grid"]
SCORE_SOURCES = ("opacity", "base_opacity", "reliability", "random", "r_filter", "grid")
# Score sources read from the encoder's visualization dump, and the key they read.
ENCODER_MAPS = {
    "base_opacity": "opacities_before_reliability",
    "reliability": "reliability",
    "r_filter": "reliability",
}


@dataclass
class GaussianRetentionCfg:
    """Disabled by default, which leaves the NoPoSplat tensors untouched."""

    enabled: bool = False
    apply_during_train: bool = True
    apply_during_eval: bool = True
    # opacity | base_opacity (before the reliability gate) | reliability |
    # random | r_filter (drop r < r_threshold first, rest at random) |
    # grid (fixed Bayer-ordered pixel positions in every cell)
    score_source: ScoreSource = "opacity"
    keep_ratio: float = 0.5
    # 0: one global top-k. >0: keep keep_ratio of every cell_size x cell_size
    # pixel cell of every view (stratified retention).
    cell_size: int = 0
    r_threshold: float = 0.5
    random_seed: int = 0
    # >0: each training step keeps a ratio drawn from [train_keep_ratio_min, keep_ratio].
    train_keep_ratio_min: float = 0.0
    # Training steps before pruning starts (all Gaussians are rendered until then).
    start_step: int = 0
    # Learned r (ablation): >0 adds a loss that pushes reliability of the
    # Gaussians outside the budget toward r_min and of the kept ones toward 1.
    budget_weight: float = 0.0
    # Ramp budget_weight from 0 over the first start_step steps.
    budget_warmup: bool = False
    # Budget weight multiplier once pruning starts.
    budget_hard_scale: float = 1.0
    log_diagnostics: bool = True
    diagnostics_every_n_steps: int = 100


@dataclass(frozen=True)
class RetentionResult:
    gaussians: Gaussians
    indices: Tensor | None  # [B, K] kept indices in original order; None = all kept
    input_count: int
    output_count: int
    scores: Tensor | None = None
    cell_ids: Tensor | None = None

    @property
    def applied(self) -> bool:
        return self.output_count < self.input_count

    @property
    def kept_fraction(self) -> float:
        return self.output_count / self.input_count if self.input_count > 0 else 1.0


def gather_gaussians(gaussians: Gaussians, indices: Tensor) -> Gaussians:
    """Gather every Gaussian field with the same per-batch ``[B, K]`` indices."""

    batch, count = gaussians.opacities.shape
    if indices.ndim != 2 or indices.shape[0] != batch or indices.dtype != torch.long:
        raise ValueError(f"Retention indices must be long [B={batch}, K], got {indices.shape}.")
    if indices.numel() == 0 or int(indices.min()) < 0 or int(indices.max()) >= count:
        raise ValueError(f"Retention indices must be a non-empty subset of [0, {count}).")

    def gather(field: Tensor) -> Tensor:
        index = indices.reshape(*indices.shape, *[1] * (field.ndim - 2))
        return field.gather(1, index.expand(*indices.shape, *field.shape[2:]))

    return Gaussians(
        means=gather(gaussians.means),
        covariances=gather(gaussians.covariances),
        harmonics=gather(gaussians.harmonics),
        opacities=gather(gaussians.opacities),
    )


def selection_mask(scores: Tensor, indices: Tensor | None) -> Tensor:
    if indices is None:
        return torch.ones_like(scores, dtype=torch.bool)
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, indices, True)


def score_statistics(scores: Tensor, indices: Tensor | None) -> dict[str, Tensor]:
    detached = scores.detach()
    mask = selection_mask(detached, indices)
    stats = {
        "score_mean": detached.mean(),
        "score_std": detached.float().std(unbiased=False),
        "selected_score_mean": detached.masked_select(mask).mean(),
    }
    if not bool(mask.all()):
        stats["rejected_score_mean"] = detached.masked_select(~mask).mean()
    return stats


class GaussianRetention(nn.Module):
    """Select a budgeted subset of already-predicted Gaussians.

    Owns no parameters. Indices come from detached scores, while gradients
    through the gathered Gaussian attributes stay intact for pruning-aware
    fine-tuning.
    """

    def __init__(self, cfg: GaussianRetentionCfg | None = None) -> None:
        super().__init__()
        self.cfg = cfg or GaussianRetentionCfg()
        self._validate_cfg()

    def _validate_cfg(self) -> None:
        cfg = self.cfg
        if cfg.score_source not in SCORE_SOURCES:
            raise ValueError(f"Unsupported retention score_source: {cfg.score_source!r}.")
        if not 0.0 < float(cfg.keep_ratio) <= 1.0:
            raise ValueError(f"gaussian_retention.keep_ratio must be in (0, 1], got {cfg.keep_ratio}.")
        if not 0.0 <= float(cfg.train_keep_ratio_min) <= float(cfg.keep_ratio):
            raise ValueError("gaussian_retention.train_keep_ratio_min must be in [0, keep_ratio].")
        size = int(cfg.cell_size)
        if size < 0:
            raise ValueError("gaussian_retention.cell_size must be nonnegative.")
        if cfg.score_source == "grid" and (size <= 0 or size & (size - 1)):
            raise ValueError("score_source=grid needs a power-of-two cell_size.")
        if int(cfg.start_step) < 0 or int(cfg.diagnostics_every_n_steps) <= 0:
            raise ValueError("Invalid gaussian_retention start_step or diagnostics interval.")
        for name in ("budget_weight", "budget_hard_scale"):
            value = float(getattr(cfg, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"gaussian_retention.{name} must be finite and nonnegative.")
        if cfg.budget_warmup and int(cfg.start_step) == 0:
            raise ValueError("gaussian_retention.budget_warmup ramps over start_step, which must be > 0.")

    def applies(self, stage: Literal["train", "eval"], global_step: int) -> bool:
        if not self.cfg.enabled:
            return False
        if stage == "train":
            return bool(self.cfg.apply_during_train) and global_step >= int(self.cfg.start_step)
        if stage == "eval":
            return bool(self.cfg.apply_during_eval)
        raise ValueError(f"Unsupported retention stage: {stage!r}.")

    @property
    def encoder_map(self) -> str | None:
        """The encoder visualization-dump key the score source needs, if any."""

        return ENCODER_MAPS.get(self.cfg.score_source)

    def train_keep_ratio(self, selection_id: int) -> float | None:
        """Random per-step budget in [train_keep_ratio_min, keep_ratio], or None."""

        low = float(self.cfg.train_keep_ratio_min)
        if low <= 0.0:
            return None
        generator = torch.Generator().manual_seed(int(self.cfg.random_seed) * 1_000_003 + int(selection_id))
        return low + float(torch.rand((), generator=generator)) * (float(self.cfg.keep_ratio) - low)

    @staticmethod
    def identity(gaussians: Gaussians) -> RetentionResult:
        # A literal pass-through: no gathers, copies, or device synchronization.
        count = int(gaussians.opacities.shape[1])
        return RetentionResult(gaussians, None, count, count)

    def scores(
        self,
        gaussians: Gaussians,
        layout: GaussianGridLayout,
        encoder_map: Tensor | None,
        selection_id: int,
    ) -> Tensor:
        opacities = gaussians.opacities
        source = self.cfg.score_source
        if source == "opacity":
            return opacities
        if source == "grid":
            scores = grid_scores(layout, int(self.cfg.cell_size), opacities.device)
            return scores.to(opacities.dtype).unsqueeze(0).expand_as(opacities).contiguous()
        if source == "random":
            return random_scores(opacities, int(self.cfg.random_seed), selection_id)
        if encoder_map is None:
            raise RuntimeError(
                f"score_source={source!r} needs the encoder map {self.encoder_map!r}; "
                "enable model.encoder.gaussian_reliability."
            )
        scores = flatten_gaussian_scores(encoder_map, *opacities.shape, name=self.encoder_map)
        scores = scores.to(device=opacities.device, dtype=opacities.dtype)
        if source == "r_filter":
            random = random_scores(opacities, int(self.cfg.random_seed), selection_id)
            return r_filter_scores(scores, random, float(self.cfg.r_threshold))
        return scores

    def forward(
        self,
        gaussians: Gaussians,
        layout: GaussianGridLayout,
        encoder_map: Tensor | None = None,
        selection_id: int = 0,
        keep_ratio: float | None = None,
    ) -> RetentionResult:
        if not self.cfg.enabled:
            return self.identity(gaussians)
        batch, total = gaussians.opacities.shape
        if layout.gaussian_count != total:
            raise ValueError(f"Grid layout has {layout.gaussian_count} Gaussians, got {total}.")
        scores = self.scores(gaussians, layout, encoder_map, selection_id)
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("Gaussian retention scores contain NaN or Inf values.")
        keys = scores.detach()
        cell_ids = None
        cell_size = int(self.cfg.cell_size)
        if cell_size > 0:
            cell_ids = gaussian_cell_ids(layout, cell_size, keys.device)
            keys = stratified_scores(keys, cell_ids, layout.cell_count(cell_size))
        ratio = float(self.cfg.keep_ratio if keep_ratio is None else keep_ratio)
        kept = max(1, min(total, int(round(total * ratio))))
        if kept >= total:
            return RetentionResult(gaussians, None, total, total, scores, cell_ids)
        # Selection must not otherwise reorder the surviving Gaussians.
        indices = torch.topk(keys, kept, dim=1, sorted=True).indices.sort(dim=1).values
        return RetentionResult(
            gather_gaussians(gaussians, indices), indices, total, kept, scores, cell_ids
        )
