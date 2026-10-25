from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .local_cell_retention import LocalCellRetentionCfg, stratified_cell_scores
from .types import Gaussians


RetentionMode = Literal["fixed_ratio", "fixed_count", "elastic"]
RetentionScoreSource = Literal[
    "opacity",
    "base_opacity",
    "reliability",
    "random",
    "reliability_filtered",
    "external",
]


@dataclass
class GaussianRetentionCfg:
    """Configuration for optional pre-render Gaussian retention.

    The feature is disabled by default so existing checkpoints and execution
    paths remain unchanged. Fixed-size modes support batched training. Elastic
    mode intentionally supports batch size one only because ``Gaussians`` uses
    a dense ``[B, G, ...]`` representation.
    """

    enabled: bool = False
    apply_during_train: bool = True
    apply_during_eval: bool = True
    score_source: RetentionScoreSource = "opacity"
    external_score_key: str = "retention_score"
    random_seed: int = 0
    # reliability_filtered: Gaussians with reliability below this floor are
    # dropped first; the rest of the budget is chosen at random.
    reliability_floor: float = 0.5
    # Budget loss on reliability during training: Gaussians that the current
    # ranking (stratified by local_cells.cell_size) would drop are pushed toward
    # r_min and the kept ones toward 1. Before start_step all Gaussians are still rendered, so the
    # reconstruction loss decides which ones can fade and r learns a ranking
    # for pruning. 0 disables it.
    budget_weight: float = 0.0
    # Ramp the budget weight linearly from 0 over the soft phase (start_step)
    # instead of applying it in full from the first step.
    budget_warmup: bool = False
    # Budget weight multiplier once hard pruning starts (after start_step).
    budget_hard_scale: float = 1.0
    mode: RetentionMode = "fixed_ratio"
    keep_ratio: float = 0.5
    keep_count: int | None = None
    threshold: float = 0.5
    max_gaussians: int | None = None
    start_step: int = 0
    log_diagnostics: bool = True
    diagnostics_every_n_steps: int = 100
    local_cells: LocalCellRetentionCfg = field(
        default_factory=LocalCellRetentionCfg
    )


@dataclass(frozen=True)
class GaussianRetentionResult:
    gaussians: Gaussians
    indices: Tensor | None
    input_count: int
    output_count: int
    applied: bool

    @property
    def kept_fraction(self) -> float:
        if self.input_count <= 0:
            return 1.0
        return self.output_count / self.input_count


def _validate_gaussians(gaussians: Gaussians) -> tuple[int, int]:
    fields = {
        "means": gaussians.means,
        "covariances": gaussians.covariances,
        "harmonics": gaussians.harmonics,
        "opacities": gaussians.opacities,
    }
    if gaussians.opacities.ndim != 2:
        raise ValueError(
            "Gaussian opacities must have shape [B, G], got "
            f"{tuple(gaussians.opacities.shape)}."
        )
    batch, count = gaussians.opacities.shape
    if count <= 0:
        raise ValueError("Gaussian retention requires at least one Gaussian.")
    for name, field in fields.items():
        if field.ndim < 2 or field.shape[:2] != (batch, count):
            raise ValueError(
                f"Gaussian field '{name}' must start with [B, G]={(batch, count)}, "
                f"got {tuple(field.shape)}."
            )
    return batch, count


def flatten_gaussian_scores(
    value: Tensor,
    expected_batch: int,
    expected_gaussians: int,
    name: str,
) -> Tensor:
    """Align a dense encoder score map with flattened Gaussian ordering.

    Encoder maps are flattened in ``view, pixel, surface`` order. When the
    adapter emits multiple samples per surface, each source score is repeated
    consecutively to match ``view, pixel, surface, sample`` ordering.
    """

    if value.ndim == 2:
        scores = value
    elif value.ndim == 3 and value.shape[-1] == 1:
        scores = value.squeeze(-1)
    elif value.ndim == 4:
        scores = value.reshape(value.shape[0], -1)
    elif value.ndim == 5:
        scores = value.reshape(value.shape[0], -1)
    elif value.ndim == 6 and value.shape[-1] == 1:
        scores = value.squeeze(-1).reshape(value.shape[0], -1)
    else:
        raise ValueError(
            f"Retention score '{name}' has unsupported shape {tuple(value.shape)}. "
            "Expected [B,G], [B,G,1], [B,V,H,W], [B,V,H,W,S], "
            "or [B,V,H,W,S,1]."
        )

    if scores.shape[0] != expected_batch:
        raise ValueError(
            f"Retention score '{name}' batch mismatch: expected "
            f"{expected_batch}, got {scores.shape[0]}."
        )
    score_count = int(scores.shape[1])
    if score_count != expected_gaussians:
        if score_count <= 0 or expected_gaussians % score_count != 0:
            raise ValueError(
                f"Retention score '{name}' cannot align with Gaussians: "
                f"score count={score_count}, Gaussian count={expected_gaussians}."
            )
        scores = scores.repeat_interleave(
            expected_gaussians // score_count,
            dim=1,
        )
    return scores


def deterministic_random_scores(
    reference: Tensor,
    seed: int,
    selection_id: int,
) -> Tensor:
    """Generate reproducible scores without modifying PyTorch's global RNG."""

    if reference.ndim != 2:
        raise ValueError(
            f"Random score reference must have shape [B,G], got {reference.shape}."
        )
    modulus = 2**63 - 1
    mixed_seed = (
        int(seed) * 1_000_003 + int(selection_id) * 97_409 + 17
    ) % modulus
    generator = torch.Generator(device=reference.device)
    generator.manual_seed(mixed_seed)
    scores = torch.rand(
        reference.shape,
        generator=generator,
        device=reference.device,
        dtype=torch.float32,
    )
    return scores.to(dtype=reference.dtype)


def filtered_random_scores(reliability: Tensor, random: Tensor, floor: float) -> Tensor:
    """Rank Gaussians below the reliability floor last, and the rest at random.

    Random scores lie in [0, 1), so the +1 offset keeps every Gaussian at or
    above the floor ahead of every Gaussian below it.
    """

    return random + (reliability >= floor).to(random.dtype)


def retention_budget_weight(cfg: GaussianRetentionCfg, step: int) -> float:
    """Budget loss weight at a training step: optional ramp, then hard scale."""

    weight = float(cfg.budget_weight)
    soft = int(cfg.start_step)
    if step >= soft:
        return weight * float(cfg.budget_hard_scale)
    if cfg.budget_warmup:
        return weight * step / soft
    return weight


def retention_budget_loss(
    reliability: Tensor,
    cell_ids: Tensor,
    cells: int,
    keep_ratio: float,
    r_min: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Pull the Gaussians each cell would drop to r_min and the kept ones to 1.

    The split follows the stratified top-k of the detached reliability. Both
    sides are needed: reliability multiplies opacity, so a one-sided push lets
    every gate shrink together while base opacity compensates, and the gap
    between kept and dropped never opens. Also returns detached kept/dropped
    reliability means for monitoring.
    """

    batch, count = reliability.shape
    keys = stratified_cell_scores(reliability.detach(), cell_ids, cells)
    kept = max(1, min(count, int(round(count * float(keep_ratio)))))
    indices = torch.topk(keys, kept, dim=1).indices
    dropped = ~retention_selection_mask(reliability, indices)
    detached = reliability.detach()
    stats = {"kept_mean": detached.masked_select(~dropped).mean()}
    if not bool(dropped.any()):
        return (1.0 - reliability).mean(), stats
    stats["dropped_mean"] = detached.masked_select(dropped).mean()
    loss = (reliability - float(r_min)).masked_select(dropped).mean()
    loss = loss + (1.0 - reliability).masked_select(~dropped).mean()
    return loss, stats


def gather_gaussians(gaussians: Gaussians, indices: Tensor) -> Gaussians:
    """Gather every Gaussian field with the same per-batch indices."""

    batch, count = _validate_gaussians(gaussians)
    if indices.ndim != 2 or indices.shape[0] != batch:
        raise ValueError(
            f"Retention indices must have shape [B, K] with B={batch}, got "
            f"{tuple(indices.shape)}."
        )
    if indices.dtype != torch.long:
        raise ValueError(f"Retention indices must be torch.long, got {indices.dtype}.")
    if indices.numel() == 0:
        raise ValueError("Gaussian retention cannot select zero Gaussians.")
    if int(indices.min().item()) < 0 or int(indices.max().item()) >= count:
        raise ValueError(
            f"Retention indices must be in [0, {count}), got an invalid index."
        )

    def gather(field: Tensor) -> Tensor:
        view_shape = [indices.shape[0], indices.shape[1]] + [1] * (field.ndim - 2)
        expand_shape = [indices.shape[0], indices.shape[1], *field.shape[2:]]
        gather_index = indices.reshape(view_shape).expand(expand_shape)
        return field.gather(1, gather_index)

    return Gaussians(
        means=gather(gaussians.means),
        covariances=gather(gaussians.covariances),
        harmonics=gather(gaussians.harmonics),
        opacities=gather(gaussians.opacities),
    )


def merge_into_kept(
    gaussians: Gaussians,
    indices: Tensor,
    cell_ids: Tensor,
) -> Gaussians:
    """Fold every dropped Gaussian into the nearest kept Gaussian of its cell.

    Kept Gaussians become opacity-weighted moment matches of their group, so
    pruning removes primitives without leaving uncovered gaps. Cells must have
    equal size. Returns the merged Gaussians at ``indices``.
    """

    batch, count = _validate_gaussians(gaussians)
    sizes = torch.bincount(cell_ids)
    size = int(sizes[0].item())
    if bool((sizes != size).any()):
        raise ValueError("Gaussian merging requires equally sized cells.")
    cells = count // size
    order = torch.argsort(cell_ids, stable=True)

    def grouped(field: Tensor) -> Tensor:
        return field.index_select(1, order).reshape(
            batch, cells, size, *field.shape[2:]
        )

    kept = grouped(retention_selection_mask(gaussians.opacities, indices))
    means = grouped(gaussians.means)
    covariances = grouped(gaussians.covariances)
    harmonics = grouped(gaussians.harmonics)
    opacities = grouped(gaussians.opacities)

    # assign[b, c, i, j]: member i is absorbed by kept member j (kept -> itself).
    distance = (means[..., :, None, :] - means[..., None, :, :]).square().sum(-1)
    distance = distance.detach().masked_fill(~kept[..., None, :], torch.inf)
    nearest = distance.argmin(-1)
    has_target = torch.isfinite(distance.amin(-1))
    assign = F.one_hot(nearest, size).to(opacities.dtype)
    assign = assign * has_target[..., None].to(opacities.dtype)

    weights = assign * (opacities[..., :, None] + 1e-6)
    total = weights.sum(-2).clamp_min(1e-12)
    merged_means = torch.einsum("bcij,bcix->bcjx", weights, means) / total[..., None]
    offset = means[..., :, None, :] - merged_means[..., None, :, :]
    spread = covariances[..., :, None, :, :] + offset[..., :, None] * offset[..., None, :]
    merged_covariances = (
        torch.einsum("bcij,bcijxy->bcjxy", weights, spread) / total[..., None, None]
    )
    merged_harmonics = (
        torch.einsum("bcij,bcixd->bcjxd", weights, harmonics) / total[..., None, None]
    )
    transmittance = torch.log1p(-opacities.clamp(max=1.0 - 1e-6))
    merged_opacities = 1.0 - torch.exp(torch.einsum("bcij,bci->bcj", assign, transmittance))

    inverse = torch.argsort(order)

    def flat(field: Tensor) -> Tensor:
        return field.reshape(batch, count, *field.shape[3:]).index_select(1, inverse)

    return gather_gaussians(
        Gaussians(
            means=flat(merged_means),
            covariances=flat(merged_covariances),
            harmonics=flat(merged_harmonics),
            opacities=flat(merged_opacities),
        ),
        indices,
    )


def retention_selection_mask(
    scores: Tensor,
    indices: Tensor | None,
) -> Tensor:
    if scores.ndim != 2:
        raise ValueError(f"Retention scores must have shape [B,G], got {scores.shape}.")
    if indices is None:
        return torch.ones_like(scores, dtype=torch.bool)
    if indices.ndim != 2 or indices.shape[0] != scores.shape[0]:
        raise ValueError("Retention indices do not match the score batch.")
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return mask


def retention_score_statistics(
    scores: Tensor,
    indices: Tensor | None,
) -> dict[str, Tensor]:
    """Return compact score summaries without retaining histogram tensors."""

    detached = scores.detach()
    mask = retention_selection_mask(detached, indices)
    stats = {
        "score_mean": detached.mean(),
        "score_std": detached.float().std(unbiased=False),
        "selected_score_mean": detached.masked_select(mask).mean(),
    }
    rejected = detached.masked_select(~mask)
    if rejected.numel() > 0:
        stats["rejected_score_mean"] = rejected.mean()
    return stats


class GaussianRetention(nn.Module):
    """Select a budgeted subset of already-predicted Gaussians.

    This module owns no trainable parameters. The hard selection indices are
    computed from detached scores, while gradients through the gathered
    Gaussian attributes remain intact for pruning-aware fine-tuning.
    """

    def __init__(self, cfg: GaussianRetentionCfg | None = None) -> None:
        super().__init__()
        self.cfg = cfg or GaussianRetentionCfg()
        self._validate_cfg()

    def _validate_cfg(self) -> None:
        if self.cfg.mode not in ("fixed_ratio", "fixed_count", "elastic"):
            raise ValueError(f"Unsupported Gaussian retention mode: {self.cfg.mode!r}.")
        if self.cfg.score_source not in (
            "opacity",
            "base_opacity",
            "reliability",
            "random",
            "reliability_filtered",
            "external",
        ):
            raise ValueError(
                f"Unsupported Gaussian retention score source: {self.cfg.score_source!r}."
            )
        if not 0.0 < float(self.cfg.keep_ratio) <= 1.0:
            raise ValueError(
                "gaussian_retention.keep_ratio must be in (0, 1], got "
                f"{self.cfg.keep_ratio}."
            )
        if self.cfg.keep_count is not None and int(self.cfg.keep_count) <= 0:
            raise ValueError("gaussian_retention.keep_count must be positive when set.")
        if self.cfg.mode == "fixed_count" and self.cfg.keep_count is None:
            raise ValueError(
                "gaussian_retention.keep_count is required for fixed_count mode."
            )
        if self.cfg.max_gaussians is not None and int(self.cfg.max_gaussians) <= 0:
            raise ValueError("gaussian_retention.max_gaussians must be positive when set.")
        if not math.isfinite(float(self.cfg.threshold)):
            raise ValueError("gaussian_retention.threshold must be finite.")
        if int(self.cfg.start_step) < 0:
            raise ValueError("gaussian_retention.start_step must be non-negative.")
        if int(self.cfg.diagnostics_every_n_steps) <= 0:
            raise ValueError(
                "gaussian_retention.diagnostics_every_n_steps must be positive."
            )
        if not math.isfinite(float(self.cfg.budget_weight)) or self.cfg.budget_weight < 0:
            raise ValueError("gaussian_retention.budget_weight must be finite and nonnegative.")
        if not math.isfinite(float(self.cfg.budget_hard_scale)) or self.cfg.budget_hard_scale < 0:
            raise ValueError("gaussian_retention.budget_hard_scale must be finite and nonnegative.")
        if not self.cfg.external_score_key:
            raise ValueError("gaussian_retention.external_score_key must not be empty.")

    def applies(self, stage: Literal["train", "eval"], global_step: int) -> bool:
        if not self.cfg.enabled:
            return False
        if stage == "train":
            return bool(self.cfg.apply_during_train) and global_step >= int(
                self.cfg.start_step
            )
        if stage == "eval":
            return bool(self.cfg.apply_during_eval)
        raise ValueError(f"Unsupported Gaussian retention stage: {stage!r}.")

    @staticmethod
    def identity(gaussians: Gaussians) -> GaussianRetentionResult:
        # Keep the disabled path as a literal pass-through: no gathers, copies,
        # device synchronization, or additional tensor-value validation.
        count = int(gaussians.opacities.shape[1])
        return GaussianRetentionResult(
            gaussians=gaussians,
            indices=None,
            input_count=count,
            output_count=count,
            applied=False,
        )

    def _fixed_keep_count(self, total: int) -> int:
        if self.cfg.mode == "fixed_ratio":
            requested = int(round(total * float(self.cfg.keep_ratio)))
        elif self.cfg.mode == "fixed_count":
            if self.cfg.keep_count is None:
                raise ValueError(
                    "gaussian_retention.keep_count is required for fixed_count mode."
                )
            requested = int(self.cfg.keep_count)
        else:
            raise RuntimeError("_fixed_keep_count called for elastic mode.")
        return max(1, min(total, requested))

    def _elastic_indices(self, scores: Tensor, total: int) -> Tensor:
        if scores.shape[0] != 1:
            raise ValueError(
                "Elastic Gaussian retention currently requires batch size 1; "
                f"got B={scores.shape[0]}."
            )
        candidates = torch.nonzero(
            scores[0] >= float(self.cfg.threshold), as_tuple=False
        ).flatten()
        if candidates.numel() == 0:
            candidates = torch.topk(scores[0], 1, sorted=True).indices

        cap = total
        if self.cfg.max_gaussians is not None:
            cap = min(cap, int(self.cfg.max_gaussians))
        if candidates.numel() > cap:
            candidate_scores = scores[0].index_select(0, candidates)
            local = torch.topk(candidate_scores, cap, sorted=True).indices
            candidates = candidates.index_select(0, local)
        else:
            order = torch.argsort(
                scores[0].index_select(0, candidates), descending=True
            )
            candidates = candidates.index_select(0, order)
        return candidates.unsqueeze(0)

    @staticmethod
    def _validate_required_mask(
        required_mask: Tensor | None,
        scores: Tensor,
    ) -> Tensor | None:
        if required_mask is None:
            return None
        if required_mask.shape != scores.shape:
            raise ValueError(
                "Required retention mask must match scores: "
                f"expected {tuple(scores.shape)}, got {tuple(required_mask.shape)}."
            )
        if required_mask.dtype != torch.bool:
            raise ValueError("Required retention mask must have dtype torch.bool.")
        if required_mask.device != scores.device:
            raise ValueError("Required retention mask must be on the score device.")
        return required_mask

    @staticmethod
    def _fixed_indices(
        scores: Tensor,
        kept: int,
        required_mask: Tensor | None,
    ) -> Tensor:
        if required_mask is None:
            return torch.topk(scores, kept, dim=1, sorted=True).indices

        selected: list[Tensor] = []
        for batch_index in range(scores.shape[0]):
            required = torch.nonzero(
                required_mask[batch_index],
                as_tuple=False,
            ).flatten()
            if required.numel() > kept:
                raise ValueError(
                    "Minimum local-cell retention exceeds the global budget: "
                    f"required={required.numel()}, budget={kept}."
                )
            remaining = kept - int(required.numel())
            if remaining > 0:
                candidates = scores[batch_index].masked_fill(
                    required_mask[batch_index],
                    -torch.inf,
                )
                extra = torch.topk(candidates, remaining, sorted=True).indices
                required = torch.cat((required, extra), dim=0)
            selected.append(required)
        return torch.stack(selected, dim=0)

    def _elastic_indices_with_required(
        self,
        scores: Tensor,
        total: int,
        required_mask: Tensor | None,
    ) -> Tensor:
        if required_mask is None:
            return self._elastic_indices(scores, total)
        if scores.shape[0] != 1:
            raise ValueError(
                "Elastic Gaussian retention currently requires batch size 1; "
                f"got B={scores.shape[0]}."
            )
        required = torch.nonzero(required_mask[0], as_tuple=False).flatten()
        cap = total
        if self.cfg.max_gaussians is not None:
            cap = min(cap, int(self.cfg.max_gaussians))
        if required.numel() > cap:
            raise ValueError(
                "Minimum local-cell retention exceeds the elastic upper bound: "
                f"required={required.numel()}, upper_bound={cap}."
            )

        candidate_mask = scores[0] >= float(self.cfg.threshold)
        candidate_mask = candidate_mask | required_mask[0]
        if not bool(candidate_mask.any()):
            candidate_mask[torch.argmax(scores[0])] = True
        candidates = torch.nonzero(candidate_mask, as_tuple=False).flatten()
        if candidates.numel() > cap:
            remaining = cap - int(required.numel())
            if remaining > 0:
                optional_mask = candidate_mask & ~required_mask[0]
                optional = torch.nonzero(optional_mask, as_tuple=False).flatten()
                optional_scores = scores[0].index_select(0, optional)
                extra = optional.index_select(
                    0,
                    torch.topk(optional_scores, remaining, sorted=True).indices,
                )
                candidates = torch.cat((required, extra), dim=0)
            else:
                candidates = required
        return candidates.unsqueeze(0)

    def forward(
        self,
        gaussians: Gaussians,
        scores: Tensor,
        required_mask: Tensor | None = None,
    ) -> GaussianRetentionResult:
        if not self.cfg.enabled:
            return self.identity(gaussians)
        batch, total = _validate_gaussians(gaussians)
        if scores.ndim == 3 and scores.shape[-1] == 1:
            scores = scores.squeeze(-1)
        if scores.shape != (batch, total):
            raise ValueError(
                "Gaussian retention scores must have shape "
                f"[B, G]={(batch, total)}, got {tuple(scores.shape)}."
            )
        if not scores.is_floating_point():
            raise ValueError("Gaussian retention scores must be floating point.")
        if scores.device != gaussians.opacities.device:
            raise ValueError("Gaussian retention scores must be on the Gaussian device.")
        if not torch.isfinite(scores).all():
            raise ValueError("Gaussian retention scores contain NaN or Inf values.")

        detached_scores = scores.detach()
        required_mask = self._validate_required_mask(required_mask, detached_scores)
        if self.cfg.mode == "elastic":
            indices = self._elastic_indices_with_required(
                detached_scores,
                total,
                required_mask,
            )
        else:
            kept = self._fixed_keep_count(total)
            if kept >= total:
                return self.identity(gaussians)
            indices = self._fixed_indices(
                detached_scores,
                kept,
                required_mask,
            )

        if indices.shape[1] >= total:
            return self.identity(gaussians)
        # Selection should not otherwise reorder the surviving Gaussians.
        indices = indices.sort(dim=1).values

        selected = gather_gaussians(gaussians, indices)
        output_count = int(indices.shape[1])
        return GaussianRetentionResult(
            gaussians=selected,
            indices=indices,
            input_count=total,
            output_count=output_count,
            applied=output_count < total,
        )
