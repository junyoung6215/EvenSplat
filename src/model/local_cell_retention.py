from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class LocalCellRetentionCfg:
    """Optional cell-level demand allocation applied before global selection."""

    enabled: bool = False
    cell_size: int = 16
    demand_key: str = "cell_demand"
    demand_floor: float = 0.0
    min_per_cell: int = 0
    demand_source: str = "external"
    detail_strength: float = 1.0


@dataclass(frozen=True)
class GaussianGridLayout:
    """Dense source-grid layout used by NoPoSplat before flattening Gaussians."""

    views: int
    height: int
    width: int
    surfaces: int
    samples_per_surface: int

    def __post_init__(self) -> None:
        for name, value in (
            ("views", self.views),
            ("height", self.height),
            ("width", self.width),
            ("surfaces", self.surfaces),
            ("samples_per_surface", self.samples_per_surface),
        ):
            if int(value) <= 0:
                raise ValueError(f"Gaussian grid layout {name} must be positive.")

    @property
    def gaussian_count(self) -> int:
        return (
            int(self.views)
            * int(self.height)
            * int(self.width)
            * int(self.surfaces)
            * int(self.samples_per_surface)
        )

    def cell_shape(self, cell_size: int) -> tuple[int, int]:
        if int(cell_size) <= 0:
            raise ValueError("cell_size must be positive.")
        return (
            (int(self.height) + int(cell_size) - 1) // int(cell_size),
            (int(self.width) + int(cell_size) - 1) // int(cell_size),
        )

    def cell_count(self, cell_size: int) -> int:
        cell_h, cell_w = self.cell_shape(cell_size)
        return int(self.views) * cell_h * cell_w


@dataclass(frozen=True)
class LocalCellAllocation:
    scores: Tensor
    cell_ids: Tensor
    required_mask: Tensor | None
    cell_count: int


def gaussian_cell_ids(
    layout: GaussianGridLayout,
    cell_size: int,
    device: torch.device,
) -> Tensor:
    """Return flattened cell ids in view/pixel/surface/sample order."""

    cell_h, cell_w = layout.cell_shape(cell_size)
    y = torch.arange(layout.height, device=device)
    x = torch.arange(layout.width, device=device)
    cell_y = torch.div(y, int(cell_size), rounding_mode="floor")
    cell_x = torch.div(x, int(cell_size), rounding_mode="floor")
    pixel_cells = (
        cell_y[:, None] * cell_w + cell_x[None, :]
    ).reshape(1, layout.height, layout.width, 1, 1)
    view_offsets = (
        torch.arange(layout.views, device=device)
        .reshape(layout.views, 1, 1, 1, 1)
        .mul(cell_h * cell_w)
    )
    return (
        (pixel_cells + view_offsets)
        .expand(
            layout.views,
            layout.height,
            layout.width,
            layout.surfaces,
            layout.samples_per_surface,
        )
        .reshape(-1)
        .long()
    )


def pool_cell_demand(
    demand: Tensor,
    layout: GaussianGridLayout,
    cell_size: int,
) -> Tensor:
    """Convert full-resolution or already-pooled demand to ``[B, V, Ch, Cw]``."""

    cell_h, cell_w = layout.cell_shape(cell_size)
    valid_spatial_shapes = {
        (int(layout.height), int(layout.width)),
        (cell_h, cell_w),
    }
    if demand.ndim == 5:
        candidates = []
        if demand.shape[2] == 1:
            candidates.append(demand.squeeze(2))
        if demand.shape[-1] == 1:
            candidates.append(demand.squeeze(-1))
        matching = [
            candidate
            for candidate in candidates
            if tuple(int(v) for v in candidate.shape[-2:]) in valid_spatial_shapes
        ]
        if not matching:
            raise ValueError(
                "Five-dimensional local demand must have a singleton channel "
                "before or after its spatial dimensions."
            )
        demand = matching[0]
    if demand.ndim != 4:
        raise ValueError(
            "Local cell demand must have shape [B,V,H,W], [B,V,1,H,W], "
            f"or [B,V,H,W,1], got {tuple(demand.shape)}."
        )
    if int(demand.shape[1]) != int(layout.views):
        raise ValueError(
            f"Local cell demand expected V={layout.views}, got {demand.shape[1]}."
        )

    spatial_shape = tuple(int(v) for v in demand.shape[-2:])
    if spatial_shape == (layout.height, layout.width):
        batch = int(demand.shape[0])
        demand = F.avg_pool2d(
            demand.reshape(batch * layout.views, 1, layout.height, layout.width),
            kernel_size=int(cell_size),
            stride=int(cell_size),
            ceil_mode=True,
            count_include_pad=False,
        ).reshape(batch, layout.views, cell_h, cell_w)
    elif spatial_shape != (cell_h, cell_w):
        raise ValueError(
            "Local cell demand spatial shape must match either the source image "
            f"{(layout.height, layout.width)} or cell grid {(cell_h, cell_w)}, "
            f"got {spatial_shape}."
        )
    return demand


def _one_required_per_cell(scores: Tensor, cell_ids: Tensor, cells: int) -> Tensor:
    """Select the first maximum-scoring Gaussian in every cell."""

    batch, count = scores.shape
    expanded_ids = cell_ids.reshape(1, count).expand(batch, count)
    maxima = scores.new_full((batch, cells), -torch.inf)
    maxima.scatter_reduce_(
        1,
        expanded_ids,
        scores,
        reduce="amax",
        include_self=True,
    )
    is_maximum = scores == maxima.gather(1, expanded_ids)
    source_indices = torch.arange(count, device=scores.device).reshape(1, count)
    source_indices = source_indices.expand(batch, count)
    invalid = torch.full_like(source_indices, count)
    first_maximum = torch.full(
        (batch, cells),
        count,
        device=scores.device,
        dtype=torch.long,
    )
    first_maximum.scatter_reduce_(
        1,
        expanded_ids,
        torch.where(is_maximum, source_indices, invalid),
        reduce="amin",
        include_self=True,
    )
    if bool((first_maximum >= count).any()):
        raise RuntimeError("At least one local cell did not contain a Gaussian.")
    required = torch.zeros_like(scores, dtype=torch.bool)
    required.scatter_(1, first_maximum, True)
    return required


class LocalCellAllocator(nn.Module):
    """Combine Gaussian reliability scores with externally supplied cell demand."""

    def __init__(self, cfg: LocalCellRetentionCfg | None = None) -> None:
        super().__init__()
        self.cfg = cfg or LocalCellRetentionCfg()
        self._validate_cfg()

    def _validate_cfg(self) -> None:
        if self.cfg.demand_source not in ("external", "image_detail"):
            raise ValueError("local_cells.demand_source must be external or image_detail.")
        if not torch.isfinite(torch.tensor(self.cfg.detail_strength)) or self.cfg.detail_strength < 0:
            raise ValueError("local_cells.detail_strength must be finite and nonnegative.")
        if int(self.cfg.cell_size) <= 0:
            raise ValueError("local_cells.cell_size must be positive.")
        if not self.cfg.demand_key:
            raise ValueError("local_cells.demand_key must not be empty.")
        if not torch.isfinite(torch.tensor(float(self.cfg.demand_floor))):
            raise ValueError("local_cells.demand_floor must be finite.")
        if float(self.cfg.demand_floor) < 0.0:
            raise ValueError("local_cells.demand_floor must be non-negative.")
        if int(self.cfg.min_per_cell) not in (0, 1):
            raise ValueError("local_cells.min_per_cell currently supports only 0 or 1.")

    def forward(
        self,
        base_scores: Tensor,
        demand: Tensor,
        layout: GaussianGridLayout,
    ) -> LocalCellAllocation:
        if base_scores.ndim != 2:
            raise ValueError(
                f"Local allocation scores must have shape [B,G], got {base_scores.shape}."
            )
        if int(base_scores.shape[1]) != layout.gaussian_count:
            raise ValueError(
                "Gaussian layout does not match scores: "
                f"layout={layout.gaussian_count}, scores={base_scores.shape[1]}."
            )
        pooled = pool_cell_demand(
            demand.to(device=base_scores.device, dtype=base_scores.dtype),
            layout,
            int(self.cfg.cell_size),
        )
        if int(pooled.shape[0]) != int(base_scores.shape[0]):
            raise ValueError(
                "Local cell demand batch mismatch: "
                f"expected {base_scores.shape[0]}, got {pooled.shape[0]}."
            )
        if not bool(torch.isfinite(pooled).all()):
            raise ValueError("Local cell demand contains NaN or Inf values.")
        if bool((pooled < 0).any()):
            raise ValueError("Local cell demand must be non-negative.")

        pooled = pooled.clamp_min(float(self.cfg.demand_floor))
        if bool((pooled.reshape(pooled.shape[0], -1).amax(dim=1) <= 0).any()):
            raise ValueError(
                "Every sample needs at least one positive local cell demand value."
            )
        cell_ids = gaussian_cell_ids(
            layout,
            int(self.cfg.cell_size),
            base_scores.device,
        )
        per_gaussian_demand = pooled.reshape(pooled.shape[0], -1).index_select(
            1,
            cell_ids,
        )
        combined = base_scores * per_gaussian_demand
        required = None
        if int(self.cfg.min_per_cell) == 1:
            required = _one_required_per_cell(
                base_scores.detach(),
                cell_ids,
                layout.cell_count(int(self.cfg.cell_size)),
            )
        return LocalCellAllocation(
            scores=combined,
            cell_ids=cell_ids,
            required_mask=required,
            cell_count=layout.cell_count(int(self.cfg.cell_size)),
        )


def retention_distribution_statistics(
    selection_mask: Tensor,
    layout: GaussianGridLayout,
    cell_ids: Tensor | None = None,
) -> dict[str, Tensor]:
    """Summarize how evenly a selection is distributed across views and cells."""

    if selection_mask.ndim != 2:
        raise ValueError(
            f"Selection mask must have shape [B,G], got {selection_mask.shape}."
        )
    if int(selection_mask.shape[1]) != layout.gaussian_count:
        raise ValueError("Selection mask does not match the Gaussian grid layout.")
    selected = selection_mask.to(dtype=torch.float32)
    per_view = selected.reshape(selection_mask.shape[0], layout.views, -1).mean(-1)
    stats = {
        "view_kept_fraction_min": per_view.min(),
        "view_kept_fraction_max": per_view.max(),
        "view_kept_fraction_std": per_view.std(unbiased=False),
    }
    if cell_ids is not None:
        if cell_ids.shape != (layout.gaussian_count,):
            raise ValueError("Cell ids do not match the Gaussian grid layout.")
        cells = int(cell_ids.max().item()) + 1
        expanded_ids = cell_ids.reshape(1, -1).expand(selection_mask.shape[0], -1)
        selected_per_cell = selected.new_zeros(selection_mask.shape[0], cells)
        selected_per_cell.scatter_add_(1, expanded_ids, selected)
        total_per_cell = selected.new_zeros(cells)
        total_per_cell.scatter_add_(
            0,
            cell_ids,
            torch.ones_like(cell_ids, dtype=selected.dtype),
        )
        fractions = selected_per_cell / total_per_cell.clamp_min(1.0)
        stats.update(
            {
                "cell_kept_fraction_min": fractions.min(),
                "cell_kept_fraction_max": fractions.max(),
                "cell_kept_fraction_std": fractions.std(unbiased=False),
                "empty_cell_fraction": (fractions == 0).float().mean(),
            }
        )
    return stats
