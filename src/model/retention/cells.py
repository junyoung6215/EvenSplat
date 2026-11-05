"""Local cells on the source-pixel grid that NoPoSplat predicts Gaussians on."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class GaussianGridLayout:
    """Dense source-grid layout of the flattened ``[B, G]`` Gaussians.

    Gaussians are flattened in view / pixel / surface / sample order.
    """

    views: int
    height: int
    width: int
    surfaces: int = 1
    samples_per_surface: int = 1

    def __post_init__(self) -> None:
        for name in ("views", "height", "width", "surfaces", "samples_per_surface"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"Gaussian grid layout {name} must be positive.")

    @property
    def gaussian_count(self) -> int:
        return (
            self.views * self.height * self.width * self.surfaces * self.samples_per_surface
        )

    def cell_shape(self, cell_size: int) -> tuple[int, int]:
        if int(cell_size) <= 0:
            raise ValueError("cell_size must be positive.")
        return (
            (self.height + cell_size - 1) // cell_size,
            (self.width + cell_size - 1) // cell_size,
        )

    def cell_count(self, cell_size: int) -> int:
        cell_h, cell_w = self.cell_shape(cell_size)
        return self.views * cell_h * cell_w

    def _per_pixel(self, values: Tensor) -> Tensor:
        """Broadcast a ``[V or 1, H, W]`` map to every Gaussian of each pixel."""

        return (
            values.reshape(-1, self.height, self.width, 1, 1)
            .expand(
                self.views,
                self.height,
                self.width,
                self.surfaces,
                self.samples_per_surface,
            )
            .reshape(-1)
        )


def gaussian_cell_ids(
    layout: GaussianGridLayout,
    cell_size: int,
    device: torch.device,
) -> Tensor:
    """Cell id of every Gaussian; ``cell_size=0`` puts all Gaussians in one cell."""

    if int(cell_size) == 0:
        return torch.zeros(layout.gaussian_count, dtype=torch.long, device=device)
    _, cell_w = layout.cell_shape(cell_size)
    y = torch.arange(layout.height, device=device) // cell_size
    x = torch.arange(layout.width, device=device) // cell_size
    pixel_cells = y[:, None] * cell_w + x[None, :]
    views = torch.arange(layout.views, device=device).reshape(-1, 1, 1)
    per_view = pixel_cells[None] + views * (layout.cell_count(cell_size) // layout.views)
    return layout._per_pixel(per_view).long()


def stratified_scores(scores: Tensor, cell_ids: Tensor, cells: int) -> Tensor:
    """Rank Gaussians inside their cell so a global top-k keeps each cell's share.

    The key is ``-(within-cell rank fraction)`` plus a tie-breaker far smaller
    than one rank step, so a global top-k of the key keeps the best
    ``ceil``/``floor(ratio * cell size)`` Gaussians of every cell.
    """

    batch, count = scores.shape
    positions = torch.arange(count, device=scores.device).expand(batch, count)
    by_score = torch.argsort(scores, dim=1, descending=True, stable=True)
    global_rank = torch.empty_like(by_score).scatter_(1, by_score, positions)
    grouped = by_score.gather(1, torch.argsort(cell_ids[by_score], dim=1, stable=True))
    sizes = torch.bincount(cell_ids, minlength=cells)
    starts = torch.cumsum(sizes, 0) - sizes
    rank_sorted = positions - starts[cell_ids[grouped]]
    rank = torch.empty_like(grouped).scatter_(1, grouped, rank_sorted)
    fraction = (rank.to(scores.dtype) + 0.5) / sizes[cell_ids].to(scores.dtype)
    tie_break = 1.0 - global_rank.to(scores.dtype) / count
    return -fraction + tie_break / (4.0 * float(sizes.max().item()))


def grid_scores(layout: GaussianGridLayout, cell_size: int, device: torch.device) -> Tensor:
    """Score each Gaussian by its pixel's place in a Bayer matrix of its cell.

    Ranked inside each cell these keep a fixed, evenly spread subset of pixel
    positions: one per 2x2 cell is stride-2 subsampling, two is a checkerboard.
    """

    size = int(cell_size)
    if size <= 0 or size & (size - 1):
        raise ValueError("Grid scores need a power-of-two cell_size.")
    bayer = torch.zeros(1, 1, dtype=torch.long, device=device)
    while bayer.shape[0] < size:
        bayer = torch.cat(
            (
                torch.cat((4 * bayer, 4 * bayer + 2), dim=1),
                torch.cat((4 * bayer + 3, 4 * bayer + 1), dim=1),
            ),
            dim=0,
        )
    y = torch.arange(layout.height, device=device) % size
    x = torch.arange(layout.width, device=device) % size
    return layout._per_pixel(-bayer[y[:, None], x[None, :]].float()[None])


def distribution_statistics(
    selection_mask: Tensor,
    layout: GaussianGridLayout,
    cell_ids: Tensor | None = None,
) -> dict[str, Tensor]:
    """Summarize how evenly a selection is spread across views and cells."""

    if selection_mask.shape[1:] != (layout.gaussian_count,):
        raise ValueError("Selection mask does not match the Gaussian grid layout.")
    selected = selection_mask.to(dtype=torch.float32)
    per_view = selected.reshape(selection_mask.shape[0], layout.views, -1).mean(-1)
    stats = {
        "view_kept_fraction_min": per_view.min(),
        "view_kept_fraction_max": per_view.max(),
        "view_kept_fraction_std": per_view.std(unbiased=False),
    }
    if cell_ids is None:
        return stats
    cells = int(cell_ids.max().item()) + 1
    expanded = cell_ids.reshape(1, -1).expand(selection_mask.shape[0], -1)
    kept = selected.new_zeros(selection_mask.shape[0], cells).scatter_add_(1, expanded, selected)
    total = torch.bincount(cell_ids, minlength=cells).to(selected.dtype)
    fractions = kept / total.clamp_min(1.0)
    stats.update(
        {
            "cell_kept_fraction_min": fractions.min(),
            "cell_kept_fraction_max": fractions.max(),
            "cell_kept_fraction_std": fractions.std(unbiased=False),
            "empty_cell_fraction": (fractions == 0).float().mean(),
        }
    )
    return stats
