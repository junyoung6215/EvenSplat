from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from ...foundation.pairs import load_training_pairs
from .view_sampler import ViewSampler


@dataclass
class ViewSamplerFixedPairsCfg:
    name: Literal["fixed_pairs"]
    index_path: Path
    evaluation_index_path: Path
    num_context_views: int = 2


class ViewSamplerFixedPairs(ViewSampler[ViewSamplerFixedPairsCfg]):
    """One ordered training pair per scene; validation uses a separate index."""

    def __init__(self, cfg, stage, is_overfitting, cameras_are_circular, step_tracker):
        super().__init__(cfg, stage, is_overfitting, cameras_are_circular, step_tracker)
        if stage != "train" or is_overfitting or cfg.num_context_views != 2:
            raise ValueError("fixed_pairs requires train split, two views, and no overfit mode.")
        self.pairs = load_training_pairs(cfg.index_path)

    def sample(self, scene, extrinsics, intrinsics, device=torch.device("cpu")):
        pair = self.pairs[scene]
        if max(*pair.context, *pair.target) >= len(extrinsics):
            raise ValueError(f"Fixed pair frame out of range: {scene}")
        return (torch.tensor(pair.context, dtype=torch.int64, device=device),
                torch.tensor(pair.target, dtype=torch.int64, device=device),
                torch.tensor([pair.overlap], dtype=torch.float32, device=device))

    @property
    def num_context_views(self):
        return 2

    @property
    def num_target_views(self):
        return len(next(iter(self.pairs.values())).target)
