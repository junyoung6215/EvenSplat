from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FixedPair:
    scene: str
    context: tuple[int, int]
    target: tuple[int, ...]
    overlap: float = 0.5


def load_training_pairs(path: str | Path) -> dict[str, FixedPair]:
    """MVP: one ordered context pair per training scene; never an eval index."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1 or payload.get("split") != "train":
        raise ValueError("Fixed pairs require schema_version=1 and split='train'.")
    rows = payload.get("pairs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("The training pair manifest must contain a non-empty pairs list.")
    pairs = {}
    target_count = None
    for row in rows:
        scene = row.get("scene")
        context, target = row.get("context"), row.get("target")
        if not isinstance(scene, str) or not scene or scene in pairs:
            raise ValueError("Each fixed-pair scene must be a non-empty, unique string.")
        for name, indices in (("context", context), ("target", target)):
            if not isinstance(indices, list) or not indices or any(
                type(index) is not int or index < 0 for index in indices
            ):
                raise ValueError(f"{scene}: {name} must contain non-negative integer indices.")
            if len(set(indices)) != len(indices):
                raise ValueError(f"{scene}: duplicate {name} indices.")
        if len(context) != 2:
            raise ValueError("Fixed-pair training requires exactly two context views.")
        if set(context) & set(target):
            raise ValueError(f"{scene}: target views must be distinct from context views.")
        if target_count is not None and len(target) != target_count:
            raise ValueError("All pairs must have the same target count for batched training.")
        target_count = len(target)
        overlap = float(row.get("overlap", 0.5))
        if not math.isfinite(overlap) or not 0 <= overlap <= 1:
            raise ValueError("overlap must be finite and in [0, 1].")
        pairs[scene] = FixedPair(scene, tuple(context), tuple(target), overlap)
    return pairs
