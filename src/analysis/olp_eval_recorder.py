from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .olp_stats import gaussian_tail_stats, merge_stats, save_stats_json


class OlpEvalRecorder:
    """Small recorder for OLP latent stats and Gaussian tail stats."""

    def __init__(
        self,
        tau_alpha: float = 0.95,
        tau_scale: float | None = None,
        tau_volume: float | None = None,
        adaptive_tau_scale_quantile: float | None = 0.95,
    ) -> None:
        self.per_scene: list[dict[str, Any]] = []
        self.tau_alpha = tau_alpha
        self.tau_scale = tau_scale
        self.tau_volume = tau_volume
        self.adaptive_tau_scale_quantile = adaptive_tau_scale_quantile

    def collect_preconditioner_stats(self, encoder: nn.Module) -> dict[str, float]:
        stats_by_key: dict[str, list[float]] = {}
        for module in encoder.modules():
            latest_stats = getattr(module, "latest_stats", None)
            if not latest_stats:
                continue
            for key, value in latest_stats.items():
                stats_by_key.setdefault(f"latent_{key}", []).append(float(value))
        return {
            key: sum(values) / len(values)
            for key, values in stats_by_key.items()
            if values
        }

    def record_test_batch(
        self,
        encoder: nn.Module,
        visualization_dump: dict[str, torch.Tensor],
        scene: str,
        batch_idx: int,
        metrics: dict[str, torch.Tensor] | None = None,
        overlap: float | None = None,
        overlap_tag: str | None = None,
    ) -> None:
        stats: dict[str, Any] = {
            "scene": scene,
            "batch_idx": batch_idx,
        }
        if overlap is not None:
            stats["overlap"] = float(overlap)
        if overlap_tag is not None:
            stats["overlap_tag"] = overlap_tag
        stats.update(self.collect_preconditioner_stats(encoder))
        stats.update(
            gaussian_tail_stats(
                visualization_dump["opacities"],
                visualization_dump["scales"],
                tau_alpha=self.tau_alpha,
                tau_scale=self.tau_scale,
                tau_volume=self.tau_volume,
                adaptive_tau_scale_quantile=self.adaptive_tau_scale_quantile,
            )
        )
        if metrics is not None:
            stats.update(
                {
                    key: float(value.detach().cpu())
                    for key, value in metrics.items()
                }
            )
        self.per_scene.append(stats)

    def write(self, output_dir: str | Path) -> None:
        if not self.per_scene:
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        serializable_stats = [
            {
                key: value
                for key, value in item.items()
                if isinstance(value, (str, int, float))
            }
            for item in self.per_scene
        ]
        (output_dir / "olp_stats_per_scene.json").write_text(
            json.dumps(serializable_stats, indent=2, sort_keys=True) + "\n"
        )
        numeric_stats = [
            {
                key: value
                for key, value in item.items()
                if isinstance(value, (int, float)) and key != "batch_idx"
            }
            for item in serializable_stats
        ]
        save_stats_json(
            merge_stats(numeric_stats),
            output_dir / "olp_stats_summary.json",
        )
        grouped_stats: dict[str, dict[str, float]] = {}
        for overlap_tag in sorted(
            {
                item["overlap_tag"]
                for item in serializable_stats
                if isinstance(item.get("overlap_tag"), str)
            }
        ):
            grouped_stats[overlap_tag] = merge_stats(
                {
                    key: value
                    for key, value in item.items()
                    if isinstance(value, (int, float)) and key != "batch_idx"
                }
                for item in serializable_stats
                if item.get("overlap_tag") == overlap_tag
            )
        if grouped_stats:
            save_stats_json(
                grouped_stats,
                output_dir / "olp_stats_summary_by_overlap.json",
            )
        with (output_dir / "olp_stats_per_scene.csv").open("w", newline="") as f:
            leading = ["scene", "batch_idx"]
            fieldnames = leading + sorted(
                {
                    key
                    for item in serializable_stats
                    for key in item
                    if key not in leading
                }
            )
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(serializable_stats)
