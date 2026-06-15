from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .olp_stats import (
    gaussian_tail_stats,
    merge_stats,
    opacity_reliability_effect_stats,
    reliability_stats,
    save_stats_json,
)


class OlpEvalRecorder:
    """Small recorder for OLP latent stats and Gaussian tail stats."""

    def __init__(
        self,
        tau_alpha: float = 0.95,
        tau_scale: float | None = None,
        tau_volume: float | None = None,
        adaptive_tau_scale_quantile: float | None = 0.95,
        save_raw_dump: bool = False,
        raw_dump_max_locations: int = 256,
        save_full_raw_maps: bool = False,
    ) -> None:
        self.per_scene: list[dict[str, Any]] = []
        self.raw_dumps: list[dict[str, Any]] = []
        self.tau_alpha = tau_alpha
        self.tau_scale = tau_scale
        self.tau_volume = tau_volume
        self.adaptive_tau_scale_quantile = adaptive_tau_scale_quantile
        self.save_raw_dump = bool(save_raw_dump)
        self.raw_dump_max_locations = int(raw_dump_max_locations)
        self.save_full_raw_maps = bool(save_full_raw_maps)

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

    def collect_preconditioner_raw_dumps(self, encoder: nn.Module) -> list[dict[str, Any]]:
        dumps: list[dict[str, Any]] = []
        for module_idx, module in enumerate(encoder.modules()):
            latest_dump = getattr(module, "latest_outlier_dump", None)
            if not latest_dump:
                continue
            dump = dict(latest_dump)
            entries = dump.get("entries", [])
            if isinstance(entries, list):
                dump["entries"] = entries[: self.raw_dump_max_locations]
            dump["module_idx"] = int(module_idx)
            dump["module_class"] = module.__class__.__name__
            dumps.append(dump)
        return dumps

    def collect_preconditioner_full_maps(self, encoder: nn.Module) -> list[dict[str, Any]]:
        maps: list[dict[str, Any]] = []
        for module_idx, module in enumerate(encoder.modules()):
            latest_maps = getattr(module, "latest_outlier_maps", None)
            if not latest_maps:
                continue
            item = dict(latest_maps)
            item["module_idx"] = int(module_idx)
            item["module_class"] = module.__class__.__name__
            maps.append(item)
        return maps

    def collect_gaussian_full_maps(
        self, visualization_dump: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        opacities = visualization_dump["opacities"].detach().float().cpu()
        scales = visualization_dump["scales"].detach().float().cpu()
        scale_max = scales.reshape(-1, 3).amax(dim=-1).reshape(-1)
        out: dict[str, Any] = {
            "opacities_shape": list(opacities.shape),
            "scales_shape": list(scales.shape),
            "opacity": opacities.flatten(),
            "scale_max": scale_max,
        }
        depth = visualization_dump.get("depth")
        if isinstance(depth, torch.Tensor):
            candidate = depth.detach().float().cpu().flatten()
            if candidate.numel() == out["opacity"].numel():
                out["depth"] = candidate
        return out

    def collect_gaussian_raw_dump(
        self, visualization_dump: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        opacities = visualization_dump["opacities"].detach().float().flatten()
        scales = visualization_dump["scales"].detach().float().reshape(-1, 3)
        scale_max = scales.amax(dim=-1)
        proxy = opacities * scale_max
        n = min(self.raw_dump_max_locations, int(proxy.numel()))
        if n > 0:
            values, indices = torch.topk(proxy, n)
        else:
            values = proxy.new_empty(0)
            indices = torch.empty(0, dtype=torch.long, device=proxy.device)

        depth_flat = None
        depth = visualization_dump.get("depth")
        if isinstance(depth, torch.Tensor):
            candidate = depth.detach().float().flatten()
            if candidate.numel() == opacities.numel():
                depth_flat = candidate

        entries: list[dict[str, Any]] = []
        for rank, (value, index) in enumerate(zip(values.cpu(), indices.cpu())):
            idx = int(index.item())
            scale = scales[idx].cpu().tolist()
            item: dict[str, Any] = {
                "rank": rank,
                "flat_index": idx,
                "opacity": float(opacities[idx].cpu().item()),
                "scale": [float(v) for v in scale],
                "scale_max": float(scale_max[idx].cpu().item()),
                "opacity_x_scale_max": float(value.item()),
            }
            if depth_flat is not None:
                item["depth"] = float(depth_flat[idx].cpu().item())
            entries.append(item)

        return {
            "opacity_scale_top_locations": entries,
            "opacities_shape": list(visualization_dump["opacities"].shape),
            "scales_shape": list(visualization_dump["scales"].shape),
            "max_locations": n,
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
        if "opacities_before_reliability" in visualization_dump:
            stats.update(
                opacity_reliability_effect_stats(
                    visualization_dump["opacities_before_reliability"],
                    visualization_dump["opacities"],
                )
            )
        if "reliability" in visualization_dump:
            stats.update(reliability_stats(visualization_dump["reliability"]))
        if metrics is not None:
            stats.update(
                {
                    key: float(value.detach().cpu())
                    for key, value in metrics.items()
                }
            )
        self.per_scene.append(stats)

        if self.save_raw_dump:
            self.raw_dumps.append(
                {
                    "scene": scene,
                    "batch_idx": int(batch_idx),
                    "overlap": overlap,
                    "overlap_tag": overlap_tag,
                    "preconditioner_top_locations": self.collect_preconditioner_raw_dumps(
                        encoder
                    ),
                    "gaussian_top_locations": self.collect_gaussian_raw_dump(
                        visualization_dump
                    ),
                }
            )

    def write_full_raw_maps(
        self,
        output_dir: str | Path,
        scene: str,
        batch_idx: int,
        encoder: nn.Module,
        visualization_dump: dict[str, torch.Tensor],
    ) -> None:
        if not self.save_full_raw_maps:
            return
        output_dir = Path(output_dir)
        dump_dir = output_dir / "olp_raw_full_maps"
        dump_dir.mkdir(parents=True, exist_ok=True)
        safe_scene = str(scene).replace("/", "_")
        path = dump_dir / f"{safe_scene}_batch{int(batch_idx):06d}.pt"
        torch.save(
            {
                "scene": scene,
                "batch_idx": int(batch_idx),
                "preconditioner_maps": self.collect_preconditioner_full_maps(encoder),
                "gaussian_maps": self.collect_gaussian_full_maps(visualization_dump),
            },
            path,
        )

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

        if self.raw_dumps:
            with (output_dir / "olp_raw_diagnostics.jsonl").open("w") as f:
                for item in self.raw_dumps:
                    f.write(json.dumps(item, sort_keys=True) + "\n")
