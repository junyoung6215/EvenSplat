from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, get_cfg


@dataclass(frozen=True)
class RunPaths:
    root: Path
    train: Path
    eval: Path
    test_outputs: Path
    manifest: Path


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_run_paths(
    config: dict[str, Any],
    *,
    run_id: str | None,
    root_dir: Path | None,
) -> RunPaths:
    resolved_run_id = run_id or str(get_cfg(config, "run_id", "")) or make_run_id()
    name = str(get_cfg(config, "name"))
    storage_root = Path(str(get_cfg(config, "storage_root")))
    root = root_dir or storage_root / name / resolved_run_id
    config["run_id"] = resolved_run_id
    config["resolved_root_dir"] = str(root)
    return RunPaths(
        root=root,
        train=root / "train",
        eval=root / "eval",
        test_outputs=root / "test_outputs",
        manifest=root,
    )


def nearest_existing_parent(path: Path) -> Path:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return probe


def check_free_space(path: Path, min_free_gb: int) -> None:
    usage = shutil.disk_usage(nearest_existing_parent(path))
    available_gb = usage.free // (1024**3)
    if available_gb < min_free_gb:
        raise SystemExit(
            f"Not enough free space for {path}: "
            f"{available_gb}GB available, {min_free_gb}GB required."
        )


def guard_output_path(path: Path, *, allow_overlay_outputs: bool) -> None:
    if allow_overlay_outputs:
        return
    if not path.is_absolute():
        raise SystemExit(
            f"Refusing relative output path {path}. "
            "Use /home/data/... or pass --allow-overlay-outputs intentionally."
        )
    resolved = path.resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise SystemExit(
            f"Refusing repo-local output path {path}. "
            "Use /home/data/... or pass --allow-overlay-outputs intentionally."
        )


def validate_output_paths(paths: RunPaths, *, allow_overlay_outputs: bool) -> None:
    for path in (paths.root, paths.train, paths.eval, paths.test_outputs):
        guard_output_path(path, allow_overlay_outputs=allow_overlay_outputs)
