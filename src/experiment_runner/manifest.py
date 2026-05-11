from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .commands import Command, command_line
from .config import REPO_ROOT, get_cfg
from .paths import RunPaths


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def write_commands(path: Path, commands: list[Command]) -> None:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    lines.extend(command_line(command) for command in commands)
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o755)


def write_summary(
    config: dict[str, Any],
    paths: RunPaths,
    commands: list[Command],
    *,
    status: str,
) -> None:
    lines = [
        f"status={status}",
        f"name={get_cfg(config, 'name')}",
        f"stage={get_cfg(config, 'stage')}",
        f"run_id={get_cfg(config, 'run_id')}",
        f"root={paths.root}",
        f"train_dir={paths.train}",
        f"eval_dir={paths.eval}",
        f"test_outputs={paths.test_outputs}",
        f"preset={get_cfg(config, 'preset_path')}",
        f"git_commit={git_commit()}",
        "",
        "commands:",
    ]
    lines.extend(f"- {command_line(command)}" for command in commands)
    (paths.manifest / "summary.txt").write_text("\n".join(lines) + "\n")


def write_manifest(
    config: dict[str, Any],
    paths: RunPaths,
    commands: list[Command],
) -> None:
    paths.manifest.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(
        config=OmegaConf.create(config),
        f=paths.manifest / "resolved_config.yaml",
    )
    with (paths.manifest / "run_config.json").open("w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")
    write_commands(paths.manifest / "commands.sh", commands)
    (paths.manifest / "git_commit.txt").write_text(git_commit() + "\n")
    write_summary(config, paths, commands, status="created")


def update_status(
    config: dict[str, Any],
    paths: RunPaths,
    commands: list[Command],
    *,
    status: str,
) -> None:
    write_summary(config, paths, commands, status=status)
