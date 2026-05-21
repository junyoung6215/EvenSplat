from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .commands import (
    Command,
    build_compare_commands,
    build_eval_command,
    build_train_command,
    command_line,
    python_bin,
)
from .config import REPO_ROOT, apply_overrides, get_cfg, load_preset, validate_config
from .manifest import update_status, write_manifest
from .paths import (
    RunPaths,
    check_free_space,
    resolve_run_paths,
    validate_output_paths,
)


@dataclass(frozen=True)
class RunOptions:
    preset: str
    stage: str | None = None
    run_id: str | None = None
    root_dir: Path | None = None
    checkpoint: Path | None = None
    baseline_ckpt: Path | None = None
    olp_ckpt: Path | None = None
    overrides: list[str] = field(default_factory=list)
    dry_run: bool = False
    allow_overlay_outputs: bool = False


def load_config(options: RunOptions) -> dict[str, Any]:
    config = load_preset(options.preset)
    apply_overrides(config, options.overrides)
    if options.stage is not None:
        config["stage"] = options.stage
    if options.run_id is not None:
        config["run_id"] = options.run_id
    validate_config(config)
    return config


def env_for_run(config: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    if get_cfg(config, "wandb_mode") is not None:
        env["WANDB_MODE"] = str(get_cfg(config, "wandb_mode"))
    if get_cfg(config, "cuda_visible_devices") is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(get_cfg(config, "cuda_visible_devices"))

    python_path = Path(python_bin(config)).resolve()
    python_parent = python_path.parent
    python_prefix = python_parent.parent
    env["PATH"] = f"{python_parent}:{env.get('PATH', '')}"
    env["CONDA_PREFIX"] = str(python_prefix)
    env["CONDA_DEFAULT_ENV"] = python_prefix.name
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    cache_root = Path(
        str(
            get_cfg(
                config,
                "cache_root",
                Path(str(get_cfg(config, "resolved_root_dir"))) / "cache",
            )
        )
    )
    run_id = str(get_cfg(config, "run_id", "run"))
    tmp_root = Path(str(get_cfg(config, "tmp_root", Path("/tmp") / f"olp_{run_id}")))
    runtime_dirs = {
        "TMPDIR": tmp_root,
        "TMP": tmp_root,
        "TEMP": tmp_root,
        "XDG_CACHE_HOME": cache_root / "xdg",
        "TORCH_HOME": cache_root / "torch",
        "HF_HOME": cache_root / "huggingface",
        "WANDB_DIR": cache_root / "wandb",
        "WANDB_CACHE_DIR": cache_root / "wandb_cache",
        "CUDA_CACHE_PATH": cache_root / "cuda",
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "PYTHONPYCACHEPREFIX": cache_root / "pycache",
    }
    for key, path in runtime_dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)

    torch_lib_dir = get_cfg(config, "torch_lib_dir")
    if torch_lib_dir is None:
        try:
            import torch

            torch_lib_dir = str(Path(torch.__file__).resolve().parent / "lib")
        except Exception:
            torch_lib_dir = None

    if torch_lib_dir and Path(str(torch_lib_dir)).is_dir():
        current = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{torch_lib_dir}:{current}" if current else str(torch_lib_dir)
        )
    return env


def check_checkpoint(path: Path, *, label: str, dry_run: bool) -> None:
    if dry_run:
        return
    if not path.exists():
        raise SystemExit(f"Missing {label} checkpoint: {path}")


def build_preset_eval_command(
    config: dict[str, Any],
    paths: RunPaths,
    *,
    checkpoint: Path,
) -> Command:
    eval_cfg = get_cfg(config, "eval", {})
    return build_eval_command(
        config,
        run_dir=paths.eval,
        checkpoint=checkpoint,
        run_name=str(get_cfg(eval_cfg, "name", f"{get_cfg(config, 'name')}_eval")),
        index_path=Path(
            str(get_cfg(eval_cfg, "index_path", "assets/evaluation_index_re10k.json"))
        ),
        output_path=paths.test_outputs,
        save_image=bool(get_cfg(eval_cfg, "save_image", False)),
        save_olp_stats=bool(get_cfg(eval_cfg, "save_olp_stats", True)),
        olp_enabled=bool(get_cfg(eval_cfg, "olp_enabled", True)),
        olp_mix=float(get_cfg(eval_cfg, "olp_mix", 1.0)),
        olp_target_mix=float(get_cfg(eval_cfg, "olp_target_mix", 1.0)),
    )


def build_commands(
    config: dict[str, Any],
    paths: RunPaths,
    options: RunOptions,
) -> list[Command]:
    stage = str(get_cfg(config, "stage", "train"))

    if stage in {"train", "smoke"}:
        return [build_train_command(config, paths.train)]

    if stage == "train-eval":
        checkpoint = paths.train / "checkpoints" / "last.ckpt"
        return [
            build_train_command(config, paths.train),
            build_preset_eval_command(config, paths, checkpoint=checkpoint),
        ]

    if stage == "eval":
        eval_cfg = get_cfg(config, "eval", {})
        checkpoint = options.checkpoint or get_cfg(eval_cfg, "checkpoint")
        if checkpoint is None:
            raise SystemExit("Eval stage requires --checkpoint or eval.checkpoint.")
        checkpoint = Path(str(checkpoint))
        check_checkpoint(checkpoint, label="eval", dry_run=options.dry_run)
        return [build_preset_eval_command(config, paths, checkpoint=checkpoint)]

    if stage == "compare":
        baseline = options.baseline_ckpt or get_cfg(config, "checkpoints.baseline")
        olp = options.olp_ckpt or get_cfg(config, "checkpoints.olp")
        if baseline is not None:
            check_checkpoint(Path(str(baseline)), label="baseline", dry_run=options.dry_run)
        if olp is not None:
            check_checkpoint(Path(str(olp)), label="OLP", dry_run=options.dry_run)
        return build_compare_commands(
            config,
            paths,
            baseline_ckpt=Path(str(baseline)) if baseline is not None else None,
            olp_ckpt=Path(str(olp)) if olp is not None else None,
        )

    raise SystemExit(f"Unknown stage: {stage}")


def run_command(command: Command, *, config: dict[str, Any]) -> None:
    print(command_line(command), flush=True)
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env_for_run(config),
        check=True,
    )


def run_env_check(config: dict[str, Any], paths: RunPaths) -> None:
    if not bool(get_cfg(config, "check_env", True)):
        return
    command = [
        python_bin(config),
        "scripts/check_cuda118_env.py",
        "--output-dir",
        str(paths.manifest),
    ]
    run_command(command, config=config)


def run(options: RunOptions) -> None:
    config = load_config(options)
    paths = resolve_run_paths(
        config,
        run_id=options.run_id,
        root_dir=options.root_dir,
    )
    validate_output_paths(
        paths,
        allow_overlay_outputs=options.allow_overlay_outputs,
    )
    min_free_gb = int(get_cfg(config, "min_free_gb", 30))
    check_free_space(paths.root, min_free_gb)
    check_free_space(paths.test_outputs, min_free_gb)

    commands = build_commands(config, paths, options)
    if options.dry_run:
        print(f"run_dir: {paths.root}")
        for command in commands:
            print(command_line(command))
        return

    if paths.root.exists():
        raise SystemExit(f"Run directory already exists: {paths.root}")

    write_manifest(config, paths, commands)
    try:
        run_env_check(config, paths)
        for command in commands:
            run_command(command, config=config)
    except Exception:
        update_status(config, paths, commands, status="failed")
        raise
    update_status(config, paths, commands, status="completed")
