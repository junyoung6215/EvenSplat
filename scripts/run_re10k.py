#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.experiment_runner.runner import RunOptions, run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RE10K experiment presets.")
    parser.add_argument("--preset", required=True, help="Preset name or YAML path.")
    parser.add_argument(
        "--stage",
        choices=["train", "eval", "train-eval", "compare", "smoke"],
        help="Override the preset stage.",
    )
    parser.add_argument("--run-id", help="Stable run identifier.")
    parser.add_argument("--root-dir", type=Path, help="Override full run directory.")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint for eval stage.")
    parser.add_argument("--baseline-ckpt", type=Path, help="Baseline checkpoint.")
    parser.add_argument("--olp-ckpt", type=Path, help="OLP checkpoint.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override a preset value, e.g. train.max_steps=9000.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    parser.add_argument(
        "--allow-overlay-outputs",
        action="store_true",
        help="Allow relative or repo-local output paths intentionally.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        RunOptions(
            preset=args.preset,
            stage=args.stage,
            run_id=args.run_id,
            root_dir=args.root_dir,
            checkpoint=args.checkpoint,
            baseline_ckpt=args.baseline_ckpt,
            olp_ckpt=args.olp_ckpt,
            overrides=args.override,
            dry_run=args.dry_run,
            allow_overlay_outputs=args.allow_overlay_outputs,
        )
    )


if __name__ == "__main__":
    main()
