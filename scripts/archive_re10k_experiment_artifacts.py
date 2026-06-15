#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


METRIC_FILES = {
    "benchmark.json",
    "peak_memory.json",
    "olp_stats_summary.json",
    "olp_stats_summary_by_overlap.json",
    "olp_stats_per_scene.json",
    "olp_stats_per_scene.csv",
    "gaussian_cap_metrics.csv",
    "gaussian_cap_metrics.json",
    "gaussian_cap_summary.json",
    "olp_raw_diagnostics.jsonl",
}

METADATA_FILES = {
    "command.sh",
    "pretrained_load_audit.json",
    "manifest.txt",
    "run_config.env",
}

HEAVY_SUFFIXES = {
    ".ckpt",
    ".gif",
    ".mp4",
    ".npy",
    ".npz",
    ".pth",
    ".pt",
    ".torch",
}

DEFAULT_SKIP_PARTS = {"smoke", "sanity"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy GitHub-friendly RE10K OLP-H experiment artifacts from outputs/ "
            "into experiment_records/."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/olp_re10k"),
        help="Source output root produced by the RE10K scripts.",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("experiment_records"),
        help="Destination root for curated artifacts.",
    )
    parser.add_argument(
        "--name",
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Archive folder name under --archive-root.",
    )
    parser.add_argument(
        "--extra-path",
        action="append",
        default=[],
        type=Path,
        help="Additional small file or directory to copy, e.g. logs/olp_ft_7k_bs4.txt.",
    )
    parser.add_argument(
        "--include-images",
        action="store_true",
        help="Copy a small sample of representative render images.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=12,
        help="Maximum number of render images to copy when --include-images is set.",
    )
    parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="Also include paths with smoke/sanity in their directory names.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without writing files.",
    )
    return parser.parse_args()


def should_skip(path: Path, include_smoke: bool) -> bool:
    if include_smoke:
        return False
    lowered_parts = [part.lower() for part in path.parts]
    return any(skip in part for part in lowered_parts for skip in DEFAULT_SKIP_PARTS)


def is_heavy(path: Path) -> bool:
    return path.suffix.lower() in HEAVY_SUFFIXES


def copy_file(src: Path, dst: Path, copied: list[str], dry_run: bool) -> None:
    if is_heavy(src):
        return
    copied.append(str(dst))
    if dry_run:
        print(f"copy {src} -> {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(
    src_dir: Path,
    dst_dir: Path,
    copied: list[str],
    dry_run: bool,
    include_smoke: bool,
) -> None:
    for src in sorted(src_dir.rglob("*")):
        if not src.is_file():
            continue
        if should_skip(src, include_smoke) or is_heavy(src):
            continue
        copy_file(src, dst_dir / src.relative_to(src_dir), copied, dry_run)


def collect_named_files(
    output_root: Path,
    archive_dir: Path,
    copied: list[str],
    dry_run: bool,
    include_smoke: bool,
) -> None:
    wanted = METRIC_FILES | METADATA_FILES
    for src in sorted(output_root.rglob("*")):
        if not src.is_file():
            continue
        if src.name not in wanted and src.suffix.lower() != ".log":
            continue
        if should_skip(src.relative_to(output_root), include_smoke):
            continue
        dst = archive_dir / "outputs" / src.relative_to(output_root)
        copy_file(src, dst, copied, dry_run)


def collect_metadata_dirs(
    output_root: Path,
    archive_dir: Path,
    copied: list[str],
    dry_run: bool,
    include_smoke: bool,
) -> None:
    for src_dir in sorted(output_root.rglob("*")):
        if not src_dir.is_dir():
            continue
        if src_dir.name not in {".hydra", "fingerprint"}:
            continue
        rel = src_dir.relative_to(output_root)
        if should_skip(rel, include_smoke):
            continue
        copy_tree(src_dir, archive_dir / "outputs" / rel, copied, dry_run, include_smoke)


def collect_images(
    output_root: Path,
    archive_dir: Path,
    copied: list[str],
    dry_run: bool,
    include_smoke: bool,
    max_images: int,
) -> None:
    if max_images <= 0:
        return

    candidates: list[Path] = []
    for src in sorted(output_root.rglob("*")):
        if not src.is_file() or src.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if "color" not in {part.lower() for part in src.parts}:
            continue
        rel = src.relative_to(output_root)
        if should_skip(rel, include_smoke):
            continue
        candidates.append(src)

    for src in candidates[:max_images]:
        dst = archive_dir / "representative_renders" / src.relative_to(output_root)
        copy_file(src, dst, copied, dry_run)


def extra_destination(archive_dir: Path, extra: Path) -> Path:
    if extra.is_absolute():
        return archive_dir / "extra" / Path(*extra.parts[1:])
    return archive_dir / "extra" / extra


def collect_extra_paths(
    archive_dir: Path,
    extra_paths: list[Path],
    copied: list[str],
    dry_run: bool,
    include_smoke: bool,
) -> None:
    for extra in extra_paths:
        if not extra.exists():
            print(f"missing extra path: {extra}")
            continue
        dst = extra_destination(archive_dir, extra)
        if extra.is_file():
            copy_file(extra, dst, copied, dry_run)
            continue
        copy_tree(extra, dst, copied, dry_run, include_smoke)


def write_manifest(
    archive_dir: Path,
    output_root: Path,
    copied: list[str],
    args: argparse.Namespace,
) -> None:
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "archive_dir": str(archive_dir),
        "include_images": args.include_images,
        "max_images": args.max_images,
        "include_smoke": args.include_smoke,
        "copied_file_count": len(copied),
        "copied_files": copied,
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    archive_dir = args.archive_root / args.name
    copied: list[str] = []

    if not output_root.exists():
        print(f"output root does not exist: {output_root}")
    else:
        collect_named_files(output_root, archive_dir, copied, args.dry_run, args.include_smoke)
        collect_metadata_dirs(output_root, archive_dir, copied, args.dry_run, args.include_smoke)
        if args.include_images:
            collect_images(
                output_root,
                archive_dir,
                copied,
                args.dry_run,
                args.include_smoke,
                args.max_images,
            )

    collect_extra_paths(archive_dir, args.extra_path, copied, args.dry_run, args.include_smoke)
    write_manifest(archive_dir, output_root, copied, args)
    print(f"archived {len(copied)} files under {archive_dir}")


if __name__ == "__main__":
    main()
