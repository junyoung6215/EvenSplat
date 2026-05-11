#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a smaller RE10K evaluation index from an existing index."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("assets/evaluation_index_re10k.json"),
        help="Source evaluation index JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination evaluation index JSON.",
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Number of valid scenes to keep.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Number of valid scenes to skip before selecting.",
    )
    parser.add_argument(
        "--targets-per-scene",
        type=int,
        default=0,
        help="Limit target frames per scene; 0 keeps all targets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open("r") as f:
        index = json.load(f)

    valid_items = [(scene, entry) for scene, entry in index.items() if entry is not None]
    selected = valid_items[args.start : args.start + args.count]
    if len(selected) < args.count:
        raise SystemExit(
            f"Only found {len(selected)} valid scenes after start={args.start}; "
            f"requested {args.count}."
        )

    subset = {}
    for scene, entry in selected:
        entry = dict(entry)
        if args.targets_per_scene > 0:
            entry["target"] = entry["target"][: args.targets_per_scene]
        subset[scene] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(subset, f, indent=2)
        f.write("\n")

    target_count = sum(len(entry["target"]) for entry in subset.values())
    print(
        f"Wrote {len(subset)} scenes / {target_count} target images to {args.output}"
    )


if __name__ == "__main__":
    main()
