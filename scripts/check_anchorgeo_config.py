from __future__ import annotations

import sys
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
TMP_DIR = REPO_ROOT / ".tmp"
TMP_DIR.mkdir(exist_ok=True)
os.environ.setdefault("TMP", str(TMP_DIR))
os.environ.setdefault("TEMP", str(TMP_DIR))
os.environ.setdefault("TMPDIR", str(TMP_DIR))

from src.config import load_typed_root_config  # noqa: E402


def compose_cfg(overrides: list[str]):
    config_dir = str(REPO_ROOT / "config")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name="main", overrides=overrides)


def check(name: str, overrides: list[str]) -> None:
    cfg = compose_cfg(overrides)
    typed = load_typed_root_config(cfg)
    print(f"[{name}]")
    print(
        OmegaConf.to_yaml(
            {
                "model.anchor_selection": cfg.model.anchor_selection,
                "loss.geo": cfg.loss.geo,
                "debug.anchor": cfg.debug.anchor,
                "debug.geo": cfg.debug.geo,
            }
        ).strip()
    )
    print(
        "typed:",
        typed.model.anchor_selection.enabled,
        typed.model.anchor_selection.method,
        typed.model.anchor_selection.apply_to,
        typed.model.anchor_selection.renormalize_cameras,
        typed.model.anchor_selection.allow_unsafe_reorder,
        typed.loss[-1].geo.enabled,
        typed.loss[-1].geo.weight,
        typed.loss[-1].geo.coordinate_frame,
        typed.loss[-1].geo.allow_ambiguous_frame,
    )


def main() -> None:
    check("baseline", ["+experiment=re10k"])
    check(
        "anchor_middle",
        [
            "+experiment=re10k",
            "model.anchor_selection.enabled=true",
            "model.anchor_selection.method=middle",
        ],
    )
    check(
        "geo_enabled",
        [
            "+experiment=re10k",
            "loss.geo.enabled=true",
            "loss.geo.weight=0.01",
        ],
    )
    check(
        "anchorgeo_train",
        [
            "+experiment=re10k",
            "mode=train",
            "model.anchor_selection.enabled=true",
            "model.anchor_selection.method=sharpness",
            "model.anchor_selection.apply_to=train",
            "loss.geo.enabled=true",
            "loss.geo.weight=0.01",
        ],
    )
    print("anchorgeo config check OK")


if __name__ == "__main__":
    main()
