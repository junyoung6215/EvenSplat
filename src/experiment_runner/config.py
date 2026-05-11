from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
PRESET_DIR = REPO_ROOT / "config" / "experiment_runs"


def load_preset(name_or_path: str) -> dict[str, Any]:
    path = Path(name_or_path)
    if path.suffix not in {".yaml", ".yml"}:
        path = PRESET_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        raise SystemExit(f"Missing experiment preset: {path}")
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(config, dict):
        raise SystemExit(f"Preset must contain a mapping: {path}")
    config["preset_path"] = str(path)
    return config


def parse_override_value(raw: str) -> Any:
    try:
        parsed = OmegaConf.to_container(OmegaConf.create(f"value: {raw}"), resolve=True)
    except Exception:
        return raw
    return parsed["value"]


def set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    node = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise SystemExit(f"Cannot override through non-mapping key: {dotted_key}")
        node = child
    node[parts[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> None:
    for override in overrides:
        if "=" not in override:
            raise SystemExit(f"Override must be key=value: {override}")
        key, raw_value = override.split("=", 1)
        set_nested(config, key, parse_override_value(raw_value))


def require_keys(config: dict[str, Any], keys: list[str]) -> None:
    for key in keys:
        if get_cfg(config, key) is None:
            raise SystemExit(f"Missing required preset key: {key}")


def validate_config(config: dict[str, Any]) -> None:
    require_keys(config, ["name", "stage", "storage_root", "data_root"])
    stage = str(get_cfg(config, "stage"))
    allowed_stages = {"train", "eval", "train-eval", "compare", "smoke"}
    if stage not in allowed_stages:
        raise SystemExit(f"Unknown preset stage: {stage}")

    if stage in {"train", "train-eval", "smoke"}:
        require_keys(
            config,
            [
                "pretrained_weights",
                "train.max_steps",
                "train.batch_size",
                "train.lr",
            ],
        )


def get_cfg(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def hydra_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
