from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class OrthogonalPreconditionerCfg:
    enabled: bool = False
    mode: str = "hadamard"
    dim: int = 256
    mix: float = 0.0
    target_mix: float = 1.0
    warmup_steps: int = 5000
    seed: int = 0
    collect_stats: bool = False


def cfg_get(cfg: Any, name: str, default: Any) -> Any:
    """Read config fields from dataclasses, OmegaConf nodes, or plain dicts."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)
