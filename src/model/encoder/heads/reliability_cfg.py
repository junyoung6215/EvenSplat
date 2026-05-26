from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GaussianReliabilityCfg:
    enabled: bool = False
    r_min: float = 0.05
    r_init: float = 0.95
    detach_gate: bool = False

