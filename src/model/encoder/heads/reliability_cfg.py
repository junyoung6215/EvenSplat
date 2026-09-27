from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class GaussianReliabilityCfg:
    enabled: bool = False
    r_min: float = 0.05
    r_init: float = 0.95
    detach_gate: bool = False


def opacity_reliability_gate(
    reliability: Tensor,
    detach_gate: bool,
) -> Tensor:
    """Return the reliability tensor used by the opacity rendering path.

    ``detach_gate`` isolates the reliability head from reconstruction gradients
    flowing through opacity. The original reliability tensor remains attached
    so an auxiliary reliability loss can still train the head.
    """

    return reliability.detach() if detach_gate else reliability

