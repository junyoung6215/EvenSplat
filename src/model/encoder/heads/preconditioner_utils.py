from __future__ import annotations

from torch import nn


def set_preconditioner_step(root: nn.Module, global_step: int) -> None:
    """Propagate a training/eval step to modules that expose set_step()."""
    for module in root.modules():
        if module is root:
            continue
        set_step = getattr(module, "set_step", None)
        if callable(set_step):
            set_step(global_step)
