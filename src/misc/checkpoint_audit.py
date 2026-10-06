from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .weight_modify import checkpoint_filter_fn


def _sample_keys(keys: list[str], limit: int = 30) -> list[str]:
    return keys[:limit]


def load_encoder_pretrained_weights(
    encoder: nn.Module,
    weight_path: str | Path,
    audit_path: str | Path | None = None,
    strict: bool = False,
    require_backbone: bool = False,
) -> dict[str, Any]:
    """Load encoder pretrained weights and write a small audit summary."""
    weight_path = Path(weight_path)
    ckpt_weights = torch.load(weight_path, map_location="cpu")

    if "model" in ckpt_weights:
        checkpoint_format = "model"
        ckpt_weights = checkpoint_filter_fn(ckpt_weights["model"], encoder)
    elif "state_dict" in ckpt_weights:
        checkpoint_format = "state_dict"
        ckpt_weights = {
            k[8:]: v
            for k, v in ckpt_weights["state_dict"].items()
            if k.startswith("encoder.")
        }
    else:
        raise ValueError(f"Invalid checkpoint format: {weight_path}")

    if not ckpt_weights:
        raise RuntimeError(f"No encoder weights found in checkpoint: {weight_path}")

    missing_keys, unexpected_keys = encoder.load_state_dict(
        ckpt_weights,
        strict=strict,
    )
    encoder_keys = set(encoder.state_dict())
    loaded_keys = set(ckpt_weights)
    matched_keys = sorted(encoder_keys & loaded_keys)
    missing_backbone = sorted(
        name for name, _ in encoder.named_parameters()
        if name.startswith("backbone.") and name in missing_keys
    )

    summary = {
        "weight_path": str(weight_path),
        "checkpoint_format": checkpoint_format,
        "strict": strict,
        "require_backbone": require_backbone,
        "missing_backbone_parameters": missing_backbone,
        "loaded_key_count": len(ckpt_weights),
        "matched_key_count": len(matched_keys),
        "missing_key_count": len(missing_keys),
        "unexpected_key_count": len(unexpected_keys),
        "matched_key_sample": _sample_keys(matched_keys),
        "missing_key_sample": _sample_keys(list(missing_keys)),
        "unexpected_key_sample": _sample_keys(list(unexpected_keys)),
    }

    print("[checkpoint-audit] Encoder pretrained load summary:")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if len(matched_keys) == 0:
        raise RuntimeError(f"Checkpoint did not match any encoder keys: {weight_path}")

    if audit_path is not None:
        audit_path = Path(audit_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if require_backbone and missing_backbone:
        raise RuntimeError(
            f"heads training requires all frozen backbone weights; missing {missing_backbone[:5]}. "
            "Use a complete NoPoSplat checkpoint. See pretrained_load_audit.json."
        )
    return summary
