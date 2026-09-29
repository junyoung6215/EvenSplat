from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor

PREPROCESSING_ID = "noposplat-rescale-center-crop-rgb-v1"


@dataclass
class FoundationPriorCfg:
    enabled: bool = False
    source: Literal["cache", "mock"] = "cache"
    root: str | None = None
    teacher_id: str = ""
    teacher_settings_id: str = ""
    preprocessing_id: str = PREPROCESSING_ID
    allow_mock: bool = False


@dataclass(frozen=True)
class PriorIdentity:
    scene: str
    context: tuple[int, int]
    image_shape: tuple[int, int]
    teacher_id: str
    teacher_settings_id: str
    preprocessing_id: str = PREPROCESSING_ID
    schema_version: int = 1

    def __post_init__(self):
        if not self.scene or not self.teacher_id or not self.teacher_settings_id:
            raise ValueError("Prior identity needs scene, teacher_id and teacher_settings_id.")
        if len(self.context) != 2 or len(set(self.context)) != 2 or any(
            type(index) is not int or index < 0 for index in self.context
        ):
            raise ValueError("Prior identity requires two distinct ordered context indices.")
        if len(self.image_shape) != 2 or any(type(n) is not int or n <= 0 for n in self.image_shape):
            raise ValueError("Prior identity requires positive image dimensions.")
        if self.schema_version != 1 or self.preprocessing_id != PREPROCESSING_ID:
            raise ValueError("Unsupported prior schema or preprocessing contract.")

    def metadata(self) -> dict:
        return json.loads(json.dumps(asdict(self)))

    @property
    def key(self) -> str:
        encoded = json.dumps(self.metadata(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


def image_fingerprint(images: Tensor) -> str:
    """Hash the exact uint8 RGB source grid after NoPo crop, before normalization."""
    if images.ndim != 4 or images.shape[0:2] != (2, 3):
        raise ValueError("Prior images must have shape [2,3,H,W].")
    if not bool(torch.isfinite(images).all()) or bool(((images < 0) | (images > 1)).any()):
        raise ValueError("Prior images must be finite RGB values in [0,1].")
    pixels = images.detach().cpu().mul(255).round().to(torch.uint8).contiguous().numpy()
    return hashlib.sha256(pixels.tobytes()).hexdigest()


def validate_prior_maps(target: Tensor, valid: Tensor, weight: Tensor, shape: tuple[int, ...]) -> None:
    if any(tuple(value.shape) != shape for value in (target, valid, weight)):
        raise ValueError(f"Prior target/valid/weight must all have shape {shape}.")
    if valid.dtype != torch.bool:
        raise ValueError("Prior valid mask must be boolean.")
    for name, value in (("target", target), ("weight", weight)):
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Prior {name} must contain finite floating point values.")
        if bool(((value < 0) | (value > 1)).any()):
            raise ValueError(f"Prior {name} must lie in [0,1]; raw teacher confidence is not a probability.")


def mock_prior(shape: tuple[int, int, int]) -> dict[str, Tensor]:
    """Diagnostic labels only: left=conflict, right=support, first row=unknown."""
    views, height, width = shape
    target = torch.zeros(views, height, width)
    target[..., width // 2:] = 1
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[:, 0] = False
    return {"teacher_target": target, "teacher_valid": valid,
            "teacher_weight": torch.ones_like(target), "is_mock": torch.tensor(True)}


def write_prior_cache(root: str | Path, identity: PriorIdentity, images: Tensor,
                      maps: dict[str, Tensor], *, kind: Literal["teacher", "mock"]) -> Path:
    """Write the NoPo-facing contract, not a foundation model's native output."""
    if kind not in ("teacher", "mock"):
        raise ValueError("Prior kind must be teacher or mock.")
    shape = (2, *identity.image_shape)
    if tuple(images.shape) != (2, 3, *identity.image_shape):
        raise ValueError("Images do not match the prior identity.")
    target, valid, weight = (maps[name].detach().cpu() for name in
                             ("teacher_target", "teacher_valid", "teacher_weight"))
    validate_prior_maps(target, valid, weight, shape)
    if "is_mock" in maps and bool(maps["is_mock"].any()) and kind != "mock":
        raise ValueError("Mock supervision cannot be written as a real teacher cache.")
    metadata = {**identity.metadata(), "image_sha256": image_fingerprint(images), "kind": kind}
    path = Path(root) / f"{identity.key}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves previous experimental artifacts.
    with path.open("xb") as handle:
        np.savez_compressed(handle, metadata=np.array(json.dumps(metadata, sort_keys=True)),
                            teacher_target=target.float().numpy().astype(np.float16),
                            teacher_valid=valid.numpy(),
                            teacher_weight=weight.float().numpy().astype(np.float16))
    return path


def read_prior_cache(root: str | Path, identity: PriorIdentity, images: Tensor,
                     *, allow_mock: bool = False) -> dict[str, Tensor]:
    if tuple(images.shape) != (2, 3, *identity.image_shape):
        raise ValueError("Images do not match the prior identity.")
    path = Path(root) / f"{identity.key}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing prior for {identity.scene}/{identity.context}: {path}")
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"].item()))
        for key, expected in identity.metadata().items():
            if metadata.get(key) != expected:
                raise ValueError(f"Prior metadata mismatch for {key}: {path}")
        if metadata.get("image_sha256") != image_fingerprint(images):
            raise ValueError(f"Prior RGB fingerprint mismatch (crop, flip or dataset changed): {path}")
        kind = metadata.get("kind")
        if kind not in ("teacher", "mock") or (kind == "mock" and not allow_mock):
            raise ValueError("Mock cache is permitted only in explicitly enabled diagnostic runs.")
        maps = {name: torch.from_numpy(payload[name].copy()) for name in
                ("teacher_target", "teacher_valid", "teacher_weight")}
    validate_prior_maps(maps["teacher_target"], maps["teacher_valid"], maps["teacher_weight"],
                        (2, *identity.image_shape))
    maps["teacher_target"] = maps["teacher_target"].float()
    maps["teacher_weight"] = maps["teacher_weight"].float()
    maps["is_mock"] = torch.tensor(kind == "mock")
    return maps


class PriorProvider:
    """CPU dataset hook. Disabled/evaluation calls are literal pass-throughs."""

    def __init__(self, cfg: FoundationPriorCfg, stage: str, *, augment: bool, fixed_pairs: bool):
        self.cfg, self.active = cfg, cfg.enabled and stage == "train"
        if not self.active:
            return
        if augment:
            raise ValueError("Prior MVP requires dataset augmentation OFF in every comparison arm.")
        if cfg.preprocessing_id != PREPROCESSING_ID:
            raise ValueError("Unsupported prior preprocessing_id.")
        if cfg.source == "mock":
            if not cfg.allow_mock:
                raise ValueError("source=mock requires allow_mock=true; diagnostic use only.")
        elif cfg.source == "cache":
            if not fixed_pairs or not cfg.root or not cfg.teacher_id or not cfg.teacher_settings_id:
                raise ValueError("Cached prior requires fixed_pairs, root, teacher_id and teacher_settings_id.")
        else:
            raise ValueError(f"Unknown prior source: {cfg.source}")

    def attach(self, example: dict) -> dict:
        if not self.active:
            return example
        context = example["context"]
        images = context["image"]
        shape = tuple(images.shape[-2:])
        image_fingerprint(images)  # validate the source grid for both modes
        if self.cfg.source == "mock":
            maps = mock_prior((2, *shape))
        else:
            identity = PriorIdentity(example["scene"], tuple(context["index"].tolist()), shape,
                                     self.cfg.teacher_id, self.cfg.teacher_settings_id,
                                     self.cfg.preprocessing_id)
            maps = read_prior_cache(self.cfg.root, identity, images, allow_mock=self.cfg.allow_mock)
        return {**example, "_foundation": maps}
