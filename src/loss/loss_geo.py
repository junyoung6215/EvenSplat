from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..losses.canonical_geo_loss import compute_reference_centered_geo_loss
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossGeoCfg:
    enabled: bool
    weight: float
    robust: Literal["charbonnier", "huber", "l1"]
    charbonnier_eps: float
    visibility_depth_threshold: float
    use_confidence_weight: bool
    stop_gradient_anchor: bool
    min_valid_pixels: int
    debug: bool
    coordinate_frame: str = "context0"
    allow_ambiguous_frame: bool = False


@dataclass
class LossGeoCfgWrapper:
    geo: LossGeoCfg


class LossGeo(Loss[LossGeoCfg, LossGeoCfgWrapper]):
    last_info: dict | None

    def __init__(self, cfg: LossGeoCfgWrapper) -> None:
        super().__init__(cfg)
        self.last_info = None

    def _zero(self, batch: BatchedExample) -> Tensor:
        image = batch["target"]["image"]
        return torch.tensor(0, dtype=torch.float32, device=image.device)

    def _get_canonical_points(self, batch: BatchedExample) -> Tensor:
        anchorgeo = batch.get("_anchorgeo", {})
        canonical_points = anchorgeo.get("canonical_points")
        if canonical_points is None:
            visualization_dump = anchorgeo.get("visualization_dump", {})
            canonical_points = visualization_dump.get("means")

        if canonical_points is None:
            raise RuntimeError(
                "loss.geo.enabled=True requires canonical per-view 3D point maps, "
                "but the current model output does not expose them. Expose "
                "visualization_dump['means'] from "
                "src/model/encoder/encoder_noposplat.py::forward or "
                "src/model/encoder/encoder_noposplat_multi.py::forward by passing "
                "a visualization_dump dictionary through ModelWrapper.training_step."
            )

        if canonical_points.ndim == 6:
            if canonical_points.shape[4] != 1:
                raise RuntimeError(
                    "loss.geo.enabled=True requires one canonical 3D point per "
                    "context pixel. visualization_dump['means'] currently has "
                    f"shape {tuple(canonical_points.shape)} with "
                    f"{canonical_points.shape[4]} Gaussian centers per pixel. "
                    "Expose a dense tensor named canonical_points with shape "
                    "[B, V, H, W, 3] from the encoder before enabling this loss."
                )
            canonical_points = canonical_points.squeeze(4)

        if canonical_points.ndim != 5 or canonical_points.shape[-1] != 3:
            raise RuntimeError(
                "loss.geo.enabled=True requires canonical per-view 3D point maps "
                "with shape [B, V, H, W, 3]. The exposed tensor has shape "
                f"{tuple(canonical_points.shape)}. Expose the per-pixel Gaussian "
                "centers from visualization_dump['means'] or a tensor named "
                "canonical_points with this exact shape."
            )
        return canonical_points

    def _validate_canonical_contract(
        self,
        batch: BatchedExample,
        canonical_points: Tensor,
    ) -> None:
        context = batch.get("context", {})
        context_images = context.get("image")
        if not isinstance(context_images, Tensor) or context_images.ndim != 5:
            raise RuntimeError(
                "loss.geo.enabled=True requires batch['context']['image'] with "
                "shape [B, V, C, H, W] to verify canonical point-map alignment."
            )
        expected_shape = (
            context_images.shape[0],
            context_images.shape[1],
            context_images.shape[-2],
            context_images.shape[-1],
            3,
        )
        if tuple(canonical_points.shape) != expected_shape:
            raise RuntimeError(
                "loss.geo.enabled=True requires canonical_points to align one "
                "3D point to one context pixel. Expected [B, V, H, W, 3] = "
                f"{expected_shape}, got {tuple(canonical_points.shape)}."
            )

        anchorgeo = batch.get("_anchorgeo", {})
        if bool(anchorgeo.get("unsafe_reorder", False)):
            raise RuntimeError(
                "loss.geo cannot be used with unsafe anchor reordering. Cameras "
                "must be renormalized to the selected context[0] frame before "
                "projecting canonical point maps."
            )

        if self.cfg.coordinate_frame != "context0":
            raise RuntimeError(
                "loss.geo currently supports coordinate_frame='context0' only. "
                f"Got {self.cfg.coordinate_frame!r}."
            )

        if self.cfg.allow_ambiguous_frame:
            return

        canonical_frame = anchorgeo.get("canonical_frame")
        view_aligned = bool(anchorgeo.get("canonical_points_view_aligned", False))
        if canonical_frame != self.cfg.coordinate_frame or not view_aligned:
            raise RuntimeError(
                "loss.geo.enabled=True requires an explicit canonical point-map "
                "contract. Expected _anchorgeo['canonical_frame']=='context0' "
                "and _anchorgeo['canonical_points_view_aligned']==True. This "
                "prevents using visualization_dump['means'] when its coordinate "
                "frame or pixel alignment is ambiguous. Set "
                "loss.geo.allow_ambiguous_frame=true only for diagnostic "
                "experiments where you have independently verified the frame."
            )

    def _get_optional_context_tensor(
        self,
        batch: BatchedExample,
        names: tuple[str, ...],
    ) -> Tensor | None:
        context = batch.get("context", {})
        anchorgeo = batch.get("_anchorgeo", {})
        for source in (anchorgeo, context):
            for name in names:
                value = source.get(name)
                if isinstance(value, Tensor):
                    return value
        return None

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        self.last_info = None
        if not self.cfg.enabled:
            return self._zero(batch)
        if self.cfg.weight == 0:
            return self._zero(batch)

        canonical_points = self._get_canonical_points(batch)
        self._validate_canonical_contract(batch, canonical_points)
        intrinsics = batch["context"].get("intrinsics")
        if intrinsics is None:
            raise RuntimeError(
                "loss.geo.enabled=True requires batch['context']['intrinsics'] "
                "with shape [B, V, 3, 3] for anchor projection."
            )

        anchorgeo = batch.get("_anchorgeo", {})
        anchor_indices = anchorgeo.get("anchor_indices")
        if anchor_indices is None:
            anchor_indices = torch.zeros(
                canonical_points.shape[0],
                dtype=torch.long,
                device=canonical_points.device,
            )

        depth = self._get_optional_context_tensor(
            batch,
            (
                "vggt_depth",
                "visibility_depth",
                "anchor_visibility_depth",
            ),
        )
        confidence = self._get_optional_context_tensor(
            batch,
            (
                "vggt_confidence",
                "vggt_conf",
                "confidence",
                "conf",
            ),
        )

        loss, info = compute_reference_centered_geo_loss(
            canonical_points,
            intrinsics,
            anchor_indices=anchor_indices,
            depth=depth,
            confidence=confidence,
            robust=self.cfg.robust,
            charbonnier_eps=self.cfg.charbonnier_eps,
            visibility_depth_threshold=self.cfg.visibility_depth_threshold,
            use_confidence_weight=self.cfg.use_confidence_weight,
            stop_gradient_anchor=self.cfg.stop_gradient_anchor,
            min_valid_pixels=self.cfg.min_valid_pixels,
            return_debug_tensors=bool(self.cfg.debug)
            or bool(anchorgeo.get("debug_geo_enabled", False)),
        )
        self.last_info = info
        return self.cfg.weight * loss
