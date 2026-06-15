from __future__ import annotations

import sys
import warnings
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.losses.canonical_geo_loss import (  # noqa: E402
    compute_reference_centered_geo_loss,
)
from src.loss.loss_geo import LossGeo, LossGeoCfg, LossGeoCfgWrapper  # noqa: E402
from src.utils.anchor_selection import (  # noqa: E402
    AnchorSelectionCfg,
    anchor_selection_applies_to_stage,
    compute_sharpness_scores,
    renormalize_cameras_to_selected_anchor,
    reorder_context_fields,
    require_safe_anchor_reorder,
    select_anchor_indices,
)


def test_sharpness_and_middle() -> None:
    b, v, c, h, w = 2, 5, 3, 16, 16
    images = torch.rand(b, v, c, h, w)
    scores = compute_sharpness_scores(images)
    assert scores.shape == (b, v)
    assert torch.isfinite(scores).all()

    batch = {"context": {"image": images}}
    cfg = AnchorSelectionCfg(enabled=True, method="middle")
    anchors = select_anchor_indices(batch, cfg)
    assert anchors.tolist() == [v // 2, v // 2]

    sharp_images = torch.zeros(b, v, c, h, w)
    checker = (torch.arange(h).view(h, 1) + torch.arange(w).view(1, w)) % 2
    sharp_images[:, 3] = checker.float().view(1, 1, h, w)
    sharp_batch = {"context": {"image": sharp_images}}
    cfg = AnchorSelectionCfg(enabled=True, method="sharpness")
    anchors = select_anchor_indices(sharp_batch, cfg)
    assert anchors.tolist() == [3, 3]


def test_reorder_context_fields() -> None:
    b, v, t = 2, 4, 2
    context_index = torch.tensor([[0, 1, 2, 3], [10, 11, 12, 13]])
    target_image = torch.randn(b, t, 3, 4, 4)
    batch = {
        "context": {
            "image": torch.arange(b * v).view(b, v, 1, 1, 1).float(),
            "index": context_index.clone(),
            "camera": torch.arange(b * v * 3).view(b, v, 3),
            "nested": {
                "confidence": torch.arange(b * v * 2).view(b, v, 2).float(),
            },
        },
        "target": {
            "image": target_image.clone(),
            "index": torch.tensor([[100, 101], [200, 201]]),
        },
    }
    reordered = reorder_context_fields(batch, torch.tensor([2, 1]))

    assert reordered["context"]["index"].tolist() == [
        [2, 0, 1, 3],
        [11, 10, 12, 13],
    ]
    assert torch.equal(reordered["target"]["image"], target_image)
    assert torch.equal(reordered["target"]["index"], batch["target"]["index"])
    assert reordered["context"]["camera"].shape == (b, v, 3)
    assert reordered["context"]["nested"]["confidence"].shape == (b, v, 2)


def _make_c2w_translation(x: float, y: float, z: float) -> torch.Tensor:
    matrix = torch.eye(4)
    matrix[:3, 3] = torch.tensor([x, y, z])
    return matrix


def test_anchor_camera_renormalization() -> None:
    b, v, t = 1, 3, 2
    context_extrinsics = torch.stack(
        [
            _make_c2w_translation(0.0, 0.0, 0.0),
            _make_c2w_translation(1.0, 0.0, 0.0),
            _make_c2w_translation(1.0, 2.0, 0.0),
        ]
    ).view(b, v, 4, 4)
    target_extrinsics = torch.stack(
        [
            _make_c2w_translation(2.0, 0.0, 0.0),
            _make_c2w_translation(1.0, 3.0, 0.0),
        ]
    ).view(b, t, 4, 4)
    target_images = torch.randn(b, t, 3, 4, 4)
    batch = {
        "context": {
            "image": torch.zeros(b, v, 3, 4, 4),
            "index": torch.tensor([[10, 11, 12]]),
            "extrinsics": context_extrinsics,
            "intrinsics": torch.eye(3).view(1, 1, 3, 3).repeat(b, v, 1, 1),
        },
        "target": {
            "image": target_images.clone(),
            "index": torch.tensor([[20, 21]]),
            "extrinsics": target_extrinsics,
        },
    }

    output = renormalize_cameras_to_selected_anchor(batch, torch.tensor([1]))

    assert output["context"]["index"].tolist() == [[11, 10, 12]]
    assert torch.equal(output["target"]["image"], target_images)
    assert output["target"]["index"].tolist() == [[20, 21]]
    assert torch.allclose(output["context"]["extrinsics"][0, 0], torch.eye(4))
    assert torch.allclose(
        output["context"]["extrinsics"][0, :, :3, 3],
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        ),
    )
    assert torch.allclose(
        output["target"]["extrinsics"][0, :, :3, 3],
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
            ]
        ),
    )


def test_anchor_safety_policy_and_apply_to() -> None:
    safe_cfg = AnchorSelectionCfg(
        enabled=True,
        method="middle",
        renormalize_cameras=True,
    )
    assert require_safe_anchor_reorder(safe_cfg) is True
    assert anchor_selection_applies_to_stage(safe_cfg, "test") is True
    assert anchor_selection_applies_to_stage(safe_cfg, "train") is False

    train_cfg = AnchorSelectionCfg(
        enabled=True,
        method="middle",
        apply_to="train",
    )
    assert anchor_selection_applies_to_stage(train_cfg, "train") is True
    assert anchor_selection_applies_to_stage(train_cfg, "test") is False

    unsafe_blocked = AnchorSelectionCfg(
        enabled=True,
        method="middle",
        renormalize_cameras=False,
        allow_unsafe_reorder=False,
    )
    try:
        require_safe_anchor_reorder(unsafe_blocked)
    except RuntimeError as exc:
        assert "without camera renormalization is unsafe" in str(exc)
    else:
        raise AssertionError("Unsafe anchor reorder was not blocked.")


def _make_point_maps(
    b: int,
    v: int,
    h: int,
    w: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    z = torch.full((h, w), 2.0)
    base = torch.stack([xs.float() * z, ys.float() * z, z], dim=-1)
    points = base.view(1, 1, h, w, 3).repeat(b, v, 1, 1, 1)
    points[:, 1:, :, :, 0] = points[:, 1:, :, :, 0] + 0.01
    points.requires_grad_(True)

    intrinsics = torch.eye(3).view(1, 1, 3, 3).repeat(b, v, 1, 1)
    intrinsics[:, :, 0, 0] = 1.0 / w
    intrinsics[:, :, 1, 1] = 1.0 / h
    intrinsics[:, :, 0, 2] = 0.0
    intrinsics[:, :, 1, 2] = 0.0

    depth = torch.full((b, v, h, w), 2.0)
    confidence = torch.ones(b, v, h, w)
    return points, intrinsics, depth, confidence


def test_geo_loss() -> None:
    b, v, h, w = 2, 3, 6, 7
    points, intrinsics, depth, confidence = _make_point_maps(b, v, h, w)
    loss, info = compute_reference_centered_geo_loss(
        points,
        intrinsics,
        anchor_indices=torch.zeros(b, dtype=torch.long),
        depth=depth,
        confidence=confidence,
        min_valid_pixels=1,
        stop_gradient_anchor=True,
        return_debug_tensors=True,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert "valid_ratio" in info
    assert float(info["valid_ratio"]) > 0.0
    assert info["used_depth_visibility"] == 1

    loss.backward()
    assert points.grad is not None
    assert torch.allclose(points.grad[:, 0], torch.zeros_like(points.grad[:, 0]))
    assert points.grad[:, 1:].abs().sum() > 0

    points2, intrinsics2, _, _ = _make_point_maps(b, v, h, w)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loss2, info2 = compute_reference_centered_geo_loss(
            points2,
            intrinsics2,
            anchor_indices=torch.zeros(b, dtype=torch.long),
            depth=None,
            confidence=None,
            min_valid_pixels=1,
            stop_gradient_anchor=True,
            warn_missing_depth=True,
        )
    assert torch.isfinite(loss2)
    assert "valid_ratio" in info2
    assert info2["used_depth_visibility"] == 0
    assert any("no cached depth" in str(item.message) for item in caught)


def test_geo_loss_nan_guard() -> None:
    b, v, h, w = 1, 3, 4, 5
    points, intrinsics, _, _ = _make_point_maps(b, v, h, w)
    points = points.detach().clone()
    points[0, 1, 0, 0, 0] = float("nan")
    points[0, 2, 1, 1, 1] = float("inf")

    loss, info = compute_reference_centered_geo_loss(
        points,
        intrinsics,
        anchor_indices=torch.zeros(b, dtype=torch.long),
        depth=None,
        confidence=None,
        min_valid_pixels=1,
        warn_missing_depth=False,
    )
    assert torch.isfinite(loss)
    assert int(info["valid_pixel_count"]) > 0

    all_bad = torch.full_like(points, float("nan"))
    loss_bad, info_bad = compute_reference_centered_geo_loss(
        all_bad,
        intrinsics,
        anchor_indices=torch.zeros(b, dtype=torch.long),
        depth=None,
        confidence=None,
        min_valid_pixels=1,
        warn_missing_depth=False,
    )
    assert torch.isfinite(loss_bad)
    assert float(loss_bad) == 0.0
    assert int(info_bad["valid_pixel_count"]) == 0


def test_loss_geo_contract_guard() -> None:
    b, v, h, w = 1, 2, 3, 4
    points, intrinsics, _, _ = _make_point_maps(b, v, h, w)
    cfg = LossGeoCfg(
        enabled=True,
        weight=0.01,
        robust="charbonnier",
        charbonnier_eps=1e-3,
        visibility_depth_threshold=0.05,
        use_confidence_weight=True,
        stop_gradient_anchor=True,
        min_valid_pixels=1,
        debug=False,
    )
    loss_fn = LossGeo(LossGeoCfgWrapper(cfg))
    batch = {
        "_anchorgeo": {
            "canonical_points": points,
            "anchor_indices": torch.zeros(b, dtype=torch.long),
        },
        "target": {"image": torch.zeros(b, 1, 3, h, w)},
        "context": {
            "image": torch.zeros(b, v, 3, h, w),
            "intrinsics": intrinsics,
        },
    }
    try:
        loss_fn.forward(None, batch, None, 0)
    except RuntimeError as exc:
        assert "explicit canonical point-map contract" in str(exc)
    else:
        raise AssertionError("LossGeo accepted ambiguous canonical frame.")

    batch["_anchorgeo"]["canonical_frame"] = "context0"
    batch["_anchorgeo"]["canonical_points_view_aligned"] = True
    loss = loss_fn.forward(None, batch, None, 0)
    assert torch.isfinite(loss)

    batch["_anchorgeo"]["unsafe_reorder"] = True
    try:
        loss_fn.forward(None, batch, None, 0)
    except RuntimeError as exc:
        assert "unsafe anchor reordering" in str(exc)
    else:
        raise AssertionError("LossGeo accepted unsafe reordered cameras.")


def test_geo_loss_missing_intermediate_error() -> None:
    cfg = LossGeoCfg(
        enabled=True,
        weight=0.01,
        robust="charbonnier",
        charbonnier_eps=1e-3,
        visibility_depth_threshold=0.05,
        use_confidence_weight=True,
        stop_gradient_anchor=True,
        min_valid_pixels=1,
        debug=False,
    )
    loss_fn = LossGeo(LossGeoCfgWrapper(cfg))
    batch = {
        "target": {"image": torch.zeros(1, 1, 3, 2, 2)},
        "context": {"intrinsics": torch.eye(3).view(1, 1, 3, 3)},
    }
    try:
        loss_fn.forward(None, batch, None, 0)
    except RuntimeError as exc:
        message = str(exc)
        assert "canonical per-view 3D point maps" in message
        assert "visualization_dump['means']" in message
    else:
        raise AssertionError("LossGeo did not raise for missing canonical point maps.")


def main() -> None:
    torch.manual_seed(123)
    test_sharpness_and_middle()
    test_reorder_context_fields()
    test_anchor_camera_renormalization()
    test_anchor_safety_policy_and_apply_to()
    test_geo_loss()
    test_geo_loss_nan_guard()
    test_loss_geo_contract_guard()
    test_geo_loss_missing_intermediate_error()
    print("anchorgeo unit smoke tests OK")


if __name__ == "__main__":
    main()
