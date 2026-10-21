"""CPU contract/gradient tests. No downloads, pytest, CUDA or training required.

The two integration harnesses execute actual class/method ASTs while bypassing
unavailable Lightning/CUDA imports. They do NOT substitute for a full server run.
Run: python -B -m unittest discover -s tests -p test_prior_retention.py -v
"""
import ast
import json
import math
import sys
import tempfile
import unittest
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Generic, TypeVar

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.foundation.pairs import load_training_pairs
from src.foundation.prior import (FoundationPriorCfg, PriorIdentity, PriorProvider,
                                  image_fingerprint, mock_prior, read_prior_cache,
                                  write_prior_cache)
from src.foundation.supervision import prior_reliability_loss
from src.model.image_demand import image_detail_demand
from src.model.local_cell_retention import LocalCellAllocator, LocalCellRetentionCfg, GaussianGridLayout
from scripts.check_prior_cache import check_or_make

def actual_definitions(relative_path, names, namespace):
    """Execute unmodified source definitions; bypass imports, not implementation."""
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and node.name in names]
    assert len(nodes) == len(names)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), relative_path, "exec"), namespace)
    return namespace


def prior_loss_classes():
    namespace = dict(nn=nn, ABC=ABC, Generic=Generic, T_cfg=TypeVar("T_cfg"),
                     T_wrapper=TypeVar("T_wrapper"), abstractmethod=abstractmethod, fields=fields)
    actual_definitions("src/loss/loss.py", ["Loss"], namespace)
    namespace.update(dataclass=dataclass, math=math, prior_reliability_loss=prior_reliability_loss)
    actual_definitions("src/loss/loss_reliability_prior.py",
                       ["LossReliabilityPriorCfg", "LossReliabilityPriorCfgWrapper",
                        "validate_prior_loss_encoder", "LossReliabilityPrior"], namespace)
    return namespace


class PriorContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.identity = PriorIdentity("scene", (8, 2), (4, 6), "teacher-revision", "settings-v1")
        self.images = torch.arange(144).reshape(2, 3, 4, 6).float() / 255
        self.maps = mock_prior((2, 4, 6))

    def manifest(self, **changes):
        payload = {"schema_version": 1, "split": "train", "pairs": [
            {"scene": "scene", "context": [8, 2], "target": [4, 5]}]}
        payload.update(changes)
        path = self.root / "pairs.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_manifest_preserves_order(self):
        pair = load_training_pairs(self.manifest())["scene"]
        self.assertEqual(pair.context, (8, 2))
        self.assertEqual(pair.target, (4, 5))

    def test_manifest_rejects_eval_split(self):
        with self.assertRaises(ValueError):
            load_training_pairs(self.manifest(split="test"))

    def test_manifest_rejects_ambiguous_or_invalid_pairs(self):
        good = {"scene": "scene", "context": [8, 2], "target": [4, 5]}
        rows = [[good, good], [{**good, "target": [8]}], [{**good, "context": [2, 2]}],
                [{**good, "context": [True, 2]}], [{**good, "overlap": float("nan")}],
                [good, {**good, "scene": "second", "target": [3]}]]
        for pairs in rows:
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                load_training_pairs(self.manifest(pairs=pairs))

    def test_cache_key_includes_order_model_settings_and_grid(self):
        changes = [dict(context=(2, 8)), dict(teacher_id="another"),
                   dict(teacher_settings_id="another"), dict(image_shape=(6, 4)), dict(scene="other")]
        for change in changes:
            self.assertNotEqual(self.identity.key, replace(self.identity, **change).key)

    def test_cache_round_trip_fp16_and_no_overwrite(self):
        self.maps["teacher_target"].fill_(0.371)
        write_prior_cache(self.root, self.identity, self.images, self.maps, kind="mock")
        loaded = read_prior_cache(self.root, self.identity, self.images, allow_mock=True)
        torch.testing.assert_close(loaded["teacher_target"], self.maps["teacher_target"], atol=0.0003, rtol=0)
        self.assertEqual(loaded["teacher_valid"].dtype, torch.bool)
        with self.assertRaises(FileExistsError):
            write_prior_cache(self.root, self.identity, self.images, self.maps, kind="mock")

    def test_mock_requires_both_truthful_write_and_read_opt_in(self):
        with self.assertRaises(ValueError):
            write_prior_cache(self.root, self.identity, self.images, self.maps, kind="teacher")
        write_prior_cache(self.root, self.identity, self.images, self.maps, kind="mock")
        with self.assertRaises(ValueError):
            read_prior_cache(self.root, self.identity, self.images)

    def test_real_cache_is_accepted_without_mock_flag(self):
        self.maps["is_mock"] = torch.tensor(False)
        write_prior_cache(self.root, self.identity, self.images, self.maps, kind="teacher")
        self.assertFalse(read_prior_cache(self.root, self.identity, self.images)["is_mock"])

    def test_missing_cache_fails_instead_of_silent_fallback(self):
        with self.assertRaises(FileNotFoundError):
            read_prior_cache(self.root, self.identity, self.images)

    def test_crop_flip_and_shape_mismatch_fail(self):
        write_prior_cache(self.root, self.identity, self.images, self.maps, kind="mock")
        for images in (self.images.flip(-1), self.images.flip(0), self.images[..., :3, :]):
            with self.subTest(shape=images.shape), self.assertRaises(ValueError):
                read_prior_cache(self.root, self.identity, images, allow_mock=True)

    def test_raw_confidence_nonfinite_mask_and_wrong_shape_rejected(self):
        changes = [{"teacher_weight": torch.full((2, 4, 6), 2.)},
                   {"teacher_target": torch.full((2, 4, 6), float("nan"))},
                   {"teacher_valid": torch.ones(2, 4, 6)},
                   {"teacher_target": torch.ones(2, 6, 4)}]
        for change in changes:
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                write_prior_cache(self.root, self.identity, self.images, {**self.maps, **change}, kind="mock")

    def test_disabled_and_eval_provider_are_exact_identity(self):
        for cfg, stage in ((FoundationPriorCfg(), "train"), (FoundationPriorCfg(enabled=True), "test"),
                           (FoundationPriorCfg(enabled=True), "val")):
            provider = PriorProvider(cfg, stage, augment=True, fixed_pairs=False)
            example = {"arbitrary": "no images or cache"}
            self.assertIs(provider.attach(example), example)

    def test_provider_configuration_fails_fast(self):
        cases = [(FoundationPriorCfg(enabled=True), False, False),
                 (FoundationPriorCfg(enabled=True, source="mock"), False, True),
                 (FoundationPriorCfg(enabled=True, source="mock", allow_mock=True), True, True)]
        for cfg, augment, fixed in cases:
            with self.assertRaises(ValueError):
                PriorProvider(cfg, "train", augment=augment, fixed_pairs=fixed)

    def test_provider_actual_cache_and_collation(self):
        write_prior_cache(self.root, self.identity, self.images, self.maps, kind="mock")
        cfg = FoundationPriorCfg(enabled=True, root=str(self.root), teacher_id=self.identity.teacher_id,
                                 teacher_settings_id=self.identity.teacher_settings_id, allow_mock=True)
        provider = PriorProvider(cfg, "train", augment=False, fixed_pairs=True)
        example = {"scene": "scene", "context": {"image": self.images, "index": torch.tensor([8, 2])}}
        attached = provider.attach(example)
        self.assertNotIn("_foundation", example)
        batch = torch.utils.data.default_collate([attached, attached])
        self.assertEqual(batch["_foundation"]["teacher_target"].shape, (2, 2, 4, 6))
        self.assertEqual(batch["_foundation"]["is_mock"].shape, (2,))

    def test_export_cache_check_pipeline(self):
        inputs, cache = self.root / "inputs", self.root / "cache"
        inputs.mkdir()
        metadata = {**self.identity.metadata(), "image_sha256": image_fingerprint(self.images)}
        np.savez_compressed(inputs / f"{self.identity.key}.npz", images=self.images.mul(255).round().byte().numpy(),
                            metadata=np.array(json.dumps(metadata)))
        self.assertEqual(check_or_make(inputs, cache, make_mock=True), 1)
        self.assertEqual(check_or_make(inputs, cache, allow_mock=True), 1)
        with self.assertRaises(ValueError):
            check_or_make(inputs, cache)


class PriorGradients(unittest.TestCase):
    def tensors(self, surfaces=1):
        reliability = torch.full((1, 2, 2, 2, surfaces, 1), .525, requires_grad=True)
        target = torch.tensor([[[[0., 1.], [0., 1.]], [[0., 1.], [0., 1.]]]], requires_grad=True)
        valid = torch.ones_like(target, dtype=torch.bool)
        weight = torch.ones_like(target, requires_grad=True)
        return reliability, target, valid, weight

    def test_two_sided_gradients_and_teacher_detached(self):
        reliability, target, valid, weight = self.tensors(2)
        loss, _ = prior_reliability_loss(reliability, target, valid, weight, .05)
        loss.backward()
        self.assertTrue((reliability.grad[..., 0, :, :] > 0).all())
        self.assertTrue((reliability.grad[..., 1, :, :] < 0).all())
        self.assertIsNone(target.grad)
        self.assertIsNone(weight.grad)

    def test_unknown_pixels_have_no_supervision(self):
        reliability, target, valid, weight = self.tensors()
        valid[..., 0] = False
        loss, _ = prior_reliability_loss(reliability, target, valid, weight, .05)
        loss.backward()
        self.assertEqual(reliability.grad[..., 0, :, :].count_nonzero(), 0)

    def test_empty_valid_or_zero_weights_have_finite_zero_gradient(self):
        for all_invalid in (True, False):
            r, t, v, w = self.tensors()
            if all_invalid:
                v.zero_()
            else:
                w = torch.zeros_like(w)
            loss, _ = prior_reliability_loss(r, t, v, w, .05)
            loss.backward()
            self.assertEqual(loss.item(), 0)
            self.assertEqual(r.grad.count_nonzero(), 0)

    def test_surface_count_does_not_scale_loss(self):
        one = prior_reliability_loss(*self.tensors(1), .05)[0]
        many = prior_reliability_loss(*self.tensors(4), .05)[0]
        torch.testing.assert_close(one, many)

    def test_range_and_shape_mismatches_fail(self):
        r, t, v, w = self.tensors()
        for r_min in (-.1, 1., float("nan"), .8):
            with self.assertRaises(ValueError):
                prior_reliability_loss(r, t, v, w, r_min)
        with self.assertRaises(ValueError):
            prior_reliability_loss(r.flatten(1), t, v, w, .05)

    def test_fp16_backward_finite(self):
        r, t, v, w = self.tensors()
        r = r.detach().half().requires_grad_()
        loss, _ = prior_reliability_loss(r, t, v, w, .05)
        loss.backward()
        self.assertTrue(torch.isfinite(r.grad).all())

    def test_actual_loss_warmup_and_mock_guard(self):
        ns = prior_loss_classes()
        cfg = ns["LossReliabilityPriorCfg"](enabled=True, warmup_start=2, warmup_end=4)
        loss = ns["LossReliabilityPrior"](ns["LossReliabilityPriorCfgWrapper"](cfg))
        prediction = SimpleNamespace(color=torch.zeros(1))
        self.assertEqual(loss(prediction, {}, None, 2), 0)
        self.assertFalse(loss.is_active(2))
        self.assertEqual(loss._warmup_factor(3), .5)
        r, t, v, w = self.tensors()
        batch = {"_reliability_prior": {"reliability": r, "r_min": .05}, "_foundation": {
            "teacher_target": t, "teacher_valid": v, "teacher_weight": w, "is_mock": torch.tensor([True])}}
        with self.assertRaises(ValueError):
            loss(prediction, batch, None, 4)
        cfg.allow_mock = True
        loss(prediction, batch, None, 4).backward()
        self.assertGreater(r.grad.abs().sum(), 0)
        self.assertEqual(loss.last_info["mock_fraction"], 1)

    def test_actual_validator_preserves_legacy_and_rejects_bad_combinations(self):
        validate = prior_loss_classes()["validate_prior_loss_encoder"]
        encoder = SimpleNamespace(gaussian_reliability_cfg=SimpleNamespace(enabled=True))
        legacy = SimpleNamespace(name="reliability", cfg=SimpleNamespace(enabled=True, weight=.1))
        prior = SimpleNamespace(name="reliability_prior", cfg=SimpleNamespace(enabled=True, weight=.1))
        validate(encoder, [legacy], SimpleNamespace(enabled=True))
        validate(encoder, [prior], SimpleNamespace(enabled=False))
        with self.assertRaises(ValueError):
            validate(encoder, [legacy, prior], SimpleNamespace(enabled=False))
        with self.assertRaises(ValueError):
            validate(encoder, [prior], SimpleNamespace(enabled=True))
        with self.assertRaises(ValueError):
            validate(SimpleNamespace(), [prior], SimpleNamespace(enabled=False))


class DemandAndWiring(unittest.TestCase):
    def test_image_demand_changes_allocation_without_changing_global_budget(self):
        images = torch.zeros(1, 2, 3, 4, 4)
        images[:, 1, :, :, 2:] = 1
        layout = GaussianGridLayout(2, 4, 4, 1, 1)
        scores = torch.linspace(.5, .6, 32).flip(0).reshape(1, 32)
        allocator = LocalCellAllocator(LocalCellRetentionCfg(enabled=True, cell_size=2))
        uniform = allocator(scores, image_detail_demand(images, 0), layout)
        torch.testing.assert_close(uniform.scores, scores)
        detailed = allocator(scores, image_detail_demand(images, 2), layout)
        selected = detailed.scores.topk(8, dim=1).indices
        self.assertEqual(selected.numel(), 8)
        self.assertTrue((selected >= 16).all())

    def test_constant_and_strength_zero_are_uniform(self):
        images = torch.ones(2, 2, 3, 4, 4) * .5
        torch.testing.assert_close(image_detail_demand(images), torch.ones(2, 2, 4, 4))
        torch.testing.assert_close(image_detail_demand(torch.rand_like(images), 0), torch.ones(2, 2, 4, 4))

    def test_detail_uses_joint_view_scale_and_no_grad(self):
        images = torch.zeros(1, 2, 3, 4, 4)
        images[:, 1, :, :, 2:] = 1
        images.requires_grad_()
        demand = image_detail_demand(images, 2)
        self.assertFalse(demand.requires_grad)
        self.assertEqual(demand[:, 0].max(), 1)
        self.assertEqual(demand[:, 1].max(), 3)

    def test_demand_bad_inputs_fail(self):
        for strength in (-1., float("inf")):
            with self.assertRaises(ValueError):
                image_detail_demand(torch.zeros(1, 2, 3, 4, 4), strength)
        with self.assertRaises(ValueError):
            image_detail_demand(torch.ones(1, 2, 3, 4, 4) * 2)

    def test_actual_wrapper_attaches_dense_gates_without_render_or_detach(self):
        tree = ast.parse((ROOT / "src/model/model_wrapper.py").read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ModelWrapper")
        names = {"_prior_loss_active", "_attach_reliability_prior"}
        methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
        harness = ast.ClassDef(name="WrapperHarness", bases=[], keywords=[], body=methods, decorator_list=[])
        module = ast.Module(body=[harness], type_ignores=[])
        ns = {}
        exec(compile(ast.fix_missing_locations(module), "actual_wrapper_methods", "exec"), ns)
        wrapper = ns["WrapperHarness"]()
        wrapper.global_step = 0
        wrapper.losses = []
        batch = {}
        wrapper._attach_reliability_prior(batch, None)
        self.assertEqual(batch, {})
        wrapper.losses = [SimpleNamespace(name="reliability_prior", is_active=lambda step: True)]
        wrapper.anchor_selection_cfg = SimpleNamespace(enabled=False)
        wrapper.encoder = SimpleNamespace(gaussian_reliability_cfg=SimpleNamespace(r_min=.05))
        gates = torch.ones(1, 2, 4, 4, 1, 1, requires_grad=True)
        wrapper._attach_reliability_prior(batch, {"reliability": gates})
        self.assertIs(batch["_reliability_prior"]["reliability"], gates)
        step = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "training_step")
        source = ast.unparse(step)
        self.assertLess(source.index("self._attach_reliability_prior("), source.index("self._apply_gaussian_retention("))


if __name__ == "__main__":
    unittest.main()
