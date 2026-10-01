"""Actual sampler/dataset control-flow harness; decoding/cropping are isolated.

These CPU tests exercise fixed-pair sharding and coverage without requiring
torchvision/einops/Lightning. Real RE10K decode/crop remains a server smoke test.
"""
import json
import sys
import tempfile
import unittest
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from types import SimpleNamespace
from typing import Generic, TypeVar
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.foundation.pairs import load_training_pairs
from src.foundation.prior import FoundationPriorCfg, PriorProvider
from src.foundation.validation import validate_prior_pipeline
from test_prior_retention import actual_definitions


def classes():
    ns = dict(torch=torch, ABC=ABC, Generic=Generic, T=TypeVar("T"), abstractmethod=abstractmethod,
              dataclass=dataclass, field=field, Path=Path, load_training_pairs=load_training_pairs)
    actual_definitions("src/dataset/view_sampler/view_sampler.py", ["ViewSampler"], ns)
    actual_definitions("src/dataset/view_sampler/view_sampler_fixed_pairs.py",
                       ["ViewSamplerFixedPairsCfg", "ViewSamplerFixedPairs"], ns)
    actual_definitions("src/dataset/dataset.py", ["DatasetCfgCommon"], ns)
    ns.update(IterableDataset=torch.utils.data.IterableDataset, json=json, cached_property=cached_property,
              FoundationPriorCfg=FoundationPriorCfg, PriorProvider=PriorProvider,
              tf=SimpleNamespace(ToTensor=lambda: None), get_fov=lambda k: torch.zeros(len(k), 2),
              apply_crop_shim=lambda example, shape: example)
    actual_definitions("src/dataset/dataset_re10k.py", ["DatasetRE10kCfg", "DatasetRE10k"], ns)
    return ns


class FixedPairPipeline(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ns = classes()
        self.pairs_path = self.root / "pairs.json"
        self.pairs_path.write_text(json.dumps({"schema_version": 1, "split": "train", "pairs": [
            {"scene": scene, "context": [3, 0], "target": [1, 2]} for scene in ("a", "b", "c")]}))
        folder = self.root / "train"
        folder.mkdir()
        self.chunk_paths = [folder / f"{index}.torch" for index in range(2)]
        for path in self.chunk_paths:
            path.touch()
        (folder / "index.json").write_text(json.dumps({"a": "0.torch", "b": "0.torch", "c": "1.torch"}))
        sampler_cfg = self.ns["ViewSamplerFixedPairsCfg"]("fixed_pairs", self.pairs_path, self.root / "eval.json")
        self.sampler = self.ns["ViewSamplerFixedPairs"](sampler_cfg, "train", False, False, None)
        self.cfg = self.ns["DatasetRE10kCfg"](
            original_image_shape=[4, 6], input_image_shape=[4, 6], background_color=[0., 0., 0.],
            cameras_are_circular=False, overfit_to_scene=None, view_sampler=sampler_cfg,
            name="re10k", roots=[self.root], baseline_min=1e-3, baseline_max=1e10, max_fov=100.,
            make_baseline_1=False, augment=False, relative_pose=False, skip_bad_shape=True,
            foundation_prior=FoundationPriorCfg(enabled=True, source="mock", allow_mock=True))
        self.chunks = {self.chunk_paths[0]: [self.row("a"), self.row("b"), self.row("unselected")],
                       self.chunk_paths[1]: [self.row("c")]}

    @staticmethod
    def row(scene):
        return {"key": scene, "cameras": torch.zeros(4, 18),
                "images": [torch.full((3, 4, 6), i / 4) for i in range(4)]}

    def dataset(self):
        dataset = self.ns["DatasetRE10k"](self.cfg, "train", self.sampler)
        dataset.convert_poses = lambda poses: (torch.eye(4).repeat(len(poses), 1, 1), torch.eye(3).repeat(len(poses), 1, 1))
        dataset.convert_images = torch.stack
        dataset.get_bound = lambda bound, n: torch.ones(n)
        return dataset

    def rows(self, dataset, worker=None):
        with patch("torch.load", side_effect=lambda path: self.chunks[path]), patch(
                "torch.utils.data.get_worker_info", return_value=worker):
            return list(dataset)

    def test_actual_sampler_order_and_bounds(self):
        context, target, _ = self.sampler.sample("a", torch.zeros(4, 4, 4), None)
        self.assertEqual(context.tolist(), [3, 0])
        self.assertEqual(target.tolist(), [1, 2])
        self.assertEqual(self.sampler.num_target_views, 2)
        with self.assertRaises(ValueError):
            self.sampler.sample("a", torch.zeros(3, 4, 4), None)

    def test_sampler_rejects_overfit_and_nontrain(self):
        for stage, overfit in (("test", False), ("train", True)):
            with self.assertRaises(ValueError):
                self.ns["ViewSamplerFixedPairs"](self.cfg.view_sampler, stage, overfit, False, None)

    def test_factory_routes_eval_to_separate_index(self):
        actual_definitions("src/dataset/view_sampler/view_sampler_evaluation.py", ["ViewSamplerEvaluationCfg"], self.ns)
        self.ns["VIEW_SAMPLERS"] = {"fixed_pairs": self.ns["ViewSamplerFixedPairs"],
                                   "evaluation": lambda cfg, *args: cfg}
        actual_definitions("src/dataset/view_sampler/__init__.py", ["get_view_sampler"], self.ns)
        for stage in ("val", "test"):
            cfg = self.ns["get_view_sampler"](self.cfg.view_sampler, stage, False, False, None)
            self.assertEqual(cfg.name, "evaluation")
            self.assertEqual(cfg.index_path, self.root / "eval.json")

    def test_actual_iterator_filters_and_attaches_maps(self):
        rows = self.rows(self.dataset())
        self.assertEqual({row["scene"] for row in rows}, {"a", "b", "c"})
        for row in rows:
            self.assertEqual(row["context"]["index"].tolist(), [3, 0])
            self.assertEqual(row["_foundation"]["teacher_target"].shape, (2, 4, 6))
            self.assertTrue(row["_foundation"]["is_mock"])

    def test_worker_sharding_disjoint_and_repeatable_across_epochs(self):
        workers = [SimpleNamespace(id=i, num_workers=2) for i in range(2)]
        datasets = [self.dataset(), self.dataset()]
        for _ in range(3):
            sets = [{row["scene"] for row in self.rows(dataset, worker)} for dataset, worker in zip(datasets, workers)]
            self.assertFalse(sets[0] & sets[1])
            self.assertEqual(sets[0] | sets[1], {"a", "b", "c"})
            for dataset in datasets:
                self.assertEqual(dataset.chunks, self.chunk_paths)

    def test_missing_chunk_content_and_duplicates_fail(self):
        original = self.chunks[self.chunk_paths[0]]
        for invalid in ([self.row("b")], [self.row("a"), self.row("a"), self.row("b")]):
            self.chunks[self.chunk_paths[0]] = invalid
            with self.assertRaises(ValueError):
                self.rows(self.dataset())
        self.chunks[self.chunk_paths[0]] = original

    def test_missing_scene_in_index_fails_before_iteration(self):
        (self.root / "train/index.json").write_text(json.dumps({"a": "0.torch", "b": "0.torch"}))
        with self.assertRaises(ValueError):
            self.dataset()

    def test_frame_and_image_shape_rejection_is_not_silent(self):
        self.chunks[self.chunk_paths[0]][0]["cameras"] = torch.zeros(2, 18)
        with self.assertRaises(ValueError):
            self.rows(self.dataset())
        self.chunks[self.chunk_paths[0]][0] = self.row("a")
        self.chunks[self.chunk_paths[0]][0]["images"] = [torch.zeros(3, 2, 2)] * 4
        with self.assertRaises(ValueError):
            self.rows(self.dataset())

    def test_disabled_prior_keeps_original_example_fields(self):
        self.cfg.foundation_prior = FoundationPriorCfg()
        for row in self.rows(self.dataset()):
            self.assertEqual(set(row), {"scene", "context", "target"})

    def test_preflight_rejects_mismatched_loss_and_loader_before_warmup(self):
        cfg = SimpleNamespace(mode="train", dataset=[SimpleNamespace(re10k=self.cfg)],
            loss=[SimpleNamespace(reliability_prior=SimpleNamespace(enabled=True, weight=.001, allow_mock=True))],
            model=SimpleNamespace(anchor_selection=SimpleNamespace(enabled=False),
                                  encoder=SimpleNamespace(gaussian_reliability=SimpleNamespace(enabled=True))))
        validate_prior_pipeline(cfg)
        self.cfg.foundation_prior.enabled = False
        with self.assertRaises(ValueError):
            validate_prior_pipeline(cfg)
        cfg.mode = "test"
        validate_prior_pipeline(cfg)


if __name__ == "__main__":
    unittest.main()
