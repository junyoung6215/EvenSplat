"""Validation regressions without Lightning, CUDA, model downloads, or video I/O.

The real ValidationWrapper is exercised through torch DataLoader. ModelWrapper
methods execute unchanged ASTs with renderer/visualization doubles; these tests
cover routing and reuse, not rasterizer numerics or actual Lightning lifecycle.
"""
import ast
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "validation_wrapper_under_test", ROOT / "src/dataset/validation_wrapper.py")
wrapper_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper_module)
ValidationWrapper = wrapper_module.ValidationWrapper


class TinyIterable(torch.utils.data.IterableDataset):
    def __init__(self, values):
        self.values = values
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        yield from self.values


class ValidationIteratorTests(unittest.TestCase):
    def test_repeated_dataloader_checks_cycle_small_subset(self):
        dataset = TinyIterable([10, 20])
        wrapped = ValidationWrapper(dataset, 1)
        loader = torch.utils.data.DataLoader(wrapped, batch_size=1, num_workers=0)
        self.assertEqual([[x.item() for x in loader] for _ in range(5)],
                         [[10], [20], [10], [20], [10]])
        self.assertEqual(len(wrapped), 1)  # unchanged one-sample validation policy
        self.assertEqual(dataset.iterations, 3)

    def test_wraps_within_one_loader_iteration(self):
        loader = torch.utils.data.DataLoader(
            ValidationWrapper(TinyIterable([1, 2]), 5), batch_size=1, num_workers=0)
        self.assertEqual([x.item() for x in loader], [1, 2, 1, 2, 1])

    def test_one_scene_can_be_validated_repeatedly(self):
        wrapped = ValidationWrapper(TinyIterable([7]), 1)
        self.assertEqual([wrapped[0] for _ in range(5)], [7] * 5)

    def test_empty_iterable_is_an_explicit_error_not_an_empty_epoch(self):
        dataset = TinyIterable([])
        loader = torch.utils.data.DataLoader(
            ValidationWrapper(dataset, 1), batch_size=1, num_workers=0)
        with self.assertRaisesRegex(RuntimeError, "Validation dataset yielded no samples"):
            list(loader)
        self.assertEqual(dataset.iterations, 2)

    def test_dataset_errors_are_not_swallowed(self):
        class Broken(torch.utils.data.IterableDataset):
            def __iter__(self):
                raise ValueError("bad camera")
                yield
        with self.assertRaisesRegex(ValueError, "bad camera"):
            ValidationWrapper(Broken(), 1)[0]

    def test_map_style_sampling_is_unchanged(self):
        wrapped = ValidationWrapper([42], 1)
        self.assertEqual(wrapped[0], 42)


def actual_methods(names, namespace=None):
    tree = ast.parse((ROOT / "src/model/model_wrapper.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelWrapper")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(methods) == len(names)
    scope = {"torch": torch, "rank_zero_only": lambda f: f, **(namespace or {})}
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *methods], type_ignores=[])),
                 "actual_model_wrapper_methods", "exec"), scope)
    return type("ValidationHarness", (), {name: scope[name] for name in names})


def batch():
    def views(n):
        return dict(image=torch.zeros(1, n, 3, 2, 2),
                    extrinsics=torch.eye(4).expand(1, n, 4, 4),
                    intrinsics=torch.eye(3).expand(1, n, 3, 3),
                    near=torch.ones(1, n), far=torch.ones(1, n) * 10,
                    index=torch.arange(n)[None])
    return {"context": views(2), "target": views(1), "scene": ["scene"]}


class ValidationVideoTests(unittest.TestCase):
    def test_validation_metrics_and_videos_receive_exact_same_subset(self):
        namespace = dict(
            compute_psnr=lambda *a: torch.tensor(1.), compute_lpips=lambda *a: torch.tensor(1.),
            compute_ssim=lambda *a: torch.tensor(1.), vis_depth_map=lambda x: x,
            inverse_normalize=lambda x: x, hcat=lambda *a: None, vcat=lambda *a: None,
            add_label=lambda *a: None, add_border=lambda x: x, prep_image=lambda x: x,
            render_projections=lambda *a, **kw: [[None]], render_cameras=lambda *a: [None])
        cls = actual_methods(["validation_step", "_render_validation_videos"], namespace)
        for kept, enabled in ((4, True), (8, True), (4, False)):
            with self.subTest(kept=kept, videos=enabled):
                obj = cls()
                obj.global_rank, obj.global_step = 1, 500
                obj.data_shim = lambda b: b
                obj._prepare_anchorgeo_batch = lambda b, *a: b
                obj.distiller = obj.encoder_visualizer = None
                obj.train_cfg = SimpleNamespace(validation_videos=enabled, extended_visualization=True)
                obj.log, obj.logger = Mock(), SimpleNamespace(log_image=Mock())
                full = SimpleNamespace(opacities=torch.ones(1, 8))
                selected = full if kept == 8 else SimpleNamespace(opacities=torch.ones(1, kept))
                def encode(context, step, visualization_dump):
                    visualization_dump["depth"] = torch.ones(1, 2, 2, 2, 1, 1)
                    return full
                obj.encoder = Mock(side_effect=encode)
                obj._apply_gaussian_retention = Mock(return_value=SimpleNamespace(gaussians=selected))
                obj.decoder = SimpleNamespace(forward=Mock(return_value=SimpleNamespace(
                    color=torch.zeros(1, 1, 3, 2, 2), depth=torch.ones(1, 1, 2, 2))))
                obj.render_video_interpolation = Mock()
                obj.render_video_wobble = Mock()
                obj.render_video_interpolation_exaggerated = Mock()
                b = batch()
                obj.validation_step(b, 3)
                obj.encoder.assert_called_once()
                obj._apply_gaussian_retention.assert_called_once()
                self.assertIs(obj.decoder.forward.call_args.args[0], selected)
                for video in (obj.render_video_interpolation, obj.render_video_wobble,
                              obj.render_video_interpolation_exaggerated):
                    if enabled:
                        video.assert_called_once()
                        self.assertIs(video.call_args.args[0], b)
                        self.assertIs(video.call_args.args[1], selected)
                    else:
                        video.assert_not_called()

    def test_standard_video_dispatch_does_not_add_extended_video(self):
        obj = actual_methods(["_render_validation_videos"])()
        obj.train_cfg = SimpleNamespace(validation_videos=True, extended_visualization=False)
        obj.render_video_interpolation = Mock()
        obj.render_video_wobble = Mock()
        obj.render_video_interpolation_exaggerated = Mock()
        obj._render_validation_videos(batch(), object())
        obj.render_video_interpolation.assert_called_once()
        obj.render_video_wobble.assert_called_once()
        obj.render_video_interpolation_exaggerated.assert_not_called()

    def test_all_trajectory_helpers_forward_provided_gaussians(self):
        names = ["render_video_interpolation", "render_video_wobble", "render_video_interpolation_exaggerated"]
        obj = actual_methods(names)()
        obj.render_video_generic = Mock()
        selected = object()
        for name in names:
            with self.subTest(name=name):
                obj.render_video_generic.reset_mock()
                getattr(obj, name)(batch(), selected)
                obj.render_video_generic.assert_called_once()
                self.assertIs(obj.render_video_generic.call_args.args[3], selected)

    def test_generic_video_uses_subset_without_encoder_or_reselection(self):
        saved = []
        scope = dict(
            repeat=lambda x, pattern, v: x[:, None].expand(x.shape[0], v),
            vis_depth_map=lambda x: x[:, None].expand(-1, 3, -1, -1),
            vcat=lambda rgb, depth: torch.cat((rgb, depth), dim=1),
            pack=lambda arrays, pattern: (np.concatenate(arrays, axis=0), None),
            wandb=SimpleNamespace(Video=lambda array, **kw: array, log=saved.append))
        obj = actual_methods(["render_video_generic"], scope)()
        obj.device = torch.device("cpu")
        obj.encoder = Mock(side_effect=AssertionError("Unexpected re-encoding"))
        obj._apply_gaussian_retention = Mock(side_effect=AssertionError("Unexpected re-selection"))
        selected = object()
        obj.decoder = SimpleNamespace(forward=Mock(return_value=SimpleNamespace(
            color=torch.zeros(1, 3, 3, 2, 2), depth=torch.ones(1, 3, 2, 2))))
        trajectory = lambda t: (torch.eye(4).expand(1, len(t), 4, 4),
                                torch.eye(3).expand(1, len(t), 3, 3))
        obj.render_video_generic(batch(), trajectory, "rgb", selected, num_frames=3)
        obj.encoder.assert_not_called()
        obj._apply_gaussian_retention.assert_not_called()
        self.assertIs(obj.decoder.forward.call_args.args[0], selected)
        self.assertEqual(saved[0]["video/rgb"].shape, (1, 4, 3, 4, 2))


class ValidationVideoConfigTests(unittest.TestCase):
    def test_legacy_default_preserved_and_retention_presets_opt_out(self):
        main = yaml.safe_load((ROOT / "config/main.yaml").read_text(encoding="utf-8"))
        prior = yaml.safe_load((ROOT / "config/experiment/retention_prior.yaml").read_text(encoding="utf-8"))
        self.assertTrue(main["train"]["validation_videos"])
        self.assertFalse(prior["train"]["validation_videos"])
        self.assertTrue(main["test"]["align_pose"])
        tree = ast.parse((ROOT / "src/model/model_wrapper.py").read_text(encoding="utf-8"))
        cfg = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TrainCfg")
        scope = {"dataclass": dataclass}
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, cfg], type_ignores=[])), "train_cfg", "exec"), scope)
        self.assertTrue(scope["TrainCfg"](None, False, 10, "", 100).validation_videos)

    def test_runner_exposes_video_switch_without_changing_alignment(self):
        from src.experiment_runner import retention as runner
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                cfg = runner.read_settings(runner.TEMPLATE)
                cfg["train"]["validation_videos"] = enabled
                plan = runner.build_plan(cfg, runner.parser().parse_args(["train"]))
                argv = plan["jobs"][0]["argv"]
                self.assertIn(f"train.validation_videos={str(enabled).lower()}", argv)
                self.assertIn("test.align_pose=true", argv)

    def test_runner_rejects_string_instead_of_boolean(self):
        from src.experiment_runner import retention as runner
        cfg = runner.read_settings(runner.TEMPLATE)
        cfg["train"]["validation_videos"] = "false"
        with self.assertRaisesRegex(ValueError, "train.validation_videos"):
            runner.build_plan(cfg, runner.parser().parse_args(["train"]))


if __name__ == "__main__":
    unittest.main()
