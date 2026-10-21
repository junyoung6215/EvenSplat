"""Real torch freeze/optimizer tests; Lightning methods use unchanged ASTs.

No CUDA/model downloads. This does not replace a server Lightning fit smoke test.
"""
import ast
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model.training_scope import (
    CHECKPOINT_SCOPE_KEY, HEAD_MODULES, apply_training_scope,
    training_scope_report, validate_resume_scope, validate_training_initialization,
)
from src.misc.checkpoint_audit import load_encoder_pretrained_weights


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Dropout(.5))
        self.backbone.add_module("intrinsic_encoder", nn.Linear(4, 4))
        for name in HEAD_MODULES:
            head = nn.Linear(4, 4) if name.startswith("downstream") else nn.Sequential(nn.Linear(4, 4))
            if name.startswith("gaussian"):
                head.add_module("reliability", nn.Linear(4, 4))
            setattr(self, name, head)

    def forward(self, image):
        features = self.backbone(image)
        return features, sum(getattr(self, name)(features) for name in HEAD_MODULES)


def actual_wrapper_class():
    tree = ast.parse((ROOT / "src/model/model_wrapper.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelWrapper")
    names = {"train", "configure_optimizers", "on_save_checkpoint", "on_load_checkpoint"}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(cls.body) == len(names)
    cls.bases = [ast.Name(id="Module", ctx=ast.Load())]
    scope = dict(Module=nn.Module, torch=torch, CHECKPOINT_SCOPE_KEY=CHECKPOINT_SCOPE_KEY,
                 validate_resume_scope=validate_resume_scope,
                 get_cfg=lambda: {"trainer": {"max_steps": 100}})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 "actual_training_methods", "exec"), scope)
    return scope["ModelWrapper"]


def wrapper(scope="heads"):
    obj = actual_wrapper_class()()
    obj.encoder = TinyEncoder()
    obj.train_cfg = SimpleNamespace(finetune_scope=scope)
    obj.optimizer_cfg = SimpleNamespace(lr=.01, warm_up_steps=1, backbone_lr_multiplier=.1)
    apply_training_scope(obj.encoder, scope)
    return obj


class TrainingScopeTests(unittest.TestCase):
    def test_full_is_a_true_noop(self):
        model = TinyEncoder()
        model.downstream_head1.weight.requires_grad_(False)
        model.backbone.eval()
        before = [p.requires_grad for p in model.parameters()]
        apply_training_scope(model, "full")
        self.assertEqual(before, [p.requires_grad for p in model.parameters()])
        self.assertFalse(model.backbone.training)

    def test_heads_freezes_entire_backbone_but_not_geometry_or_reliability(self):
        obj = wrapper()
        self.assertTrue(all(not p.requires_grad for p in obj.encoder.backbone.parameters()))
        for name in HEAD_MODULES:
            self.assertTrue(all(p.requires_grad for p in getattr(obj.encoder, name).parameters()))
        for mode in (True, False, True):
            self.assertIs(obj.train(mode), obj)
            self.assertFalse(obj.encoder.backbone.training)
            self.assertTrue(all(getattr(obj.encoder, n).training == mode for n in HEAD_MODULES))

    def test_real_optimizer_updates_heads_and_preserves_frozen_weights_and_buffers(self):
        torch.manual_seed(42)
        obj = wrapper().train()
        before = {k: v.clone() for k, v in obj.encoder.state_dict().items()}
        optimizer = obj.configure_optimizers()["optimizer"]
        features, output = obj.encoder(torch.randn(8, 4))
        self.assertFalse(features.requires_grad)
        output.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in obj.encoder.backbone.parameters()))
        optimizer.step()
        for name, value in obj.encoder.state_dict().items():
            if name.startswith("backbone."):
                self.assertTrue(torch.equal(before[name], value), name)
        for name in HEAD_MODULES:
            self.assertTrue(any(not torch.equal(before[k], v) for k, v in obj.encoder.state_dict().items()
                                if k.startswith(name + ".")), name)
        self.assertFalse(torch.equal(before["gaussian_param_head.reliability.weight"],
                                     obj.encoder.gaussian_param_head.reliability.weight))

    def test_full_still_backpropagates_to_backbone(self):
        obj = wrapper("full").train()
        features, output = obj.encoder(torch.randn(8, 4))
        self.assertTrue(features.requires_grad)
        output.square().mean().backward()
        self.assertIsNotNone(obj.encoder.backbone[0].weight.grad)

    def test_optimizer_membership_and_legacy_learning_rate_policy(self):
        for scope in ("full", "heads"):
            obj = wrapper(scope)
            groups = obj.configure_optimizers()["optimizer"].param_groups
            included = [id(p) for g in groups for p in g["params"]]
            self.assertEqual(set(included), {id(p) for p in obj.parameters() if p.requires_grad})
            self.assertEqual(len(included), len(set(included)))
            self.assertEqual([g["lr"] for g in groups], [.01, .001])
            self.assertIn(id(obj.encoder.downstream_head1.weight), [id(p) for p in groups[1]["params"]])

    def test_parameter_names_and_saved_tensors_unchanged_by_policy(self):
        model = TinyEncoder()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        apply_training_scope(model, "heads")
        self.assertEqual(set(before), set(model.state_dict()))
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()))
        # A heads checkpoint remains loadable in a fresh full-scope model.
        TinyEncoder().load_state_dict(model.state_dict(), strict=True)

    def test_report_counts_and_serializes(self):
        model = wrapper().encoder
        report = training_scope_report(model, "heads")
        self.assertEqual(report["trainable_parameters"] + report["frozen_parameters"],
                         sum(p.numel() for p in model.parameters()))
        self.assertEqual(report["groups"]["backbone"]["trainable"], 0)
        self.assertTrue(all(n.startswith("backbone.") for n in report["frozen_names"]))
        json.dumps(report)

    def test_invalid_topology_and_shared_params_fail_before_mutation(self):
        for kind in ("missing", "shared", "unknown"):
            model = TinyEncoder()
            if kind == "missing":
                del model.gaussian_param_head2
            if kind == "shared":
                model.downstream_head1 = model.backbone[0]
            with self.assertRaises(ValueError):
                apply_training_scope(model, "typo" if kind == "unknown" else "heads")
            self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_initialization_guard(self):
        with self.assertRaisesRegex(ValueError, "randomly initialized"):
            validate_training_initialization("heads", "train", None, None)
        for args in (("full", "train", None, None), ("heads", "test", None, None),
                     ("heads", "train", "model.ckpt", None), ("heads", "train", None, "resume.ckpt")):
            validate_training_initialization(*args)

    def test_checkpoint_scope_guard_and_eval_compatibility(self):
        obj = wrapper()
        checkpoint = {"state_dict": obj.state_dict()}
        obj.on_save_checkpoint(checkpoint)
        self.assertEqual(checkpoint[CHECKPOINT_SCOPE_KEY], "heads")
        obj._trainer = SimpleNamespace(state=SimpleNamespace(fn="fit"))
        obj.on_load_checkpoint(checkpoint)
        with self.assertRaisesRegex(ValueError, "fresh optimizer"):
            obj.on_load_checkpoint({})  # legacy full-scope checkpoint
        full = wrapper("full")
        full._trainer = obj._trainer
        with self.assertRaisesRegex(ValueError, "fresh optimizer"):
            full.on_load_checkpoint(checkpoint)
        full.on_load_checkpoint({})  # legacy full resume preserved
        full._trainer = SimpleNamespace(state=SimpleNamespace(fn="test"))
        full.on_load_checkpoint(checkpoint)  # heads checkpoint evaluation uses normal defaults

    def test_resume_cannot_silently_leave_frozen_weights_random(self):
        obj = wrapper()
        obj._trainer = SimpleNamespace(state=SimpleNamespace(fn=SimpleNamespace(value="fit")))
        with self.assertRaisesRegex(ValueError, "Frozen backbone weights missing"):
            obj.on_load_checkpoint({CHECKPOINT_SCOPE_KEY: "heads", "state_dict": {}})

    def test_same_scope_optimizer_state_roundtrip_and_inference_parity(self):
        for scope in ("full", "heads"):
            obj = wrapper(scope).train()
            optimizer = obj.configure_optimizers()["optimizer"]
            inputs = torch.randn(8, 4)
            obj.encoder(inputs)[1].square().mean().backward()
            optimizer.step()
            resumed = wrapper(scope)
            resumed.load_state_dict(obj.state_dict(), strict=True)
            resumed_optimizer = resumed.configure_optimizers()["optimizer"]
            resumed_optimizer.load_state_dict(optimizer.state_dict())
            self.assertEqual(len(resumed_optimizer.state), len(optimizer.state))
            normal_inference = wrapper("full").eval()
            normal_inference.load_state_dict(obj.state_dict(), strict=True)
            obj.eval()
            with torch.no_grad():
                self.assertTrue(torch.equal(obj.encoder(inputs)[1], normal_inference.encoder(inputs)[1]))

    def test_initial_weight_audit_requires_complete_backbone_only_when_requested(self):
        with tempfile.TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()):
            path, audit = Path(root) / "weights.ckpt", Path(root) / "audit.json"
            model = TinyEncoder()
            weights = {"encoder." + k: v for k, v in model.state_dict().items()}
            torch.save({"state_dict": weights}, path)
            load_encoder_pretrained_weights(model, path, audit, require_backbone=True)
            del weights["encoder.backbone.0.weight"]
            torch.save({"state_dict": weights}, path)
            with self.assertRaisesRegex(RuntimeError, "all frozen backbone weights"):
                load_encoder_pretrained_weights(model, path, audit, require_backbone=True)
            self.assertEqual(json.loads(audit.read_text())["missing_backbone_parameters"], ["backbone.0.weight"])
            load_encoder_pretrained_weights(model, path, require_backbone=False)

    def test_main_and_constructor_wire_policy_and_audit(self):
        # Wiring checks complement dynamic tests without importing CUDA-dependent modules.
        tree = ast.parse((ROOT / "src/main.py").read_text(encoding="utf-8"))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        self.assertTrue(any(isinstance(n.func, ast.Name) and n.func.id == "validate_training_initialization" for n in calls))
        load = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "load_encoder_pretrained_weights")
        self.assertIn("require_backbone", [k.arg for k in load.keywords])
        tree = ast.parse((ROOT / "src/model/model_wrapper.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelWrapper")
        ctor = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        self.assertTrue(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                            and n.func.id == "apply_training_scope" for n in ast.walk(ctor)))
        cfg = yaml.safe_load((ROOT / "config/main.yaml").read_text())
        self.assertEqual(cfg["train"]["finetune_scope"], "full")


if __name__ == "__main__":
    unittest.main()
