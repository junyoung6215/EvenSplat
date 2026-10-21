"""Launcher tests use tiny fake inputs and fake subprocesses, never a GPU."""
import ast
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiment_runner import retention as runner


class RetentionRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = runner.read_settings(runner.TEMPLATE)
        self.cfg.update(data_root=str(self.root), output_root=str(self.root / "runs"))
        self.cfg["train"]["sampler"] = "fixed"  # most tests below cover the fixed-pair path
        for key in ("initial_checkpoint", "train_pairs", "validation_index", "evaluation_index"):
            self.cfg[key] = str(self.root / (key + ".json"))
        Path(self.cfg["initial_checkpoint"]).write_bytes(b"FAKE checkpoint: never loaded")
        self.write("train_pairs", {"schema_version": 1, "split": "train", "pairs": [
            {"scene": "train_scene", "context": [0, 10], "target": [5]}]})
        self.write("validation_index", {"val_scene": {"context": [0, 10], "target": [5]}})
        self.write("evaluation_index", {"test_scene": {"context": [0, 10], "target": [5]}})

    def write(self, key, obj):
        Path(self.cfg[key]).write_text(json.dumps(obj), encoding="utf-8")

    def plan(self, *args):
        return runner.build_plan(self.cfg, runner.parser().parse_args(list(args)))

    def values(self, job):
        return {k: json.loads(v) for arg in job["argv"][5:] if "=" in arg
                for k, v in [arg.split("=", 1)] if k != "loss" and "@" not in k}

    def test_train_resets_optimizer_and_saves_full_checkpoint(self):
        p = self.plan("train")
        v = self.values(p["jobs"][0])
        self.assertIsNone(v["checkpointing.load"])
        self.assertEqual(v["model.encoder.pretrained_weights"], Path(self.cfg["initial_checkpoint"]).as_posix())
        self.assertFalse(v["checkpointing.save_weights_only"])
        self.assertFalse(v["model.gaussian_retention.enabled"])

    def test_loss_arms_share_inputs_and_schedule(self):
        self.cfg["prior"].update(root=str(self.root), teacher_id="real", teacher_settings_id="v1")
        plans = [self.plan("train", "--arm", arm) for arm in ("control", "legacy", "prior")]
        for key in ("dataset.re10k.roots", "dataset.re10k.view_sampler.index_path",
                    "model.encoder.pretrained_weights", "trainer.max_steps", "optimizer.lr", "seed"):
            self.assertEqual(len({str(self.values(p["jobs"][0])[key]) for p in plans}), 1)
        legacy = plans[1]["jobs"][0]["argv"]
        self.assertIn("loss=[mse,lpips,reliability]", legacy)
        self.assertIn("~loss.reliability_prior", legacy)
        prior = self.values(plans[2]["jobs"][0])
        self.assertEqual(prior["loss.reliability_prior.warmup_start"], 500)

    def test_random_sampler_uses_bounded_pairs_without_manifest(self):
        self.cfg["train"]["sampler"] = "random"
        Path(self.cfg["train_pairs"]).unlink()
        p = self.plan("train", "--prune")
        argv, v = p["jobs"][0]["argv"], self.values(p["jobs"][0])
        self.assertIn("dataset@dataset.re10k=re10k", argv)
        self.assertNotIn("dataset.re10k.view_sampler.index_path", v)
        self.assertEqual(v["dataset.re10k.view_sampler.warm_up_steps"], 0)
        self.assertEqual(v["dataset.re10k.view_sampler.evaluation_index_path"],
                         Path(self.cfg["validation_index"]).resolve().as_posix())
        self.assertEqual(p["training"]["sampler"], "random")
        runner.preflight(p)  # no train_pairs file needed
        eval_argv = self.plan("eval-grid")["jobs"][0]["argv"]
        self.assertNotIn("dataset@dataset.re10k=re10k", eval_argv)
        self.cfg["prior"].update(root=str(self.root), teacher_id="real", teacher_settings_id="v1")
        with self.assertRaises(ValueError):
            self.plan("train", "--arm", "prior")

    def test_training_scope_defaults_and_cli_overrides(self):
        self.assertEqual(self.values(self.plan("train")["jobs"][0])["train.finetune_scope"], "full")
        self.cfg["train"]["finetune_scope"] = "heads"
        p = self.plan("train", "--accumulate", "4", "--steps", "2000")
        v = self.values(p["jobs"][0])
        self.assertEqual(v["train.finetune_scope"], "heads")
        self.assertEqual(v["trainer.accumulate_grad_batches"], 4)
        self.assertEqual(p["training"], dict(finetune_scope="heads", accumulate=4,
                                           effective_batch_size=4, optimizer_steps=2000, sampler="fixed"))
        self.assertEqual(self.cfg["train"]["accumulate"], 1)  # no settings mutation
        self.assertEqual(self.values(self.plan("train", "--train-scope", "full")["jobs"][0])["train.finetune_scope"], "full")
        self.assertIsNone(self.plan("eval-grid")["training"])

    def test_scope_ablation_changes_only_scope_and_run_directory(self):
        a = self.values(self.plan("train", "--train-scope", "heads", "--prune", "--accumulate", "4")["jobs"][0])
        b = self.values(self.plan("train", "--train-scope", "full", "--prune", "--accumulate", "4")["jobs"][0])
        self.assertEqual({k for k in a if a[k] != b[k]}, {"train.finetune_scope", "hydra.run.dir", "test.output_path"})

    def test_invalid_training_scope_or_accumulation_rejected(self):
        for value in (0, -1):
            with self.assertRaises(ValueError):
                self.plan("train", "--accumulate", str(value))
        self.cfg["train"]["finetune_scope"] = "invalid"
        with self.assertRaises(ValueError):
            self.plan("train")

    def test_finetune_and_matched_continuation(self):
        a = self.values(self.plan("train", "--prune", "--local", "--steps", "1000")["jobs"][0])
        b = self.values(self.plan("train", "--steps", "1000")["jobs"][0])
        self.assertTrue(a["model.gaussian_retention.enabled"])
        self.assertTrue(a["model.gaussian_retention.local_cells.enabled"])
        self.assertEqual(a["trainer.max_steps"], b["trainer.max_steps"])
        self.assertEqual(a["model.encoder.pretrained_weights"], b["model.encoder.pretrained_weights"])

    def test_default_grid_is_12_conditions_plus_two_identity_checks(self):
        p = self.plan("eval-grid")
        self.assertEqual(len(p["jobs"]), 14)
        self.assertEqual(len({j["directory"] for j in p["jobs"]}), 14)
        self.assertEqual(p["split"], "validation")
        self.assertEqual(self.values(p["jobs"][0])["model.encoder.pretrained_weights"], "")

    def test_local_pairing_and_test_index(self):
        p = self.plan("eval-grid", "--scores", "reliability", "--ratios", ".5", "--local", "both", "--split", "test")
        self.assertEqual(len(p["jobs"]), 4)
        self.assertEqual([j["local"] for j in p["jobs"][-2:]], [False, True])
        self.assertEqual(p["index"], str(Path(self.cfg["evaluation_index"]).resolve()))

    def test_rejects_invalid_or_ineffective_options(self):
        for args in (("train", "--local"), ("train", "--arm", "prior"),
                     ("train", "--steps", "0"), ("train", "--run-id", "../escape"),
                     ("eval-grid", "--ratios", "0"), ("eval-grid", "--ratios", "nan"),
                     ("eval-grid", "--ratios", ".5", ".5")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.plan(*args)

    def test_disabled_reliability_cannot_rank_by_reliability(self):
        self.cfg["reliability_enabled"] = False
        with self.assertRaises(ValueError):
            self.plan("eval-grid")
        self.plan("eval-grid", "--scores", "opacity")

    def test_single_gpu_guard(self):
        self.cfg["gpu"] = "0,1"
        with self.assertRaises(ValueError):
            self.plan("train")

    def test_unknown_setting_rejected(self):
        path = self.root / "bad.yaml"
        path.write_text("train:\n  stepz: 50\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "stepz"):
            runner.read_settings(path)

    def test_path_quoting_keeps_one_literal_argument(self):
        self.cfg["initial_checkpoint"] = str(self.root / 'checkpoint with spaces.ckpt')
        p = self.plan("train")
        v = self.values(p["jobs"][0])
        self.assertEqual(v["model.encoder.pretrained_weights"], Path(self.cfg["initial_checkpoint"]).as_posix())
        with self.assertRaises(ValueError):
            runner.override("x", "${unsafe_interpolation}")

    def test_interruption_is_recorded_and_propagated(self):
        plan = self.plan("train")
        def interrupted(*args, **kwargs):
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            runner.execute(plan, run=interrupted)
        manifest = json.loads((Path(plan["output"]) / "manifest.json").read_text())
        self.assertEqual(manifest["jobs"][0]["status"], "interrupted")

    def test_nan_metrics_are_not_reported_as_valid(self):
        plan = self.plan("eval-grid", "--scores", "opacity", "--ratios", ".5")
        runner.execute(plan, run=lambda *a, **k: SimpleNamespace(returncode=0))
        directory = Path(plan["jobs"][0]["directory"])
        directory.mkdir()
        (directory / "test_metrics.json").write_text(json.dumps([
            {"psnr_ours": float("nan"), "ssim_ours": .8, "lpips_ours": .2}]))
        path = self.root / "summary.csv"
        runner.summarize(self.root / "runs", path)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["metrics_status"], "incomplete")
        self.assertEqual(row["psnr"], "")

    def test_preview_never_executes_or_creates_output(self):
        with patch.object(runner, "read_settings", return_value=self.cfg), patch.object(runner, "execute") as execute:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(["train"]), 0)
            execute.assert_not_called()
        self.assertFalse(Path(self.cfg["output_root"]).exists())

    def test_preflight_blocks_data_leakage(self):
        runner.preflight(self.plan("train"))
        self.write("validation_index", {"train_scene": {"context": [0, 10], "target": [5]}})
        with self.assertRaisesRegex(ValueError, "overlap"):
            runner.preflight(self.plan("train"))

    def test_preflight_blocks_validation_test_overlap(self):
        self.write("evaluation_index", {"val_scene": {"context": [0, 10], "target": [5]}})
        with self.assertRaisesRegex(ValueError, "overlap"):
            runner.preflight(self.plan("eval-grid"))

    def test_failure_stops_sweep_and_records_pending_jobs(self):
        plan = self.plan("eval-grid", "--scores", "opacity", "--ratios", ".5")
        calls = []
        def failed(argv, **kwargs):
            calls.append(argv)
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")
            self.assertNotIn("shell", kwargs)
            return SimpleNamespace(returncode=7)
        self.assertEqual(runner.execute(plan, run=failed), 7)
        self.assertEqual(len(calls), 1)
        manifest = json.loads((Path(plan["output"]) / "manifest.json").read_text())
        self.assertEqual([j["status"] for j in manifest["jobs"]], ["failed", "pending", "pending"])
        self.assertEqual(len(manifest["checkpoint_sha256"]), 64)
        with self.assertRaises(FileExistsError):
            runner.execute(plan, run=failed)

    def test_summary_keeps_missing_values_blank(self):
        plan = self.plan("eval-grid", "--scores", "opacity", "--ratios", ".5")
        runner.execute(plan, run=lambda *a, **k: SimpleNamespace(returncode=0))
        directory = Path(plan["jobs"][1]["directory"])
        directory.mkdir()
        (directory / "test_metrics.json").write_text(json.dumps([
            {"psnr_ours": 25., "ssim_ours": .8, "lpips_ours": .2,
             "eval/retention_output_count": 100.}]))
        csv_path = self.root / "summary.csv"
        self.assertEqual(runner.summarize(self.root / "runs", csv_path), 3)
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["psnr"], "")
        self.assertEqual(rows[0]["metrics_status"], "missing")
        self.assertEqual(rows[1]["output_count"], "100.0")
        with self.assertRaises(FileExistsError):
            runner.summarize(self.root / "runs", csv_path)


class LocalLoggerIsolationTests(unittest.TestCase):
    def test_local_logger_preserves_files_and_isolates_artifacts(self):
        # Execute just the class with a trivial base, without importing Lightning.
        tree = ast.parse((ROOT / "src/misc/LocalLogger.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        from typing import Any, Optional
        class FakeImage:
            @staticmethod
            def fromarray(value):
                return SimpleNamespace(save=lambda p: p.write_bytes(b"image"))
        scope = dict(Logger=object, rank_zero_only=lambda f: f, Path=Path,
                     LOG_PATH=Path("unused"), Any=Any, Optional=Optional, json=json, Image=FakeImage)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), "logger_test", "exec"), scope)
        with tempfile.TemporaryDirectory() as root:
            old = Path(root) / "keep.txt"
            old.write_text("keep")
            a = scope["LocalLogger"](Path(root) / "a")
            b = scope["LocalLogger"](Path(root) / "b")
            a.log_image("val/image", [None], step=1)
            a.log_metrics({"loss/total": 1.2}, step=1)
            b.log_metrics({"loss/total": 2.3}, step=1)
            self.assertEqual(old.read_text(), "keep")
            self.assertTrue((a.output_dir / "val/image/00_000001.png").exists())
            self.assertNotEqual((a.output_dir / "metrics.jsonl").read_text(),
                                (b.output_dir / "metrics.jsonl").read_text())


class HydraRunnerTests(unittest.TestCase):
    def test_generated_commands_compose_when_hydra_available(self):
        from _test_support import require_test_dependencies
        require_test_dependencies("hydra", "omegaconf")
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        cfg = runner.read_settings(runner.TEMPLATE)
        cfg["prior"].update(root="unused", teacher_id="teacher", teacher_settings_id="v1")
        variants = [("train", "--arm", a) for a in ("control", "legacy", "prior")]
        variants += [("eval-grid",), ("train", "--prune", "--local")]
        variants += [("train", "--train-scope", "heads", "--accumulate", "4")]
        for sampler in ("fixed", "random"):
          cfg["train"]["sampler"] = sampler
          for args in variants:
            if sampler == "random" and "prior" in args:
                continue
            plan = runner.build_plan(cfg, runner.parser().parse_args(args))
            with self.subTest(args=args), initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
                composed = compose(config_name="main", overrides=plan["jobs"][0]["argv"][4:])
                OmegaConf.to_container(composed, resolve=True, throw_on_missing=True)
                expected = "bounded" if sampler == "random" and args[0] == "train" else "fixed_pairs"
                self.assertEqual(composed.dataset.re10k.view_sampler.name, expected)
                if "legacy" in args:
                    self.assertEqual(set(composed.loss), {"mse", "lpips", "reliability"})


if __name__ == "__main__":
    unittest.main()
