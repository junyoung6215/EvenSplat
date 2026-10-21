"""Always check YAML contracts; compose real Hydra presets when installed.

No checkpoint, dataset, model construction or GPU execution is required.
"""
import sys
import unittest
from dataclasses import fields
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _test_support import require_test_dependencies
from test_prior_retention import prior_loss_classes


class LossConfigContractTests(unittest.TestCase):
    def test_every_loss_file_has_its_named_wrapper(self):
        for path in sorted((ROOT / "config/loss").glob("*.yaml")):
            with self.subTest(file=path.name):
                payload = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertEqual(set(payload), {path.stem},
                                 "Loss files must wrap fields under their loss name, not loss.enabled/weight.")
                self.assertIsInstance(payload[path.stem], dict)

    def test_prior_yaml_matches_actual_dataclass_fields_and_defaults(self):
        payload = yaml.safe_load((ROOT / "config/loss/reliability_prior.yaml").read_text(encoding="utf-8"))
        cls = prior_loss_classes()["LossReliabilityPriorCfg"]
        self.assertEqual(set(payload["reliability_prior"]), {field.name for field in fields(cls)})
        self.assertEqual(cls(**payload["reliability_prior"]), cls())

    def test_experiment_overrides_keep_same_loss_namespace(self):
        for name in ("retention_prior", "retention_control", "retention_mock"):
            with self.subTest(preset=name):
                payload = yaml.safe_load((ROOT / f"config/experiment/{name}.yaml").read_text(encoding="utf-8"))
                self.assertEqual(set(payload["loss"]), {"reliability_prior"})


class HydraRetentionConfigTests(unittest.TestCase):
    def setUp(self):
        require_test_dependencies("hydra", "omegaconf")

    def compose(self, name):
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
            return compose(config_name="main", overrides=[
                f"+experiment={name}", "model.encoder.pretrained_weights=unused.ckpt",
                "dataset.re10k.view_sampler.index_path=unused_train_pairs.json",
            ])

    def assert_loss_contract(self, cfg):
        from omegaconf import OmegaConf
        self.assertEqual(set(cfg.loss), {"mse", "lpips", "reliability_prior"})
        cls = prior_loss_classes()["LossReliabilityPriorCfg"]
        return cls(**OmegaConf.to_container(cfg.loss.reliability_prior, resolve=True))

    def test_prior_preset_composes_with_complete_loss(self):
        cfg = self.compose("retention_prior")
        loss = self.assert_loss_contract(cfg)
        self.assertTrue(loss.enabled)
        self.assertEqual(loss.weight, .001)
        self.assertFalse(loss.allow_mock)
        self.assertEqual(cfg.dataset.re10k.view_sampler.name, "fixed_pairs")
        self.assertFalse(cfg.train.validation_videos)

    def test_control_preset_disables_prior(self):
        cfg = self.compose("retention_control")
        loss = self.assert_loss_contract(cfg)
        self.assertFalse(loss.enabled)
        self.assertFalse(cfg.dataset.re10k.foundation_prior.enabled)
        self.assertFalse(cfg.train.validation_videos)

    def test_mock_preset_keeps_diagnostic_flags_and_warmup(self):
        cfg = self.compose("retention_mock")
        loss = self.assert_loss_contract(cfg)
        self.assertTrue(loss.enabled)
        self.assertTrue(loss.allow_mock)
        self.assertEqual((loss.warmup_start, loss.warmup_end), (0, 0))
        self.assertTrue(cfg.dataset.re10k.foundation_prior.allow_mock)
        self.assertFalse(cfg.train.validation_videos)


if __name__ == "__main__":
    unittest.main()
