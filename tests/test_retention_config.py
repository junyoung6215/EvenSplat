"""YAML contracts and typed config loading for the retention preset.

No checkpoint, dataset, model construction or GPU execution is required.
"""
import sys
import unittest
from dataclasses import fields
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _test_support import annotation_only_jaxtyping, require_test_dependencies


class LossConfigContractTests(unittest.TestCase):
    def test_every_loss_file_has_its_named_wrapper(self):
        for path in sorted((ROOT / "config/loss").glob("*.yaml")):
            with self.subTest(file=path.name):
                payload = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertEqual(set(payload), {path.stem},
                                 "Loss files must wrap fields under their loss name, not loss.enabled/weight.")
                self.assertIsInstance(payload[path.stem], dict)


class RetentionConfigContractTests(unittest.TestCase):
    def test_main_yaml_lists_every_retention_field_with_its_default(self):
        with annotation_only_jaxtyping():
            from src.model.retention import GaussianRetentionCfg
        main = yaml.safe_load((ROOT / "config/main.yaml").read_text(encoding="utf-8"))
        payload = main["model"]["gaussian_retention"]
        self.assertEqual(set(payload), {field.name for field in fields(GaussianRetentionCfg)})
        self.assertEqual(GaussianRetentionCfg(**payload), GaussianRetentionCfg())


class HydraRetentionConfigTests(unittest.TestCase):
    def setUp(self):
        require_test_dependencies("hydra", "omegaconf", "lightning", "dacite", "diff_gaussian_rasterization")

    def compose(self, *overrides):
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
            return compose(config_name="main", overrides=[
                "+experiment=retention", "model.encoder.pretrained_weights=unused.ckpt", *overrides])

    def test_preset_composes_and_loads_typed(self):
        from src.config import load_typed_root_config
        cfg = self.compose("model.gaussian_retention.enabled=true")
        self.assertEqual(set(cfg.loss), {"mse", "lpips"})
        self.assertFalse(cfg.train.validation_videos)
        typed = load_typed_root_config(cfg)
        retention = typed.model.gaussian_retention
        self.assertTrue(retention.enabled)
        self.assertEqual((retention.score_source, retention.cell_size, retention.r_threshold),
                         ("r_filter", 2, 0.35))


if __name__ == "__main__":
    unittest.main()
