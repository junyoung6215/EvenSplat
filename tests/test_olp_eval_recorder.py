from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _test_support import add_function_tests, require_test_dependencies

require_test_dependencies("jaxtyping", "beartype", direct_run=__name__ == "__main__")
from jaxtyping import install_import_hook

with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.analysis.olp_eval_recorder import OlpEvalRecorder
    from src.analysis.olp_stats import save_stats_json


def test_save_stats_json_accepts_by_overlap_summary(tmp_path):
    stats = {
        "large": {
            "floater_proxy": 0.001,
            "psnr_ours": 24.4,
        },
        "small": {
            "floater_proxy": 0.002,
            "psnr_ours": 20.9,
        },
    }

    path = tmp_path / "olp_stats_summary_by_overlap.json"
    save_stats_json(stats, path)

    assert json.loads(path.read_text()) == stats


def test_olp_eval_recorder_writes_summary_files(tmp_path):
    recorder = OlpEvalRecorder()
    recorder.per_scene = [
        {
            "scene": "scene_a",
            "batch_idx": 0,
            "overlap_tag": "large",
            "overlap": 0.7,
            "floater_proxy": 0.1,
            "psnr_ours": 24.0,
        },
        {
            "scene": "scene_b",
            "batch_idx": 1,
            "overlap_tag": "large",
            "overlap": 0.6,
            "floater_proxy": 0.3,
            "psnr_ours": 22.0,
        },
        {
            "scene": "scene_c",
            "batch_idx": 2,
            "overlap_tag": "small",
            "overlap": 0.2,
            "floater_proxy": 0.5,
            "psnr_ours": 20.0,
        },
    ]

    recorder.write(tmp_path)

    per_scene_path = tmp_path / "olp_stats_per_scene.json"
    summary_path = tmp_path / "olp_stats_summary.json"
    by_overlap_path = tmp_path / "olp_stats_summary_by_overlap.json"
    csv_path = tmp_path / "olp_stats_per_scene.csv"

    assert per_scene_path.exists()
    assert summary_path.exists()
    assert by_overlap_path.exists()
    assert csv_path.exists()

    summary = json.loads(summary_path.read_text())
    assert math.isclose(summary["psnr_ours"], 22.0)
    assert math.isclose(summary["floater_proxy"], 0.3)

    by_overlap = json.loads(by_overlap_path.read_text())
    assert math.isclose(by_overlap["large"]["psnr_ours"], 23.0)
    assert math.isclose(by_overlap["large"]["floater_proxy"], 0.2)
    assert math.isclose(by_overlap["small"]["psnr_ours"], 20.0)
    assert math.isclose(by_overlap["small"]["floater_proxy"], 0.5)


def test_olp_eval_recorder_records_reliability_stats(tmp_path):
    class DummyEncoder(torch.nn.Module):
        pass

    recorder = OlpEvalRecorder()
    visualization_dump = {
        "opacities": torch.tensor([0.1, 0.9]),
        "scales": torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
        "reliability": torch.tensor([0.25, 0.75]),
    }

    recorder.record_test_batch(DummyEncoder(), visualization_dump, "scene_a", 0)
    recorder.write(tmp_path)

    stats = recorder.per_scene[0]
    assert math.isclose(stats["reliability_mean"], 0.5)
    assert math.isclose(stats["reliability_min"], 0.25)
    assert math.isclose(stats["reliability_max"], 0.75)
    assert math.isclose(stats["reliability_below_050"], 0.5)

    summary = json.loads((tmp_path / "olp_stats_summary.json").read_text())
    assert math.isclose(summary["reliability_mean"], 0.5)


def load_tests(loader, suite, pattern):
    return add_function_tests(suite, globals())


if __name__ == "__main__":
    with TemporaryDirectory() as tmp:
        test_save_stats_json_accepts_by_overlap_summary(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_olp_eval_recorder_writes_summary_files(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_olp_eval_recorder_records_reliability_stats(Path(tmp))
