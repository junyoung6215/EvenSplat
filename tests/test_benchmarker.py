import json
import tempfile
import unittest
from pathlib import Path

from src.misc.benchmarker import Benchmarker


class BenchmarkerTest(unittest.TestCase):
    def test_time_splits_interval_across_calls(self):
        benchmarker = Benchmarker()
        with benchmarker.time("decoder", num_calls=3):
            pass
        self.assertEqual(len(benchmarker.execution_times["decoder"]), 3)

    def test_stage_memory_dump_is_valid_json(self):
        benchmarker = Benchmarker()
        with benchmarker.memory("decoder"):
            pass
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage_memory.json"
            benchmarker.dump_stage_memory(path)
            data = json.loads(path.read_text())
        self.assertIsInstance(data, dict)


if __name__ == "__main__":
    unittest.main()
