"""Regression checks for shared-process collection, without training dependencies."""
import importlib
import importlib.util
import sys
import unittest
from pathlib import Path

from _test_support import annotation_only_jaxtyping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TestImportIsolation(unittest.TestCase):
    def test_annotation_fallback_does_not_leak_on_success(self):
        before = sys.modules.get("jaxtyping")
        installed = importlib.util.find_spec("jaxtyping") is not None
        with annotation_only_jaxtyping():
            self.assertIsNotNone(importlib.util.find_spec("jaxtyping"))
        if not installed:
            self.assertIs(sys.modules.get("jaxtyping"), before)

    def test_annotation_fallback_does_not_leak_on_exception(self):
        before = sys.modules.get("jaxtyping")
        installed = importlib.util.find_spec("jaxtyping") is not None
        with self.assertRaisesRegex(RuntimeError, "expected"):
            with annotation_only_jaxtyping():
                raise RuntimeError("expected")
        if not installed:
            self.assertIs(sys.modules.get("jaxtyping"), before)

    def test_tensor_tests_collect_in_both_orders_without_fake_package_leaks(self):
        names = ("test_gaussian_retention", "test_local_cell_retention", "test_olp_preconditioner")
        for order in (names, names[::-1]):
            before = {name: sys.modules.get(name) for name in ("src.model.encoder", "src.model.encoder.heads")}
            for name in order:
                module = importlib.import_module(name)
                importlib.reload(module)
            importlib.util.find_spec("jaxtyping")  # Used to raise ValueError here.
            for name, value in before.items():
                self.assertIs(sys.modules.get(name), value)

    def test_unittest_collects_existing_function_tests(self):
        for name, count in (("test_gaussian_retention", 19), ("test_local_cell_retention", 12),
                            ("test_olp_preconditioner", 11)):
            module = importlib.import_module(name)
            self.assertEqual(unittest.defaultTestLoader.loadTestsFromModule(module).countTestCases(), count)


if __name__ == "__main__":
    unittest.main()
