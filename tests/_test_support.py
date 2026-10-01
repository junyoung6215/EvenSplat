"""Test-only import isolation and unittest support; never imported by src."""
import importlib
import importlib.util
import inspect
import sys
import types
import unittest
from contextlib import contextmanager
from importlib.machinery import ModuleSpec
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def annotation_only_jaxtyping():
    """Temporarily supply Float for pure tensor tests without leaking a fake module.

    No import hook or runtime type checking is simulated. Tests needing those
    facilities must require the real dependencies instead.
    """
    try:
        importlib.import_module("jaxtyping")
    except ModuleNotFoundError as exc:
        if exc.name != "jaxtyping":
            raise  # A broken installed package must not silently become a stub.
    else:
        yield
        return

    shim = types.ModuleType("jaxtyping")
    shim.__spec__ = ModuleSpec("jaxtyping", loader=None)

    class Float:
        def __class_getitem__(cls, item):
            return item[0]

    shim.Float = Float
    sys.modules["jaxtyping"] = shim
    try:
        yield
    finally:
        if sys.modules.get("jaxtyping") is shim:
            del sys.modules["jaxtyping"]


def require_test_dependencies(*names, direct_run=False):
    missing = [name for name in names if importlib.util.find_spec(name) is None]
    if missing:
        reason = "Missing test dependencies: " + ", ".join(missing)
        if direct_run:
            raise SystemExit("SKIP (not a passing test run): " + reason)
        raise unittest.SkipTest(reason)


def add_function_tests(suite, namespace):
    """Let unittest discovery run the existing pytest-style function tests too."""
    for name, function in sorted(namespace.items()):
        if not name.startswith("test_") or not inspect.isfunction(function):
            continue

        def run(function=function):
            parameters = inspect.signature(function).parameters
            if not parameters:
                function()
            elif set(parameters) == {"tmp_path"}:
                with TemporaryDirectory() as directory:
                    function(tmp_path=Path(directory))
            else:
                raise TypeError(f"Unsupported unittest fixture in {function.__name__}: {list(parameters)}")

        suite.addTest(unittest.FunctionTestCase(run, description=function.__name__))
    return suite
