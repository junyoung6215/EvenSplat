"""Portable retention experiment launcher; preview by default, --execute to run."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment_runner.retention import main

if __name__ == "__main__":
    raise SystemExit(main())
