from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import platform
import re
import subprocess
import sys
from pathlib import Path


EXPECTED_TORCH = "2.1.2+cu118"
EXPECTED_CUDA = "11.8"


def run_command(command: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError as exc:
        return 127, str(exc)
    return result.returncode, result.stdout


def read_expected_pins(requirements_path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw_line in requirements_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        if " @ " in line:
            continue
        if "==" not in line:
            continue
        name, version = line.split("==", 1)
        pins[name.strip()] = version.strip()
    return pins


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def check_pins(requirements_path: Path) -> list[str]:
    mismatches = []
    installed_versions = {}
    for dist in metadata.distributions():
        dist_name = dist.metadata.get("Name")
        if dist_name:
            installed_versions[normalize_name(dist_name)] = dist.version

    for name, expected in read_expected_pins(requirements_path).items():
        installed = installed_versions.get(normalize_name(name))
        if installed is None:
            mismatches.append(f"{name}: not installed, expected {expected}")
            continue
        if installed != expected:
            mismatches.append(f"{name}: installed {installed}, expected {expected}")
    return mismatches


def write_artifact(path: Path, command: list[str]) -> None:
    code, output = run_command(command)
    path.write_text(output)
    if code != 0:
        path.write_text(output + f"\n[exit_code={code}]\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--requirements",
        type=Path,
        default=Path("env/requirements-cuda118.lock"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--require-gpu-name",
        default=None,
        help="Optional substring that must appear in at least one visible GPU name.",
    )
    args = parser.parse_args()

    import torch

    output_dir = args.output_dir
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_artifact(output_dir / "pip_freeze.txt", [sys.executable, "-m", "pip", "freeze"])
        write_artifact(output_dir / "conda_explicit.txt", ["conda", "list", "--explicit"])
        write_artifact(output_dir / "nvidia_smi.txt", ["nvidia-smi"])

    code, gpu_query = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    cuda_available = torch.cuda.is_available()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    pin_mismatches = check_pins(args.requirements)

    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": gpu_names,
        "required_gpu_name": args.require_gpu_name,
        "nvidia_smi_query_exit_code": code,
        "nvidia_smi_query": gpu_query,
        "pin_mismatches": pin_mismatches,
    }

    if output_dir is not None:
        (output_dir / "env_check.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )

    failures = []
    if torch.__version__ != EXPECTED_TORCH:
        failures.append(f"torch {torch.__version__} != {EXPECTED_TORCH}")
    if torch.version.cuda != EXPECTED_CUDA:
        failures.append(f"torch.version.cuda {torch.version.cuda} != {EXPECTED_CUDA}")
    if not cuda_available:
        failures.append("torch.cuda.is_available() is false")
    if args.require_gpu_name and not any(args.require_gpu_name in name for name in gpu_names):
        failures.append(f"No GPU name contains {args.require_gpu_name!r}: {gpu_names}")
    failures.extend(pin_mismatches)

    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        print("\nEnvironment check failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
