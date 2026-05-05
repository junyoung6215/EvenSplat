from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
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


def find_paths(root: Path, pattern: str) -> list[str]:
    if not root.exists():
        return []
    return [str(path) for path in root.glob(pattern)]


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
        write_artifact(output_dir / "nvcc_version.txt", ["nvcc", "--version"])
        write_artifact(output_dir / "gcc_version.txt", ["gcc", "--version"])
        write_artifact(output_dir / "gxx_version.txt", ["g++", "--version"])

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
    nvcc_code, nvcc_output = run_command(["nvcc", "--version"])
    gcc_code, gcc_output = run_command(["gcc", "--version"])
    gxx_code, gxx_output = run_command(["g++", "--version"])
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    cuda_type_traits = (
        Path(conda_prefix) / "include" / "cuda" / "std" / "type_traits"
        if conda_prefix
        else Path()
    )
    dgr_import_ok = True
    dgr_import_error = ""
    dgr_extension_path = ""
    dgr_ldd_exit_code = 0
    dgr_ldd_output = ""
    try:
        import diff_gaussian_rasterization  # noqa: F401
        import diff_gaussian_rasterization._C as dgr_extension

        dgr_extension_path = dgr_extension.__file__
        dgr_ldd_exit_code, dgr_ldd_output = run_command(["ldd", dgr_extension_path])
    except Exception as exc:  # pragma: no cover - diagnostic path
        dgr_import_ok = False
        dgr_import_error = repr(exc)

    cv2_import_ok = True
    cv2_import_error = ""
    try:
        import cv2  # noqa: F401
    except Exception as exc:  # pragma: no cover - diagnostic path
        cv2_import_ok = False
        cv2_import_error = repr(exc)

    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "conda_prefix": conda_prefix,
        "cuda_home": os.environ.get("CUDA_HOME", ""),
        "cuda_path": os.environ.get("CUDA_PATH", ""),
        "cc": os.environ.get("CC", ""),
        "cxx": os.environ.get("CXX", ""),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": gpu_names,
        "required_gpu_name": args.require_gpu_name,
        "nvcc_exit_code": nvcc_code,
        "nvcc_version": nvcc_output,
        "gcc_exit_code": gcc_code,
        "gcc_version": gcc_output.splitlines()[0] if gcc_output else "",
        "gxx_exit_code": gxx_code,
        "gxx_version": gxx_output.splitlines()[0] if gxx_output else "",
        "cuda_type_traits_path": str(cuda_type_traits),
        "cuda_type_traits_exists": cuda_type_traits.exists(),
        "diff_gaussian_rasterization_import_ok": dgr_import_ok,
        "diff_gaussian_rasterization_import_error": dgr_import_error,
        "diff_gaussian_rasterization_extension_path": dgr_extension_path,
        "diff_gaussian_rasterization_ldd_exit_code": dgr_ldd_exit_code,
        "diff_gaussian_rasterization_ldd": dgr_ldd_output,
        "cv2_import_ok": cv2_import_ok,
        "cv2_import_error": cv2_import_error,
        "conda_lib_cudart12": find_paths(Path(conda_prefix) / "lib", "libcudart.so.12*")
        if conda_prefix
        else [],
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
    if nvcc_code != 0:
        failures.append("nvcc --version failed")
    if "release 11.8" not in nvcc_output:
        failures.append("nvcc is not CUDA 11.8")
    if not cuda_type_traits.exists():
        failures.append(f"CUDA C++ header missing: {cuda_type_traits}")
    if not dgr_import_ok:
        failures.append(f"diff_gaussian_rasterization import failed: {dgr_import_error}")
    if dgr_import_ok and dgr_ldd_exit_code != 0:
        failures.append("ldd failed for diff_gaussian_rasterization extension")
    if dgr_import_ok and "not found" in dgr_ldd_output:
        failures.append("diff_gaussian_rasterization extension has unresolved shared libraries")
    if dgr_import_ok and "libcudart.so.12" in dgr_ldd_output:
        failures.append("diff_gaussian_rasterization links against CUDA 12 runtime")
    if conda_prefix and find_paths(Path(conda_prefix) / "lib", "libcudart.so.12*"):
        failures.append("CONDA_PREFIX/lib contains CUDA 12 libcudart; use a clean CUDA 11.8 env")
    if not cv2_import_ok:
        failures.append(f"cv2 import failed: {cv2_import_error}")
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
