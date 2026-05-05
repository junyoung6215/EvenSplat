#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-noposplat-olp-cuda118}"
FINGERPRINT_DIR="${FINGERPRINT_DIR:-outputs/olp_re10k/env_fingerprint}"
ENV_FILE="${ENV_FILE:-env/environment-cuda118.yml}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-env/requirements-cuda118.lock}"
DGR_REPO="${DGR_REPO:-https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose.git}"
DGR_COMMIT="${DGR_COMMIT:-43e21bff91cd24986ee3dd52fe0bb06952e50ec7}"
INSTALL_SYSTEM_BUILD_DEPS="${INSTALL_SYSTEM_BUILD_DEPS:-1}"
COMPILE_ROPE="${COMPILE_ROPE:-1}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
MAX_JOBS="${MAX_JOBS:-4}"
CUDA_CONDA_CHANNEL="${CUDA_CONDA_CHANNEL:-nvidia/label/cuda-11.8.0}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required. Install Miniconda or load it on the server." >&2
  exit 1
fi

if [[ "${INSTALL_SYSTEM_BUILD_DEPS}" == "1" ]]; then
  if ! command -v gcc-11 >/dev/null 2>&1 || ! command -v g++-11 >/dev/null 2>&1; then
    if [[ "${EUID}" -eq 0 && -x "$(command -v apt-get)" ]]; then
      apt-get update
      apt-get install -y build-essential gcc-11 g++-11 libgl1 libglib2.0-0
    else
      echo "gcc-11/g++-11 are required for CUDA 11.8 extension builds." >&2
      echo "Install them or rerun with CC/CXX pointing to a supported compiler." >&2
      exit 1
    fi
  fi
fi

conda env create -n "${ENV_NAME}" -f "${ENV_FILE}" || {
  echo "Conda env may already exist; updating ${ENV_NAME} instead."
  conda env update -n "${ENV_NAME}" -f "${ENV_FILE}"
}

eval "$(conda shell.bash hook)"
# Some CUDA conda activation hooks reference unset variables under `set -u`.
set +u
conda activate "${ENV_NAME}"
set -u

conda install -y -c "${CUDA_CONDA_CHANNEL}" \
  cuda-version=11.8 \
  cuda-nvcc=11.8 \
  cuda-cudart-dev=11.8 \
  cuda-cccl=11.8

export CUDA_HOME="${CONDA_PREFIX}"
export CUDA_PATH="${CONDA_PREFIX}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
if command -v gcc-11 >/dev/null 2>&1 && command -v g++-11 >/dev/null 2>&1; then
  export CC="${CC:-$(command -v gcc-11)}"
  export CXX="${CXX:-$(command -v g++-11)}"
fi
export TORCH_CUDA_ARCH_LIST
export MAX_JOBS

python -m pip install --upgrade pip
python -m pip install --force-reinstall "setuptools==69.5.1" wheel ninja packaging
python -m pip install \
  torch==2.1.2+cu118 \
  torchvision==0.16.2+cu118 \
  torchaudio==2.1.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

python - <<'PY'
import torch

print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
PY

test -f "${CONDA_PREFIX}/include/cuda/std/type_traits"

python -m pip install -v --no-build-isolation --no-cache-dir \
  "diff_gaussian_rasterization @ git+${DGR_REPO}@${DGR_COMMIT}"

python - <<'PY'
import diff_gaussian_rasterization

print("diff_gaussian_rasterization import OK")
PY

FILTERED_REQUIREMENTS="$(mktemp)"
grep -vE "^(torch==|torchvision==|torchaudio==|diff_gaussian_rasterization @)" \
  "${REQUIREMENTS_FILE}" > "${FILTERED_REQUIREMENTS}"
python -m pip install --no-deps -r "${FILTERED_REQUIREMENTS}"
rm -f "${FILTERED_REQUIREMENTS}"

if [[ "${COMPILE_ROPE}" == "1" ]]; then
  pushd src/model/encoder/backbone/croco/curope >/dev/null
  python setup.py build_ext --inplace
  popd >/dev/null
fi

mkdir -p "${FINGERPRINT_DIR}"
git rev-parse HEAD > "${FINGERPRINT_DIR}/git_commit.txt"
python scripts/check_cuda118_env.py --requirements "${REQUIREMENTS_FILE}" --output-dir "${FINGERPRINT_DIR}"

echo "CUDA 11.8 environment is ready: ${ENV_NAME}"
echo "Fingerprint saved to ${FINGERPRINT_DIR}"
