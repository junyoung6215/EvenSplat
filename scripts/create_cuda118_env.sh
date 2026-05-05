#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-noposplat-olp-cuda118}"
FINGERPRINT_DIR="${FINGERPRINT_DIR:-outputs/olp_re10k/env_fingerprint}"
ENV_FILE="${ENV_FILE:-env/environment-cuda118.yml}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-env/requirements-cuda118.lock}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required. Install Miniconda or load it on the server." >&2
  exit 1
fi

conda env create -n "${ENV_NAME}" -f "${ENV_FILE}" || {
  echo "Conda env may already exist; updating ${ENV_NAME} instead."
  conda env update -n "${ENV_NAME}" -f "${ENV_FILE}"
}

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

python -m pip install \
  torch==2.1.2+cu118 \
  torchvision==0.16.2+cu118 \
  torchaudio==2.1.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

python -m pip install -r "${REQUIREMENTS_FILE}"

if [[ "${COMPILE_ROPE:-1}" == "1" ]]; then
  pushd src/model/encoder/backbone/croco/curope >/dev/null
  python setup.py build_ext --inplace
  popd >/dev/null
fi

python - <<'PY'
import diff_gaussian_rasterization

print("diff_gaussian_rasterization import OK")
PY

mkdir -p "${FINGERPRINT_DIR}"
git rev-parse HEAD > "${FINGERPRINT_DIR}/git_commit.txt"
python scripts/check_cuda118_env.py --requirements "${REQUIREMENTS_FILE}" --output-dir "${FINGERPRINT_DIR}"

echo "CUDA 11.8 environment is ready: ${ENV_NAME}"
echo "Fingerprint saved to ${FINGERPRINT_DIR}"
