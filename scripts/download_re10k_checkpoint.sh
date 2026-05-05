#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_PATH="${CHECKPOINT_PATH:-pretrained_weights/re10k.ckpt}"
REPO_ID="${REPO_ID:-botaoye/NoPoSplat}"
FILENAME="${FILENAME:-re10k.ckpt}"

if [[ -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint already exists: ${CHECKPOINT_PATH}"
  exit 0
fi

mkdir -p "$(dirname "${CHECKPOINT_PATH}")"

python - <<'PY'
from pathlib import Path
import os
import shutil

from huggingface_hub import hf_hub_download

checkpoint_path = Path(os.environ.get("CHECKPOINT_PATH", "pretrained_weights/re10k.ckpt"))
repo_id = os.environ.get("REPO_ID", "botaoye/NoPoSplat")
filename = os.environ.get("FILENAME", "re10k.ckpt")

downloaded = Path(
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(checkpoint_path.parent),
        local_dir_use_symlinks=False,
    )
)

if downloaded.resolve() != checkpoint_path.resolve():
    shutil.copy2(downloaded, checkpoint_path)

print(f"Downloaded checkpoint: {checkpoint_path}")
PY

test -f "${CHECKPOINT_PATH}"
du -h "${CHECKPOINT_PATH}"

