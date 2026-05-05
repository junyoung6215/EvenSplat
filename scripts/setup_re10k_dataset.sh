#!/usr/bin/env bash
set -euo pipefail

# Prepare the processed RealEstate10K dataset expected by NoPoSplat.
#
# This script does not download raw RealEstate10K videos. NoPoSplat expects the
# processed pixelSplat/MVSplat-style torch chunk format:
#
#   datasets/re10k/train/*.torch
#   datasets/re10k/train/index.json
#   datasets/re10k/test/*.torch
#   datasets/re10k/test/index.json
#
# Supported modes:
#   1. Link an existing processed dataset:
#      RE10K_SOURCE_DIR=/path/to/re10k bash scripts/setup_re10k_dataset.sh
#
#   2. Download a direct archive URL (.zip, .tar, .tar.gz, .tgz):
#      RE10K_DATA_URL=https://.../re10k.zip bash scripts/setup_re10k_dataset.sh
#
#   3. Verify an already prepared datasets/re10k directory:
#      bash scripts/setup_re10k_dataset.sh

TARGET_DIR="${TARGET_DIR:-datasets/re10k}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-downloads/re10k}"
RE10K_SOURCE_DIR="${RE10K_SOURCE_DIR:-}"
RE10K_DATA_URL="${RE10K_DATA_URL:-}"
RE10K_ARCHIVE_PATH="${RE10K_ARCHIVE_PATH:-}"
FORCE="${FORCE:-0}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

verify_re10k() {
  local root="$1"
  local missing=0
  local first_train_chunk=""
  local first_test_chunk=""

  for path in \
    "${root}/train/index.json" \
    "${root}/test/index.json"; do
    if [[ ! -f "${path}" ]]; then
      echo "Missing: ${path}" >&2
      missing=1
    fi
  done

  # Some download paths, including Hugging Face local-dir workflows, may expose
  # chunk files as symlinks to a cache. Treat symlinked chunks as valid.
  #
  # Use -print -quit instead of piping through head. With `set -o pipefail`,
  # `find | head` can be reported as failed when find receives SIGPIPE after
  # head exits, which incorrectly marks valid large datasets as missing.
  first_train_chunk="$(find -L "${root}/train" -maxdepth 1 -name "*.torch" -type f -print -quit 2>/dev/null || true)"
  if [[ -z "${first_train_chunk}" ]]; then
    echo "Missing train torch chunks: ${root}/train/*.torch" >&2
    missing=1
  fi

  first_test_chunk="$(find -L "${root}/test" -maxdepth 1 -name "*.torch" -type f -print -quit 2>/dev/null || true)"
  if [[ -z "${first_test_chunk}" ]]; then
    echo "Missing test torch chunks: ${root}/test/*.torch" >&2
    missing=1
  fi

  if [[ "${missing}" != "0" ]]; then
    return 1
  fi

  echo "RE10K dataset OK: ${root}"
  du -sh "${root}" || true
}

find_re10k_root() {
  local base="$1"

  if verify_re10k "${base}" >/dev/null 2>&1; then
    echo "${base}"
    return 0
  fi

  local candidate
  candidate="$(find "${base}" -maxdepth 4 -type f -path "*/train/index.json" 2>/dev/null | head -1 || true)"
  if [[ -n "${candidate}" ]]; then
    candidate="$(dirname "$(dirname "${candidate}")")"
    if verify_re10k "${candidate}" >/dev/null 2>&1; then
      echo "${candidate}"
      return 0
    fi
  fi

  return 1
}

prepare_target_parent() {
  mkdir -p "$(dirname "${TARGET_DIR}")"
  if [[ -e "${TARGET_DIR}" || -L "${TARGET_DIR}" ]]; then
    if [[ "${FORCE}" == "1" ]]; then
      rm -rf "${TARGET_DIR}"
    else
      verify_re10k "${TARGET_DIR}" || die "${TARGET_DIR} exists but is not a valid processed RE10K dataset. Set FORCE=1 to replace it."
      exit 0
    fi
  fi
}

archive_name_from_url() {
  local url="$1"
  local name
  name="$(basename "${url%%\?*}")"
  [[ -n "${name}" ]] || name="re10k_dataset"
  echo "${name}"
}

download_archive() {
  local url="$1"
  mkdir -p "${DOWNLOAD_DIR}"
  local archive="${DOWNLOAD_DIR}/$(archive_name_from_url "${url}")"

  if [[ ! -f "${archive}" ]]; then
    echo "Downloading ${url}"
    if command -v curl >/dev/null 2>&1; then
      curl -L --fail --retry 3 -o "${archive}" "${url}"
    elif command -v wget >/dev/null 2>&1; then
      wget -O "${archive}" "${url}"
    else
      die "curl or wget is required to download RE10K."
    fi
  else
    echo "Archive already exists: ${archive}"
  fi

  echo "${archive}"
}

extract_archive() {
  local archive="$1"
  local extract_dir="${DOWNLOAD_DIR}/extracted"
  rm -rf "${extract_dir}"
  mkdir -p "${extract_dir}"

  case "${archive}" in
    *.zip)
      command -v unzip >/dev/null 2>&1 || die "unzip is required for ${archive}"
      unzip -q "${archive}" -d "${extract_dir}"
      ;;
    *.tar|*.tar.gz|*.tgz)
      tar -xf "${archive}" -C "${extract_dir}"
      ;;
    *)
      die "Unsupported archive type: ${archive}. Use .zip, .tar, .tar.gz, or .tgz."
      ;;
  esac

  echo "${extract_dir}"
}

prepare_target_parent

if [[ -n "${RE10K_SOURCE_DIR}" ]]; then
  [[ -d "${RE10K_SOURCE_DIR}" ]] || die "RE10K_SOURCE_DIR does not exist: ${RE10K_SOURCE_DIR}"
  verify_re10k "${RE10K_SOURCE_DIR}" || die "RE10K_SOURCE_DIR is not a processed RE10K chunk dataset."
  ln -s "$(realpath "${RE10K_SOURCE_DIR}")" "${TARGET_DIR}"
  verify_re10k "${TARGET_DIR}"
  exit 0
fi

if [[ -n "${RE10K_DATA_URL}" ]]; then
  RE10K_ARCHIVE_PATH="$(download_archive "${RE10K_DATA_URL}")"
fi

if [[ -n "${RE10K_ARCHIVE_PATH}" ]]; then
  [[ -f "${RE10K_ARCHIVE_PATH}" ]] || die "RE10K_ARCHIVE_PATH does not exist: ${RE10K_ARCHIVE_PATH}"
  extracted_dir="$(extract_archive "${RE10K_ARCHIVE_PATH}")"
  re10k_root="$(find_re10k_root "${extracted_dir}")" || die "Archive did not contain a valid processed RE10K dataset."
  mv "${re10k_root}" "${TARGET_DIR}"
  verify_re10k "${TARGET_DIR}"
  exit 0
fi

verify_re10k "${TARGET_DIR}" || cat >&2 <<'EOF'
No processed RE10K dataset was found.

Provide one of:
  RE10K_SOURCE_DIR=/path/to/processed/re10k
  RE10K_DATA_URL=https://.../processed_re10k.zip
  RE10K_ARCHIVE_PATH=/path/to/processed_re10k.zip

Expected structure:
  datasets/re10k/train/index.json
  datasets/re10k/train/*.torch
  datasets/re10k/test/index.json
  datasets/re10k/test/*.torch

NoPoSplat follows the processed pixelSplat/MVSplat chunk format, not raw
RealEstate10K videos.
EOF

exit 1
