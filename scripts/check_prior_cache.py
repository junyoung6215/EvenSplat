"""Check every exported input against a cache, or create diagnostic mock caches.

No model execution, network access, or overwriting. Real adapters should call
write_prior_cache(..., kind='teacher') with calibrated, source-aligned labels.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.foundation.prior import (PriorIdentity, image_fingerprint, mock_prior,
                                  read_prior_cache, write_prior_cache)


def check_or_make(inputs: Path, cache: Path, *, make_mock=False, allow_mock=False):
    paths = sorted(inputs.glob("*.npz"))
    if not paths:
        raise ValueError("No exported prior inputs found.")
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"].item()))
            images = torch.from_numpy(payload["images"].copy()).float() / 255
        fingerprint = metadata.pop("image_sha256")
        metadata["context"] = tuple(metadata["context"])
        metadata["image_shape"] = tuple(metadata["image_shape"])
        identity = PriorIdentity(**metadata)
        if identity.key != path.stem or image_fingerprint(images) != fingerprint:
            raise ValueError(f"Exported input identity/hash mismatch: {path}")
        if make_mock:
            write_prior_cache(cache, identity, images, mock_prior((2, *identity.image_shape)), kind="mock")
        read_prior_cache(cache, identity, images, allow_mock=allow_mock or make_mock)
    return len(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--make-mock", action="store_true")
    parser.add_argument("--allow-mock", action="store_true")
    args = parser.parse_args()
    count = check_or_make(args.inputs, args.cache, make_mock=args.make_mock, allow_mock=args.allow_mock)
    print(f"Validated {count} pairs. Mock accepted: {args.allow_mock or args.make_mock}.")


if __name__ == "__main__":
    main()
