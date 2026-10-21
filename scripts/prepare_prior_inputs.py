"""Export the exact NoPo source grid on the server; does not run a teacher.

Run as: python -m scripts.prepare_prior_inputs --help
Uses the existing dataset/crop code, so the NoPo CPU dependencies must be present.
Only trusted local RE10K .torch chunks should be supplied.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.foundation.prior import PriorIdentity, image_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-id", required=True)
    parser.add_argument("--teacher-settings-id", required=True)
    args = parser.parse_args()
    # Lazy import permits --help without importing the full dataset dependency tree.
    from src.dataset.dataset_re10k import DatasetRE10k, DatasetRE10kCfg
    from src.dataset.view_sampler.view_sampler_fixed_pairs import (
        ViewSamplerFixedPairs, ViewSamplerFixedPairsCfg)

    sampler_cfg = ViewSamplerFixedPairsCfg("fixed_pairs", args.pairs,
                                          Path("assets/evaluation_index_re10k.json"))
    cfg = DatasetRE10kCfg(
        original_image_shape=[360, 640], input_image_shape=[256, 256],
        background_color=[0., 0., 0.], cameras_are_circular=False,
        overfit_to_scene=None, view_sampler=sampler_cfg, name="re10k",
        roots=[args.data_root], baseline_min=1e-3, baseline_max=1e10,
        max_fov=100., make_baseline_1=True, augment=False, relative_pose=True,
        skip_bad_shape=True)
    sampler = ViewSamplerFixedPairs(sampler_cfg, "train", False, False, None)
    dataset = DatasetRE10k(cfg, "train", sampler)
    args.output.mkdir(parents=True, exist_ok=True)
    count = 0
    for example in dataset:
        images = example["context"]["image"]
        identity = PriorIdentity(example["scene"], tuple(example["context"]["index"].tolist()),
                                 tuple(images.shape[-2:]), args.teacher_id, args.teacher_settings_id)
        metadata = {**identity.metadata(), "image_sha256": image_fingerprint(images)}
        with (args.output / f"{identity.key}.npz").open("xb") as handle:
            np.savez_compressed(handle, images=images.mul(255).round().byte().numpy(),
                                metadata=np.array(json.dumps(metadata, sort_keys=True)))
        count += 1
    print(f"Exported {count} ordered training pairs (RGB uint8 [2,3,256,256]); no teacher executed.")


if __name__ == "__main__":
    main()
