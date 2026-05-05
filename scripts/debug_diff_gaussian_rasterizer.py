from __future__ import annotations

import argparse

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal diff_gaussian_rasterization native smoke test."
    )
    parser.add_argument("--num-gaussians", type=int, default=1)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")

    device = torch.device("cuda")
    torch.manual_seed(0)

    means3d = torch.randn(args.num_gaussians, 3, device=device) * 0.2
    means3d[:, 2] = 2.0 + means3d[:, 2].abs()
    means3d.requires_grad_(True)

    means2d = torch.zeros_like(means3d, requires_grad=True)
    colors_precomp = torch.full((args.num_gaussians, 3), 0.5, device=device)
    opacities = torch.full((args.num_gaussians, 1), 0.5, device=device)

    # Upper triangular covariance order: (0,0), (0,1), (0,2), (1,1), (1,2), (2,2).
    cov3d_precomp = torch.zeros(args.num_gaussians, 6, device=device)
    cov3d_precomp[:, 0] = 0.01
    cov3d_precomp[:, 3] = 0.01
    cov3d_precomp[:, 5] = 0.01

    viewmatrix = torch.eye(4, device=device)
    projmatrix = torch.eye(4, device=device)
    projmatrix_raw = torch.eye(4, device=device)

    settings = GaussianRasterizationSettings(
        image_height=args.height,
        image_width=args.width,
        tanfovx=1.0,
        tanfovy=1.0,
        bg=torch.zeros(3, device=device),
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        projmatrix_raw=projmatrix_raw,
        sh_degree=0,
        campos=torch.zeros(3, device=device),
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(settings)

    print(
        "running rasterizer",
        {
            "num_gaussians": args.num_gaussians,
            "height": args.height,
            "width": args.width,
            "backward": args.backward,
            "device": torch.cuda.get_device_name(0),
        },
        flush=True,
    )

    image, radii, depth, opacity, n_touched = rasterizer(
        means3D=means3d,
        means2D=means2d,
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacities,
        cov3D_precomp=cov3d_precomp,
        theta=None,
        rho=None,
    )
    torch.cuda.synchronize()

    print(
        "forward OK",
        {
            "image_shape": tuple(image.shape),
            "depth_shape": tuple(depth.shape),
            "radii_shape": tuple(radii.shape),
            "n_touched_shape": tuple(n_touched.shape)
            if hasattr(n_touched, "shape")
            else type(n_touched).__name__,
            "image_mean": float(image.mean().detach().cpu()),
        },
        flush=True,
    )

    if args.backward:
        image.mean().backward()
        torch.cuda.synchronize()
        print("backward OK", flush=True)


if __name__ == "__main__":
    main()
