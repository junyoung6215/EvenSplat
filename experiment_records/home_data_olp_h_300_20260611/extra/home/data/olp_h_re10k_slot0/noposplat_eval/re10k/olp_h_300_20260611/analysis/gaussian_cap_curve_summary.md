# Gaussian budget cap curve summary

- cap_ratio 1.0 is the no-pruning render-time reference for this no-align cap experiment.
- delta_from_full_* is each budget minus cap_ratio=1.0 within the same model.

| Model | Cap | PSNR | dPSNR | SSIM | dSSIM | LPIPS | dLPIPS | Kept frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NoPoSplat | 1.00 | 23.612 | 0.000 | 0.777 | 0.000 | 0.181 | 0.000 | 1.000 |
| NoPoSplat | 0.75 | 16.711 | -6.901 | 0.664 | -0.113 | 0.283 | 0.102 | 0.750 |
| NoPoSplat | 0.50 | 11.393 | -12.219 | 0.456 | -0.321 | 0.433 | 0.251 | 0.500 |
| NoPoSplat | 0.25 | 7.823 | -15.790 | 0.210 | -0.567 | 0.587 | 0.406 | 0.250 |
| NoPoSplat | 0.10 | 6.154 | -17.458 | 0.078 | -0.699 | 0.671 | 0.490 | 0.100 |
| OLP-H | 1.00 | 22.557 | 0.000 | 0.731 | 0.000 | 0.205 | 0.000 | 1.000 |
| OLP-H | 0.75 | 18.035 | -4.522 | 0.648 | -0.084 | 0.278 | 0.074 | 0.750 |
| OLP-H | 0.50 | 12.028 | -10.528 | 0.439 | -0.292 | 0.429 | 0.225 | 0.500 |
| OLP-H | 0.25 | 8.106 | -14.451 | 0.201 | -0.531 | 0.576 | 0.371 | 0.250 |
| OLP-H | 0.10 | 6.324 | -16.233 | 0.075 | -0.656 | 0.659 | 0.454 | 0.100 |
