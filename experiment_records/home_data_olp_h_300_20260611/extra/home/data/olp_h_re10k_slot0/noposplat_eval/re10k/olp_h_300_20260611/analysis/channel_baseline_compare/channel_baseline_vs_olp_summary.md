# NoPoSplat vs OLP-H channel diagnostic comparison

## Setup
- NoPoSplat baseline was evaluated with the preconditioner wrapper enabled but `mix=0.0`, `target_mix=0.0`.
- This makes the diagnostic output an identity latent at the same insertion point, without applying the Hadamard transform.
- OLP-H uses the saved OLP-H rawfull metric300 run with Hadamard `mix=1.0`.

## Scene-level means
| Metric | NoPoSplat identity | OLP-H output | Delta |
| --- | ---: | ---: | ---: |
| `latent_output_olp_score_mean` | 4.09874 | 7.88128 | +3.78254 |
| `latent_output_olp_score_p95` | 5.50869 | 10.6148 | +5.1061 |
| `latent_output_olp_score_p99` | 6.02131 | 11.2763 | +5.25495 |
| `latent_output_top1_channel_mode_freq` | 0.28145 | 0.73602 | +0.454571 |
| `latent_output_top1_value_abs_p99` | 2.37569 | 3.79472 | +1.41903 |
| `latent_output_topk_energy_ratio_mean` | 0.310256 | 0.495147 | +0.184891 |
| `latent_output_topk_energy_ratio_p95` | 0.470991 | 0.637916 | +0.166925 |
| `latent_output_topk_energy_ratio_p99` | 0.526717 | 0.673589 | +0.146871 |

## Scene-level top1 channel modes
### NoPoSplat identity
| Rank | Channel | Scenes | Ratio |
| ---: | ---: | ---: | ---: |
| 1 | 96 | 69 | 23.0% |
| 2 | 98 | 27 | 9.0% |
| 3 | 76 | 20 | 6.7% |
| 4 | 32 | 18 | 6.0% |
| 5 | 50 | 15 | 5.0% |
| 6 | 94 | 13 | 4.3% |
| 7 | 37 | 10 | 3.3% |
| 8 | 131 | 8 | 2.7% |

### OLP-H output
| Rank | Channel | Scenes | Ratio |
| ---: | ---: | ---: | ---: |
| 1 | 200 | 159 | 53.0% |
| 2 | 91 | 74 | 24.7% |
| 3 | 120 | 56 | 18.7% |
| 4 | 229 | 11 | 3.7% |

## Interpretation
- This comparison separates baseline latent channel concentration from OLP-H output channel concentration.
- If OLP-H has higher score / mode frequency / top-k energy, the concentration is amplified by the OLP-H transform or checkpoint behavior.
- If NoPoSplat is already similarly concentrated, channel outliers may be an existing latent property rather than an OLP-H-specific mechanism.
