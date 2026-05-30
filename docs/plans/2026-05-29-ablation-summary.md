# ARGOS Chunk Sensitivity Ablation Summary

## Scope
- Main grid: datasets `07927a9a18fa19ae`, `88cf3a776ba00e7c`, `9ee5879409dccef9`
- Chunk sizes: `100`, `250`, `500`, `1000`, `2500`, `5000`
- Repeats: `1`, `2`, `3`
- Comparison available in current artifacts: `evidence` vs `fixed`

## Overall main-grid result

| Condition | point_f1_fixed (mean±std) | point_f1 (mean±std) | event_f1pa (mean±std) | runtime_sec (mean±std) |
|---|---:|---:|---:|---:|
| evidence | 0.135 ± 0.185 | 0.167 ± 0.171 | 0.072 ± 0.176 | 99.856 ± 65.742 |
| fixed | 0.213 ± 0.226 | 0.235 ± 0.220 | 0.261 ± 0.268 | 148.366 ± 142.395 |

## Per-chunk delta on main grid

| chunk_size | Δ point_f1_fixed | Δ point_f1 | Δ event_f1pa | Δ runtime_sec |
|---:|---:|---:|---:|---:|
| 100 | 0.111 | 0.074 | 0.323 | 9.858 |
| 250 | 0.176 | 0.139 | 0.119 | 52.605 |
| 500 | 0.027 | 0.017 | 0.351 | -48.665 |
| 1000 | 0.041 | 0.057 | 0.292 | 50.504 |
| 2500 | -0.021 | -0.004 | -0.196 | 154.496 |
| 5000 | 0.063 | 0.062 | 0.185 | 40.057 |

## Stress test: 9ee5879409dccef9 covbias subset

- evidence-only rows: 12
- point_f1_fixed: 0.068 ± 0.113
- point_f1: 0.117 ± 0.089
- event_f1pa: 0.006 ± 0.009
- runtime_sec: 67.309 ± 23.195

## Takeaway
- In the current artifact set, `fixed` is consistently stronger on `point_f1_fixed` than `evidence` on the main grid.
- `fixed` also comes with higher runtime on average, so the result should be presented as a quality/cost trade-off rather than a free win.
- The baseline condition is not present in the current results directory, so it remains a follow-up task before making a full baseline claim.
