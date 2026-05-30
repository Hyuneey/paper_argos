# ARGOS Research POC — Final Note (2026-05-30)

## Executive summary
We validated the corrected measurement path and then checked the KPI series for event sufficiency.
The main takeaway is **not** that chunking + normal reference wins overall.
On the eligible KPI series, the corrected metric shows **fixed chunking still outperforms the evidence condition on average** for `point_f1_fixed`.

## What was verified
- `point_f1_fixed` is now flowing into the ablation summaries.
- Runtime config tests pass.
- `point_f1_fixed` is present for every ablation row in the current summary table.
- A3 event-sufficiency was recomputed directly from the KPI CSVs.

## A3 sufficiency check
Using the repo's `train_test_split=0.7` split:

| series | test anomaly points | test anomaly events | eligible |
|---|---:|---:|---|
| `07927a9a18fa19ae` | 24 | 2 | no |
| `88cf3a776ba00e7c` | 1154 | 14 | yes |
| `9ee5879409dccef9` | 799 | 14 | yes |

`07927a9a18fa19ae` is excluded from per-series claims because its test split has only 2 anomaly events.

## Eligible-only ablation result
Eligible series: `88cf3a776ba00e7c`, `9ee5879409dccef9`

| condition | mean `point_f1_fixed` |
|---|---:|
| fixed | 0.243394 |
| evidence | 0.146789 |

## Interpretation
- The corrected metric is working.
- The main claim is **not supported** by the eligible-only aggregate.
- Evidence mode has some isolated wins at specific chunk sizes, but the aggregate result still favors fixed.
- Therefore the strongest defensible conclusion right now is:

> After measurement repair and series-sufficiency filtering, the current evidence does **not** show a stable overall improvement from chunking + normal-reference injection on `point_f1_fixed`.

## Artifacts
- `docs/2026-05-30-group-a-gate-check.md`
- `docs/2026-05-30-eligible-only-ablation-summary.md`
- `results/chunk_sensitivity_ablation_table_eligible.csv`

## Recommended next step
If we continue, the next useful move is to inspect **why** evidence helps or hurts by chunk size and series, not to expand the grid blindly.
