# Group A Gate Check — 2026-05-30

## Status
- **A1/A4/A6 bookkeeping is reflected in code and tests pass.**
- Current summary rows show `point_f1_fixed` is present for every ablation row.
- The fixed/evidence split is already visible in the ablation table.

## Verified from `results/chunk_sensitivity_ablation_table.csv`
- Rows checked: **36**
- Missing `point_f1_fixed`: **0**
- Mean `point_f1_fixed`:
  - fixed: **0.213448**
  - evidence: **0.146039**

## A3 recovery from raw KPI CSVs

Recomputed from `results/datasets/kpi_preliminary/*.csv` using the repo's `train_test_split=0.7` logic:

| series | test anomaly points | test anomaly events | eligible (>=5 events) |
|---|---:|---:|---|
| `07927a9a18fa19ae` | 24 | 2 | no |
| `88cf3a776ba00e7c` | 1154 | 14 | yes |
| `9ee5879409dccef9` | 799 | 14 | yes |

## Gate interpretation
- **A3 is only partially satisfied.**
- The two larger KPI series pass the event-sufficiency threshold.
- `07927a9a18fa19ae` is **not eligible** for per-series F1 claims because its test split contains only 2 anomaly events.
- Therefore the proper next step is to **exclude ineligible series from the main claim table** or rerun a broader grid that satisfies the sufficiency rule.

## Immediate next slice
1. Restrict the main table to eligible series only (`88cf3a776ba00e7c`, `9ee5879409dccef9`).
2. Recompute the ablation table / summary on the eligible set.
3. Then freeze the fixed-chunk baseline with `temperature=0.0` and `repeat>=5`.
