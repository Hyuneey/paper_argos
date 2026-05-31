# ARGOS Final Decision

**Conclusion:** GO-reframe

## Leg summary
- **Performance**: FAIL — evidence mean point_f1_fixed=0.129812 vs fixed mean point_f1_fixed=0.260201 (Δ=-0.130389).
- **Cost**: PASS — tokens fixed=15362.018519 evidence=6406.518519 (Δ=8955.500000); prompt rows fixed=1558.333333 evidence=458.944444 (Δ=1099.388889).
- **Traceability**: PASS — heldout_support_gap fixed=0.171302 evidence=0.222333 (Δ=0.051032).

## Inputs
- Performance CSV: `results/c1_comparison_suite_summary.csv` (8 rows)
- Cost CSV: `results/chunk_sensitivity_fixed_evidence_summary.csv` (108 rows)
- Consistency CSV: `results/c2_consistency_summary.csv` (4 rows)
- Morphology CSV: `results/c3_morphology_diagnostic.csv` (26 rows)
- Commit hash: `a8cb059`

## Representative values
- Performance: evidence mean point_f1_fixed=0.129812, fixed mean point_f1_fixed=0.260201.
- Cost: evidence mean tokens=6406.518519, fixed mean tokens=15362.018519.
- Traceability: evidence mean heldout_support_gap=0.222333, fixed mean heldout_support_gap=0.171302.
- Morphology: most common label is `mixed` across 26 series.

## Recommended next step
- Tighten the evidence condition around the series where the performance leg underperforms, then rerun the final comparison suite.
