# ARGOS evidence-aware segment selection pilot summary

## Scope

- Dataset family: KPI
- Prepared series evaluated:
  - `07927a9a18fa19ae`
  - `88cf3a776ba00e7c`
  - `9ee5879409dccef9`
- Common run settings:
  - `llm_provider=chatgpt-oauth`
  - requested `llm_engine=gpt-4-mini`
  - resolved Codex OAuth model: `gpt-5.4-mini`
  - `top_k=1`
  - `max_iter=1`
  - `repeats=3`
  - chunk sizes: `100, 250, 500, 1000, 2500, 5000`

## Main result

The fixed-chunk baseline is highly sensitive to chunk size. The evidence-aware
selector can improve both test F1 and cost, but the current default weighting
does not generalize across all series.

## Best test F1 by series

| series | fixed best chunk | fixed best test F1 | evidence best chunk | evidence best test F1 | delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `07927a9a18fa19ae` | 5000 | 0.3321 | 2500 | 0.5294 | +0.1973 |
| `88cf3a776ba00e7c` | 5000 | 0.3943 | 5000 | 0.6333 | +0.2390 |
| `9ee5879409dccef9` | 5000 | 0.4055 | 500 | 0.2335 | -0.1720 |

Average over per-series best settings:

- fixed best test F1: `0.3773`
- evidence best test F1: `0.4654`

## Cost at each method's best setting

| series | fixed tokens | evidence tokens | fixed prompt rows | evidence prompt rows | fixed runtime sec | evidence runtime sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `07927a9a18fa19ae` | 14192.7 | 8239.3 | 5000 | 625 | 71.3 | 118.5 |
| `88cf3a776ba00e7c` | 51983.7 | 14284.7 | 5000 | 1250 | 424.2 | 161.6 |
| `9ee5879409dccef9` | 51720.3 | 2842.3 | 5000 | 125 | 171.9 | 98.3 |

Average over per-series best settings:

- fixed detection tokens: `39298.9`
- evidence detection tokens: `8455.4`

## Interpretation

1. `chunk_size` matters materially for ARGOS rule induction.
2. The evidence-aware selector is promising: on 2 of 3 evaluated series it beat
   the best fixed baseline while also using fewer prompt rows and fewer tokens.
3. The current selector is not robust. On `9ee5879409dccef9`, the selector
   repeatedly chose dense anomaly-centered windows with limited anomaly coverage
   and little normal context, which appears to hurt generalization.
4. This means the current result supports a paper claim about:
   - fixed chunk instability
   - the potential value of evidence-aware segment selection
   - the need for better selector objectives than anomaly density alone

## Current blocker for stronger claims

These runs are still pilot-scale:

- only 3 KPI series
- `top_k=1`
- `max_iter=1`
- model path is Codex OAuth `gpt-5.4-mini`, not a stable public API baseline

The current default selector can be presented as a proof of concept, but not as
an established improvement over fixed chunking without a broader sweep or a
better-tuned selector objective.

## Result files

- `results/chunk_sensitivity_07927_fixed_summary.csv`
- `results/chunk_sensitivity_07927_evidence_summary.csv`
- `results/chunk_sensitivity_88cf_fixed_summary.csv`
- `results/chunk_sensitivity_88cf_evidence_summary.csv`
- `results/chunk_sensitivity_9ee5_fixed_summary.csv`
- `results/chunk_sensitivity_9ee5_evidence_summary.csv`
