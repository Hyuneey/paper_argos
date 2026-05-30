# Evidence vs Fixed — Gap Analysis (Eligible Series Only)

## Dataset scope
Eligible series only:
- `88cf3a776ba00e7c`
- `9ee5879409dccef9`

## Main pattern
Across the eligible-only table:
- `fixed` mean `point_f1_fixed`: **0.243394**
- `evidence` mean `point_f1_fixed`: **0.146789**
- Mean gap (`fixed - evidence`): **+0.096605**

So the evidence condition is **cheaper/faster**, but not better on the corrected metric.

## Chunk-size pattern
| chunk_size | gap on `point_f1_fixed` (fixed - evidence) | runtime gap (fixed - evidence) | detection token gap (fixed - evidence) |
|---|---:|---:|---:|
| 100 | +0.102696 | -1.56s | +926 |
| 250 | +0.176914 | +66.37s | +2037 |
| 500 | +0.048590 | -25.39s | +3656 |
| 1000 | +0.028101 | -4.19s | +7257 |
| 2500 | +0.193776 | +88.54s | +18680 |
| 5000 | +0.029552 | +148.88s | +37555 |

## Series-level pattern
- `88cf3a776ba00e7c`
  - mean gap: **-0.009784**
  - wins/losses: **2 / 4**
  - evidence occasionally wins, but not stably
- `9ee5879409dccef9`
  - mean gap: **+0.202994**
  - wins/losses: **6 / 0**
  - fixed wins across all chunk sizes

## Interpretation
- Evidence mode seems to buy **lower runtime and far fewer detection tokens**.
- But that cost reduction does **not** translate into a stable `point_f1_fixed` gain.
- The result appears **series-dependent** rather than universally helpful.

## Most likely next question
Why does evidence help on `88cf...` at a few chunks but consistently lag on `9ee5...`?
That is now the real research question.
