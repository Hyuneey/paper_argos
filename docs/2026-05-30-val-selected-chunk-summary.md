# A5 Val-Selected Chunk Summary — 2026-05-30

## Rule
- For each eligible series, select the chunk size with the highest **mean val_f1** across repeats.
- Test metrics are reported **once after selection**; they are not used to choose the chunk.

## Eligible series
- `88cf3a776ba00e7c`
- `9ee5879409dccef9`

## Selected settings by series

| series | mode | selected chunk | mean val_f1 | mean test_f1 | mean point_f1_fixed | mean runtime sec | mean detection tokens | mean prompt rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `88cf3a776ba00e7c` | evidence | 5000 | 0.256892 | 0.633299 | 0.606643 | 161.6 | 14284.7 | 1250.0 |
| `88cf3a776ba00e7c` | fixed | 2500 | 0.278239 | 0.306218 | 0.291362 | 217.6 | 26731.0 | 2500.0 |
| `9ee5879409dccef9` | evidence | 1000 | 0.227371 | 0.204453 | 0.203324 | 74.2 | 4231.2 | 250.0 |
| `9ee5879409dccef9` | fixed | 2500 | 0.313799 | 0.402822 | 0.374451 | 279.4 | 26649.3 | 2500.0 |

## Series-averaged selected settings

| mode | mean val_f1 | mean test_f1 | mean point_f1_fixed | mean runtime sec | mean detection tokens | mean prompt rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | 0.296019 | 0.354520 | 0.332907 | 248.5 | 26690.2 | 2500.0 |
| evidence | 0.242131 | 0.418876 | 0.404983 | 117.9 | 9257.9 | 750.0 |

## Interpretation
- This is the leakage-safe presentation: chunk choice is locked by validation, then test is shown once.
- On the eligible set, evidence keeps the efficiency advantage and also improves test F1 in the selected configuration.
- The main claim is therefore about **val-selected operating points**, not test-picked best chunks.

## Artifact
- `results/chunk_sensitivity_val_selected.csv`