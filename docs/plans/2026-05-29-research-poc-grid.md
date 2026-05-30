# ARGOS Research POC Grid Freeze

**Status:** frozen for the next execution slice

## Primary grid

These are the datasets and settings to use for the main ablation table:

- **Datasets**: `07927a9a18fa19ae`, `88cf3a776ba00e7c`, `9ee5879409dccef9`
- **Chunk sizes**: `100`, `250`, `500`, `1000`, `2500`, `5000`
- **Repeats**: `1`, `2`, `3`

**Scale:** 54 runs per condition

## Auxiliary stress test

Keep this separate from the main claim table:

- **Dataset**: `9ee5879409dccef9`
- **Chunk sizes available**: `100`, `250`, `500`, `1000`
- **Repeats**: `1`, `2`, `3`

**Scale:** 12 runs

## Exclusions for the main table

- `chunk_sensitivity_gpt4mini_oauth` is treated as a separate plumbing check, not part of the research claim.
- Incomplete runs with missing test evals are excluded from summary tables.

## Research claims to report

1. Whether `chunking + normal-reference injection` improves `point_f1_fixed`.
2. Whether the gain persists across chunk sizes and repeats.
3. Whether token/runtime cost remains acceptable.

## Next execution slice

Build the ablation table from the frozen grid and compare:
- baseline
- chunking only
- chunking + normal reference
- chunking + normal reference + fixed eval
