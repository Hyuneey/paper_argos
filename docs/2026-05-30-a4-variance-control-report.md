# A4 Variance Control Report — 2026-05-30

## Status
- `run_chunk_sensitivity.py` already defaults to `temperature=0.0` and `repeats=5`.
- `driver.write_metadata()` now records `max_iter` and `seed` in `metadata.json`.
- `aggregate_chunk_sensitivity.py` now supports a grouped chunk-level summary with mean / std / bootstrap 95% CI.
- The new grouped summary CSV has been generated from the current experiment artifacts.

## Generated artifacts
- Run-level summary: `results/chunk_sensitivity_summary.csv`
- Grouped CI summary: `results/chunk_sensitivity_grouped_summary.csv`

## Validation
- Full test suite passed: `unittest discover -s tests -p 'test_*.py' -v` → 18/18 passing

## Notes
- The grouped summary is keyed by dataset, selection mode, model, temperature, top_k, max_iter, seed, and chunk size.
- CI reporting is now in place for the fixed/evidence comparison workflow.
