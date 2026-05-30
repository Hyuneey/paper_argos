# ARGOS Research POC Execution Plan

> **For Hermes:** execute this plan in small slices and keep the evidence trail in `results/` and `docs/`.

**Goal:** Validate whether chunking + normal-reference injection improves anomaly detection, especially `point_f1_fixed`, and document where the approach helps or fails.

**Architecture:**
We will treat the current pipeline as a research POC with one primary claim and a tight ablation matrix. First lock the claim, then compare baseline vs chunking vs chunking+reference across the same datasets and seeds. Next, inspect failure cases and cost metrics so the final conclusion is not just a score delta but a falsifiable research finding.

**Tech Stack:**
- Python / repo-local venv
- Existing ARGOS experiment pipeline under `scripts/`, `agent/`, `datasets/`, `runtime/`
- CSV summaries in `results/`
- Markdown notes in `docs/`

---

## 1) Claim to Validate

**Primary claim:**
> Chunking plus normal-reference injection improves `point_f1_fixed` and/or robustness on high-density anomaly chunks, relative to the current baseline.

**Secondary claims:**
- The gain is not limited to one seed.
- The gain survives at multiple chunk sizes.
- The method remains cost-acceptable in tokens/runtime.

---

## 2) Ablation Matrix

Run the same dataset/seed/chunk-size grid under these conditions:

| Condition | Description | Purpose |
|---|---|---|
| A. Baseline | Existing pipeline, no chunk/reference change | Control |
| B. Chunking only | Chunking enabled, no normal reference | Isolate chunking effect |
| C. Chunking + normal reference | Chunking plus `reference_dfs` injection | Measure incremental effect |
| D. Chunking + normal reference + fixed eval | Same as C, but report `point_f1_fixed` explicitly | Validate the target metric |

**Minimum reporting set:**
- `train_f1`, `val_f1`, `test_f1`
- `point_f1`
- `point_f1_fixed`
- `point_f1pa`
- `event_f1pa`
- runtime
- token counts per agent
- prompt rows / rule length

---

## 3) Execution Steps

### Task 1: Lock the experiment grid
**Objective:** Define exactly which datasets, seeds, and chunk sizes will be included.

**Output:**
- A compact grid in this plan or a companion markdown note.
- The grid should be fixed before re-running anything.

**Checklist:**
- [ ] datasets chosen
- [ ] chunk sizes chosen
- [ ] repeat IDs chosen
- [ ] one sentence describing why each choice is included

---

### Task 2: Re-run or reuse summaries
**Objective:** Ensure every experiment condition has a complete summary row with the same schema.

**Files:**
- `scripts/aggregate_chunk_sensitivity.py`
- `results/chunk_sensitivity_*.csv`

**Checks:**
- No rows with missing `point_f1_fixed`.
- No incomplete runs are counted.
- Every summary file has the same column order.

---

### Task 3: Build the ablation table
**Objective:** Compare conditions on the same axes so the effect is visible at a glance.

**Table columns:**
- dataset
- chunk_size
- repeat_id
- condition
- point_f1_fixed
- point_f1
- event_f1pa
- runtime_sec
- token_count_detection
- token_count_repair
- token_count_review

**Rule:** average over repeats, but keep per-repeat values available.

---

### Task 4: Failure-case review
**Objective:** Identify where the method breaks or becomes noisy.

**Procedure:**
- Sample 5–10 failures from the weakest condition gaps.
- Group them into a small taxonomy:
  - boundary-crossing anomalies
  - short anomalies
  - over-sensitive reference injection
  - rule overfitting
  - high-density false positives

**Output:**
- One markdown section per category.
- At least one concrete example per category.

---

### Task 5: Robustness and cost
**Objective:** Show the result is stable and not too expensive.

**Report:**
- mean ± std across repeats
- sensitivity to chunk size
- token/runtime trade-off
- any point where gains flatten or reverse

**Acceptance bar:**
- If the gain only appears in one seed, call it out as inconclusive.
- If cost jumps sharply for small gain, call it a weak trade-off.

---

### Task 6: Final research note
**Objective:** Convert the result into a concise artifact that can be reused in a paper or update.

**Deliverable:**
- 1-page markdown summary
- one table
- one failure-case list
- one conclusion paragraph with a falsifiable claim

---

## 4) Immediate Next Step

Start with **Task 1** and freeze the experiment grid before adding new runs.

If the user says "go", execute the next concrete slice immediately rather than waiting for another plan revision.
