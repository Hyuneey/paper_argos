# paper_argos Final Decision Execution Plan

> **For Hermes:** Use this plan task-by-task. Do not declare completion from code/test success alone; only commit-backed result files count.

**Goal:** 측정 정정 → comparison → traceability leg까지 실제 실행하고, 사전 등록 기준으로 GO / GO-reframe / NO-GO 결론을 도출한다.

**Architecture:**
- 먼저 환경 정직성 및 실행 가능 여부를 확정하고, 모델/temperature를 고정한다.
- 그다음 corrected measurement로 헤드라인 숫자를 재생성하고, comparison suite로 조건별 비용/성능을 비교한다.
- 마지막으로 held-out 기반 consistency evaluator를 추가해 traceability leg를 수치화하고, 최종 decision 문서로 결론을 고정한다.

**Tech Stack:** Python, `./.venv/bin/python`, `unittest`, repo-local scripts, CSV artifacts, Git commits

---

## Current-state snapshot

### Complete
- fixed-decision F1 metric is implemented.
- reference injection is present in the detection/review flow.
- provenance logging and held-out window pool scaffolding exist.
- comparison suite scripts already exist.
- report files are numbered for chronological ordering.

### Partial
- comparison/summary CSVs already exist for previous work, but not for the new final-decision flow.
- provenance and held-out scaffolding are present, but the traceability leg is not yet measured.

### Blocked / not yet done
- `consistency evaluator` does not exist yet.
- `heldout_support_gap` does not exist yet.
- no fresh end-to-end final-decision run has been executed for this brief.
- Phase 3/4/5 result CSVs and `reports/DECISION.md` do not exist yet.

---

## Phase 0 — Execution honesty and environment lock

**Gate 0:** confirm the agent can actually run LLM-backed experiments, and pin the model/temperature before proceeding.

### Task 0.1: Verify execution capability
**Objective:** confirm whether the current environment can perform actual LLM calls and produce run artifacts.

**Files:**
- Inspect: `scripts/run_chunk_sensitivity.py`
- Inspect: `scripts/run_comparison_suite.py`
- Inspect: `scripts/aggregate_comparison_suite.py`

**Checklist:**
- verify actual execution path, not just code presence
- verify required auth/model access
- record whether execution is possible in this environment

**Gate condition:** if actual execution is not possible here, stop and report who/where will execute the runs.

### Task 0.2: Pin model and runtime settings
**Objective:** fix the model and sampling parameters so the run is reproducible.

**Files:**
- Modify: run commands / experiment config used for Phase 1–3

**Required settings:**
- `temperature=0.0`
- pinned model snapshot or stable open model
- no oauth-dynamic ambiguity

### Task 0.3: Define execution proof
**Objective:** make the acceptance criteria for “run happened” explicit.

**Proof requires:**
- committed CSV artifact
- non-zero `token_count_detection_mean`
- non-zero `runtime_sec_mean`
- repeat-to-repeat variation showing actual LLM calls
- git commit hash

---

## Phase 1 — Corrected re-run for performance/cost

**Goal:** regenerate the headline metrics under corrected measurement.

**Gate 1:** the rerun must satisfy all of the following in the committed artifact:
- `temperature == 0.0`
- evidence rows with `*_anomaly_density == 1.0` are zero
- eligible series count is at least 6
- `n_repeats >= 5`

### Task 1.1: Enforce floor exclusion
**Objective:** hard-exclude density-1.0 candidates and remove the old tie-break behavior.

**Files:**
- Inspect/modify: `segment_selection/selector.py`
- Inspect/modify: `segment_selection/utility_scorer.py`
- Inspect/modify: `configs/segment_selector_default.yaml`

**Verification:**
- density-1.0 candidates are excluded before selection
- no density-first tie-break remains for this path

### Task 1.2: Re-run corrected chunk sensitivity
**Objective:** run the corrected configuration with reference injection on.

**Files:**
- Run: `scripts/run_chunk_sensitivity.py`

**Required settings:**
- `temperature=0.0`
- `repeats>=5`
- `top_k>=5`
- reference injection active

**Expected artifact:**
- grouped summary CSV with corrected metrics and cost columns

### Task 1.3: Expand eligible series if needed
**Objective:** reach at least 6 eligible series using the dataset preparation workflow.

**Files:**
- Inspect/modify: `scripts/prepare_kpi_dataset.py`
- Inspect: dataset eligibility logic used by the current pipeline

**Gate condition:** if eligible series remain below 6, do not proceed to Phase 2.

### Task 1.4: Fix affiliation precision bug if still present
**Objective:** ensure the corrected metric is consistent with the intended unit.

**Files:**
- Inspect/modify: `eval_metrics/affiliation_f1.py`
- Inspect/modify: connected evaluation call sites

**Expected behavior:**
- precision is computed with the corrected point-based denominator/numerator definition

### Task 1.5: Summarize corrected results
**Objective:** produce a clean summary for performance and cost.

**Expected headline columns:**
- `point_f1_fixed`
- `affiliation_f1`
- `token_count_detection_mean`
- `prompt_rows_mean`
- mean ± 95% CI

**Artifacts:**
- `results/chunk_sensitivity_grouped_summary.csv`

---

## Phase 2 — Comparison suite execution

**Goal:** compare the five windowing strategies under a single protocol.

**Gate 2:** the following files must exist and contain actual values:
- `results/c1_comparison_suite_rows.csv`
- `results/c1_comparison_suite_summary.csv`
- `results/c1_comparison_suite_best_fixed_selected.csv`

### Task 2.1: Run all comparison conditions
**Objective:** execute the comparison suite for the five strategies.

**Files:**
- Run: `scripts/run_comparison_suite.py`

**Conditions:**
- `fixed`
- `best-fixed(val)`
- `random-window(same length)`
- `anomaly-centered(density-only)`
- `event-bounded + matched reference`

### Task 2.2: Aggregate suite outputs
**Objective:** produce per-run rows, summary, and best-fixed selection output.

**Files:**
- Run: `scripts/aggregate_comparison_suite.py`

**Expected outputs:**
- `results/c1_comparison_suite_rows.csv`
- `results/c1_comparison_suite_summary.csv`
- `results/c1_comparison_suite_best_fixed_selected.csv`

### Task 2.3: Validate comparison metrics
**Objective:** ensure the comparison summary contains the required metrics and costs.

**Must include:**
- `point_f1_fixed`
- `affiliation_f1`
- `token_count_detection`
- `prompt_rows`
- `mean ± CI`

---

## Phase 3 — Traceability / consistency leg

**Goal:** measure the traceability leg with a held-out consistency evaluator.

**Gate 3:** `results/c2_consistency_summary.csv` exists and contains non-null `heldout_support_gap` for both fixed and evidence conditions, with delta and CI.

### Task 3.1: Confirm provenance shape
**Objective:** ensure each rule trace records the evidence windows and reference windows needed for evaluation.

**Files:**
- Inspect: `segment_selection/trace_logger.py`
- Inspect: `datasets/dataset.py`

**Expected provenance fields:**
- anomaly window start/end
- normal reference window
- prompt rows
- mode
- fixed-condition chunk indices
- held-out window IDs

### Task 3.2: Validate held-out pool leak safety
**Objective:** confirm the held-out pool contains windows not used in generation.

**Files:**
- Inspect: `datasets/dataset.py`
- Inspect: `runtime/engine.py`

**Verification:**
- generation evidence and held-out evidence do not overlap
- no leakage between creation and evaluation pools

### Task 3.3: Implement consistency evaluator
**Objective:** create `scripts/evaluate_rule_evidence_consistency.py`.

**Files to create:**
- `scripts/evaluate_rule_evidence_consistency.py`

**Responsibilities:**
- evaluate each rule against:
  - generated evidence
  - matched normal reference
  - held-out anomaly windows
  - held-out normal windows
- run both `fixed` and `evidence` conditions against the same held-out pool

### Task 3.4: Define metrics
**Objective:** compute the traceability metrics needed for the final decision.

**Metrics:**
- `heldout_anomaly_support_rate`
- `heldout_normal_violation_rate`
- `heldout_support_gap = anomaly_support - normal_violation`
- `local_support_gap` as sanity check only

### Task 3.5: Aggregate consistency summary
**Objective:** write `results/c2_consistency_summary.csv`.

**Expected content:**
- series × condition
- mean ± CI for each metric
- delta for `heldout_support_gap` between evidence and fixed

---

## Phase 4 — Morphology diagnostics

**Goal:** explain when evidence helps and when it hurts.

### Task 4.1: Build morphology table
**Objective:** correlate series-level performance and support gap with anomaly morphology.

**Files to create:**
- `results/c3_morphology_diagnostic.csv`

**Dimensions to capture:**
- anomaly shape / drift / spike / length / frequency
- series-level performance pattern
- series-level support-gap pattern

### Task 4.2: Write a short diagnostic note
**Objective:** provide a concise explanation of the main series-dependent pattern.

**Output:**
- short note in `docs/` or `reports/`

---

## Phase 5 — Final decision

**Goal:** lock a pre-registered GO / GO-reframe / NO-GO conclusion.

### Task 5.1: Evaluate the three legs
**Objective:** compute pass/fail for each leg using the pre-registered thresholds.

**Legs:**
- **Performance:** evidence point_f1_fixed vs best-fixed
- **Cost:** token or prompt-row reduction
- **Traceability:** `heldout_support_gap` delta

### Task 5.2: Write final decision document
**Objective:** create `reports/DECISION.md`.

**Must include:**
1. actual numbers for all 3 legs
2. PASS/FAIL/WEAK judgment for each leg
3. single final conclusion: `GO`, `GO-reframe`, or `NO-GO`
4. commit hash and result file paths
5. one recommended next step if not full GO

---

## Completion protocol

Whenever reporting completion for a phase, include:
1. exact command(s) run
2. CSV head or representative values
3. the gate values required by the phase
4. git commit hash

If any of those are missing, the phase is not complete.

---

## Immediate next step

1. Verify Phase 0 execution capability.
2. If actual LLM execution is possible here, proceed to Phase 1.
3. If not, stop and assign the execution venue explicitly.
