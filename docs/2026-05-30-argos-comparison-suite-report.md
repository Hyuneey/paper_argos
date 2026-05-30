# ARGOS Comparison Suite Result Report — 2026-05-30

## 0. Current status
- 체크리스트 항목 `T1.5 ~ T2.6`은 코드/테스트 기준으로 완료 상태입니다.
- 비교/집계 파이프라인의 provenance, held-out pool, summary/delta 집계가 반영되었습니다.
- `unittest` 기준 검증은 통과했습니다.

## 1. Final deliverables
- `segment_selection/trace_logger.py`
  - selection trace에 `provenance` 블록 추가
- `datasets/dataset.py`
  - provenance 수집 및 trace 전달
  - held-out window pool 생성/저장 추가
- `runtime/engine.py`
  - `stats.json`에 held-out pool 경로/개수 기록 추가
- `scripts/aggregate_comparison_suite.py`
  - condition summary 지표 확장
  - fixed vs evidence delta CSV 생성 추가
- `tests/test_segment_selection.py`
  - provenance / held-out pool 테스트 추가
- `tests/test_aggregate_comparison_suite.py`
  - summary / delta 집계 테스트 추가
- `tests/test_review_agent_metrics.py`
  - consistency evaluator 관련 테스트

## 2. Change summary
### 2.1 Provenance logging
- selection trace JSON에 provenance를 별도 블록으로 기록하도록 변경했습니다.
- 최소 침습 방식으로 기존 trace schema를 유지하면서 확장했습니다.

### 2.2 Held-out window pool
- validation split 기반 held-out window pool을 생성하도록 추가했습니다.
- `held_out_window_pool.json` artifact를 남기도록 했습니다.
- runtime stats에는 pool의 path/count만 기록합니다.

### 2.3 Comparison suite metrics
- summary에 다음 지표를 추가했습니다.
  - `mean_point_f1`
  - `mean_event_f1pa`
  - `std_point_f1`
  - `std_event_f1pa`
  - `ci95_point_f1`
  - `ci95_event_f1pa`
- fixed vs evidence delta table을 `dataset + series + chunk_size` 축으로 생성하도록 정리했습니다.

## 3. Generated artifacts
- `results/c1_comparison_suite_rows.csv`
- `results/c1_comparison_suite_summary.csv`
- `results/c1_comparison_suite_best_fixed_selected.csv`
- `results/c1_comparison_suite_fixed_vs_evidence_delta.csv`

### Note on delta output
- 현재 실행에 사용한 suite root `experiments/c1_comparison_suite_v1`에는 `fixed`, `random`, `anomaly_centered`, `event_bounded_reference` 조건만 존재합니다.
- 따라서 `fixed vs evidence` pair가 없어 `results/c1_comparison_suite_fixed_vs_evidence_delta.csv`는 현재 비어 있습니다.
- evidence 조건 run이 추가되면 같은 집계 스크립트를 다시 돌려 delta table을 채울 수 있습니다.

## 4. Validation
- `./.venv/bin/python -m unittest tests.test_aggregate_comparison_suite tests.test_eval_metrics tests.test_review_agent_metrics tests.test_dataset_stats tests.test_segment_selection tests.test_runtime_config`
- 결과: **19/19 passing**
- `python3 -m py_compile scripts/aggregate_comparison_suite.py tests/test_aggregate_comparison_suite.py`
- 결과: **pass**

## 5. Commit checkpoint
- 코드 변경과 report 작성이 끝났으므로 커밋/푸시만 남은 상태입니다.
- 권장 푸시 대상: `paper/main`
