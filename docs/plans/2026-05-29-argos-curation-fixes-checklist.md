# ARGOS curation fixes checklist

기준 문서: `work_brief_argos_curation_fixes.md`

## 0. 현재 판정
- [x] 측정 정정/요약 방향을 문서로 정리함
- [x] `point_f1_fixed` 집계 반영
- [x] `NORMAL REFERENCE` 주입 방향을 문서화/반영함
- [x] Group A 전체 완료

## 1. Group A — 측정 정정
### A1. 이진 룰용 고정-결정 지표
- [x] `point_f1_fixed` 반영
- [x] summary에 포함
- [x] 기존 oracle-threshold 지표를 주력에서 완전히 분리

### A2. 비-PA range 지표
- [x] affiliation F1 또는 composite F1 추가
- [x] flooding 룰에 낮은 precision 페널티가 들어가는지 확인

### A3. test split 통계/적격성
- [x] split별 anomaly point 수 / event 수 / 비율 로깅
- [x] test event 수 적은 series 플래그 처리

### A4. variance 통제
- [x] temperature = 0.0 고정
- [x] repeats를 5 이상으로 확대
- [x] mean ± std + 95% CI 보고

### A5. best chunk 선택 기준
- [x] test가 아니라 val 기준으로 chunk 선택
- [x] 선택된 chunk와 test 결과를 분리 보고

### A6. 모델 재현성
- [x] pinned 공개 모델로 실험 고정
- [x] metadata.json에 모델/temperature/top_k/max_iter/seed 기록

## 2. Group B — core method 정렬
### B1. matched reference 동시 주입
- [x] normal reference를 프롬프트에 넣는 방향으로 정리함
- [x] anomaly window + normal reference가 동시에 trace에 남는지 확인

### B2. event-bounded 후보 생성
- [x] event morphology에 맞는 가변 window 후보 생성

### B3. utility / floor 규칙
- [x] normal-context floor 적용
- [x] density=1.0 후보가 선택되지 않는지 확인

## 3. Group C — 비교 실험 재설계
- [x] baseline / chunking only / chunking+reference / fixed eval의 비교군을 명시
- [x] mean ± CI 비교표 생성
- [x] 다수 series 집계 또는 적격 series 선정

## 4. 현재까지 확보된 산출물
- [x] `results/chunk_sensitivity_ablation_table.csv`
- [x] `docs/plans/2026-05-29-ablation-summary.md`
- [x] `docs/plans/2026-05-29-research-poc-plan.md`
- [x] `docs/plans/2026-05-29-research-poc-grid.md`

## 5. 완료 상태
- 모든 항목 완료
- 추가 실험은 새 이슈/새 체크리스트에서 시작
