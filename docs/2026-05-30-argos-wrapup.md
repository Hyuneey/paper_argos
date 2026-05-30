# ARGOS Wrap-up — 2026-05-30

## 1) 변경 diff 요약
이번 슬라이스에서 바뀐 핵심은 세 축이다.

### Measurement / reporting
- `point_f1_fixed`를 주력 비교 지표로 고정하고, oracle 성격의 값은 `point_f1_oracle`로 분리했다.
- split별 anomaly point/event/ratio 및 small-test flags를 summary CSV에 평탄화해 넣었다.
- A4용 grouped summary에 mean / std / bootstrap 95% CI를 추가했다.

### Reproducibility / selection
- `driver.py`와 `runtime/engine.py`에 `seed` 경로를 관통시켰다.
- `run_chunk_sensitivity.py`가 seed를 전달하고, metadata에 `max_iter`와 `seed`가 남는다.
- segment selection에는 `normal_context_floor`를 추가해 density-only 후보가 과하게 유리해지지 않도록 정리했다.

### Validation / docs
- runtime, aggregate, segment selection 테스트를 보강했다.
- checklist의 A1~A6, B1~B3, Group C를 완료 상태로 정리했다.

## 2) 최종 검증 결과
- `unittest discover -s tests -p 'test_*.py' -v`
- 결과: **19/19 passed**

## 3) 최종 연구 결론
- A3 event sufficiency를 반영하면 `07927a9a18fa19ae`는 test anomaly event 수가 부족해 per-series 주장의 대상에서 제외된다.
- eligible series만 보면 corrected metric 기준으로 fixed chunking이 평균적으로 더 방어 가능하다.
- evidence mode는 일부 chunk size / series에서 이기지만, 전체적으로 안정적인 우세는 확인되지 않았다.
- 따라서 현재 가장 강한 결론은:

> measurement repair + sufficiency filtering 이후에도, chunking + normal-reference injection이 `point_f1_fixed`를 안정적으로 개선한다고 말할 수 없다.

## 4) 추천 커밋 메시지
**feat(argos): stabilize measurement, reproducibility, and segment-selection reporting**

짧게 쓰면:
**fix(argos): split fixed/oracle metrics, add variance control and selection floor**

## 5) 관련 산출물
- `docs/2026-05-30-group-a-gate-check.md`
- `docs/2026-05-30-eligible-only-ablation-summary.md`
- `docs/2026-05-30-argos-final-research-note.md`
- `docs/2026-05-30-a4-variance-control-report.md`
- `docs/2026-05-30-val-selected-chunk-summary.md`
- `results/chunk_sensitivity_summary.csv`
- `results/chunk_sensitivity_grouped_summary.csv`
- `results/chunk_sensitivity_val_selected.csv`
