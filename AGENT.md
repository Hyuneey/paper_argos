# Argos Agent Guide

이 문서는 이 레포를 처음 잡는 사람/에이전트가 **무엇을 보고, 어떤 순서로 작업해야 하는지**를 정리한 안내서다.

## 1. 한 줄 정의

Argos는 **LLM 기반의 시계열 이상 탐지 규칙 생성 시스템**이다.
연구의 핵심은 단순 성능 수치가 아니라, **측정이 깨끗한지**, **선택이 증거 지향적인지**, **결과가 재현 가능한지**다.

## 2. 먼저 기억할 것

- 이 레포는 실험/연구 코드다. 기능 추가보다 **측정 신뢰성**이 우선이다.
- 결과를 논할 때는 항상 **fixed metric**과 **oracle 성격의 값**을 분리해서 보자.
- `evidence` 모드는 비용을 줄일 수 있어도, 곧바로 성능 우위로 해석하면 안 된다.
- test split에 anomaly가 거의 없거나 0이면, 그 시리즈는 주장 근거로 쓰기 어렵다.
- selector의 `normal_context_floor`는 단순 점수 보정이 아니라 **게이트 수준의 제약**으로 취급한다.
- prompt / query / selector / tests 는 한 묶음이다. 하나만 바꾸고 끝내면 안 된다.

## 3. 현재 연구의 중심 축

최근 작업의 핵심은 대체로 아래 4가지였다.

1. **measurement**
   - `point_f1_fixed`를 주력 지표로 본다.
   - oracle 값은 따로 분리한다.

2. **reproducibility**
   - seed, metadata, split 통계, run trace를 남긴다.

3. **selection**
   - anomaly만 보지 않고 normal reference를 함께 보도록 정리한다.
   - density-only 후보가 과하게 유리해지지 않도록 제한한다.

4. **validation**
   - 핵심 회귀 테스트를 유지한다.
   - `pytest`가 없으면 `unittest` 또는 `compileall`로 대체 검증한다.

## 4. 레포 구조 요약

이 레포에는 별도의 `PROJECT_REGISTRY.md`가 없다. 대신 아래 문서들을 기준으로 작업 맥락을 잡는다.
- `README.md` — 프로젝트 개요 / 실행 진입점 / 기본 구조
- `RESULTS_AND_INTERPRETATION.md` — 결과 해석과 주의점
- `docs/2026-05-30-argos-wrapup.md` — 최근 변경 요약
- `docs/2026-05-30-evidence-gap-analysis.md` — evidence vs fixed 해석
- `docs/plans/*.md` — 진행 중인 계획/체크리스트


### 핵심 실행 흐름
- `driver.py`
  - 실험의 메인 진입점
- `runtime/`
  - 학습/평가 엔진
- `datasets/`
  - 데이터셋 로딩과 분할 관련 로직
- `eval_metrics/`
  - 평가 지표

### 에이전트/선택 로직
- `agent/`
  - detection / repair / review / mutate 계열 에이전트
- `agent/prompts/`
  - LLM 프롬프트 템플릿
- `segment_selection/`
  - 현재 연구에서 중요한 segment selection 로직
- `selector/`
  - 레거시/호환성 코드가 섞여 있을 수 있음

### 실험/분석 스크립트
- `scripts/run_chunk_sensitivity.py`
  - chunk sensitivity 재실행 래퍼
- `scripts/aggregate_chunk_sensitivity.py`
  - run 결과 집계
- `scripts/repo_dashboard.py`
  - 로컬 HTML 대시보드 생성
- `scripts/prepare_kpi_dataset.py`
  - KPI 데이터 준비

### 산출물
- `experiments/`
  - 실행별 원시 산출물, trace, metadata
- `results/`
  - 요약 CSV / 분석 산출물
- `docs/`
  - 연구 결론, gap analysis, wrap-up, 대시보드 HTML

## 5. 작업 순서 권장

### A. 코드를 건드리기 전
1. 관련 문서부터 본다.
2. `driver.py`, `runtime/`, `agent/`, `segment_selection/` 중 해당 흐름을 따라간다.
3. 기존 테스트가 있는지 확인한다.
4. 변경이 prompt/query/selector 중 어디에 걸리는지 먼저 분류한다.

### B. 코드 변경 후
1. 가장 작은 단위의 테스트를 먼저 돌린다.
2. `pytest`가 없으면 `uv run python -m unittest ...` 또는 `python -m compileall ...`로 확인한다.
3. 결과 파일이 바뀌면 `experiments/`와 `results/`의 산출물 의미를 다시 점검한다.
4. 결론 문장은 수치와 실험 범위를 함께 적는다.

### C. 실험 재실행 시
1. 데이터 소스가 실제로 있는지 확인한다.
2. seed, provider, temperature, chunk size, top_k 를 명시한다.
3. run metadata와 trace를 남긴다.
4. 집계 후 fixed/evidence 비교를 다시 본다.

## 6. 이 레포의 자주 쓰는 판단 기준

- **fixed vs evidence**
  - evidence는 비용 절감 후보일 수는 있어도, 자동으로 “더 좋다”가 아니다.
- **eligible-only 해석**
  - anomaly 수가 부족한 시리즈는 따로 제외할 수 있다.
- **selector 해석**
  - normal context가 부족한 후보는 점수만으로 살리면 안 된다.
- **run 완료 신호**
  - `best_rule_path.txt`, `metadata.json`, `selection_trace*.json` 존재 여부를 같이 본다.
- **테스트 해석**
  - 문법 통과만으로 충분하지 않다. prompt/query/selector의 의미적 회귀를 봐야 한다.

## 7. 자주 보는 파일

- `README.md` — 원래 프로젝트 설명
- `RESULTS_AND_INTERPRETATION.md` — 연구 결과 해석
- `docs/2026-05-30-argos-wrapup.md` — 최근 변경 요약
- `docs/2026-05-30-evidence-gap-analysis.md` — evidence vs fixed 비교
- `scripts/run_chunk_sensitivity.py` — 재실행 래퍼
- `scripts/aggregate_chunk_sensitivity.py` — 집계 스크립트
- `tests/test_detection_prompt.py` — prompt/query 회귀
- `tests/test_segment_selection.py` — selector 회귀

## 8. 실수하기 쉬운 지점

- prompt를 바꿨는데 query 조립은 그대로 두는 경우
- selector 정책을 바꿨는데 테스트를 안 고치는 경우
- 결과 CSV만 보고 데이터 수/eligible 여부를 안 보는 경우
- test split anomaly가 없는 시리즈를 일반화 근거로 쓰는 경우
- `results/`가 `.gitignore`라서 커밋에 빠지는 경우
- 인증/환경 문제를 코드 버그로 오해하는 경우

## 9. 추천 실행 습관

- 큰 변경 전에 먼저 **현재 구조 요약**을 만든다.
- 변경 범위가 작아도 **테스트 1개 + compileall 1개** 정도는 확인한다.
- 연구 결론은 항상 다음 3개를 같이 쓴다:
  - 무엇을 바꿨는지
  - 어떤 데이터 범위인지
  - 어떤 지표에서 좋아졌는지/안 좋아졌는지

## 10. 에이전트용 짧은 규칙

- 추측으로 결론 내리지 말 것.
- 측정과 해석을 분리할 것.
- prompt/query/selector/tests 를 한 세트로 볼 것.
- 결과가 애매하면 더 강한 결론을 쓰지 말 것.
- 필요한 경우에는 문서부터 갱신할 것.

---

필요하면 다음 단계로 이 문서를 더 쪼개서 아래처럼 분리할 수 있다.

- `AGENT.md` — 작업 원칙
- `docs/research-guide.md` — 연구 해석 가이드
- `docs/repo-map.md` — 코드 구조 맵
- `docs/experiment-playbook.md` — 실험 실행/재현 절차
