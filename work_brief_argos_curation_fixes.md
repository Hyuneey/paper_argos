# 작업 의뢰서 — paper_argos PoC 측정 정정 및 연구방향 정렬

- 대상 repo: `https://github.com/Hyuneey/paper_argos`
- 작성일: 2026-05-29
- 수신: 코딩/리서치 에이전트
- 성격: **기존 PoC 결과는 측정 결함으로 인해 아직 신뢰할 수 없음. 1차 목표는 "더 좋은 결과"가 아니라 "신뢰할 수 있는 측정의 복구"다.**

> 이 의뢰서는 대화 맥락 없이 단독으로 실행 가능하도록 작성되었다. 작업 전 아래 "배경·제약"과 §0~§2를 반드시 읽고, §3 작업을 우선순위(P0→P2) 순서로 진행한다. **측정 정정(Group A) 없이 결론을 내리거나 새 실험을 확장하지 말 것.**

---

## 배경·제약 (반드시 먼저 읽을 것)

**이 연구가 왜 이 형태인가.**
이 작업은 더 넓은 방향(이상치 *유형* 다중분류 중심의 detector 연구)에서 출발했으나, 지도교수 피드백과 선행연구 검토를 거쳐 **의도적으로 좁혀진** 것이다. 최종 형태는 "새로운 detector"가 아니라 **ARGOS-style LLM rule generation의 입력단(training-time evidence curation)을 분석·개선하는 연구**다. 논문의 척추는 **입력 비용 + rule-evidence traceability**이며, **성능(F1)은 개선 목표가 아니라 유지(preservation) 제약**이다. 따라서 "성능을 끌어올리는" 방향의 확장은 이 연구의 목표가 아니다.

**이미 검토 후 의도적으로 버린 방향 (복원 금지 — 이유 포함).** 아래는 "더 좋아 보여서" agent가 선의로 다시 끌고 들어오기 쉬운 것들이다. 절대 도입하지 말 것.
- **ARTIST-style RL controller / adaptive segment selector** — 학습·추론 시점 selector는 누수·비용·범위 폭증을 부르고, 본 연구는 *deterministic training-time curator*만 다룬다.
- **Outlier Exposure** — detector boundary 학습 방식으로, 본 연구의 prompt-evidence curation과 다른 문제다.
- **causal / root-cause explanation, LLM reasoning faithfulness 검증** — 본 연구의 설명성은 "rule-evidence consistency"로 한정한다.
- **새 SOTA detector / full ARGOS 대체** — 본 연구는 ARGOS 입력단 분석이지 detector 경쟁이 아니다.
- **multivariate SWaT 본실험** — multivariate 인과·센서 선택 oracle 문제로 범위가 폭증. (필요시 향후 정성 case study로만.)

**운영 규칙 (agent가 헛발질하기 쉬운 지점).**
1. **train-LLM-only는 의도된 controlled testbed다.** ARGOS의 headline 숫자는 base-detector fusion(Aggregator)에서 나온다. F1이 낮게 나와도 **fusion/Aggregator를 켜서 F1을 끌어올리지 말 것**이며, ARGOS headline 숫자와 직접 비교하지 말 것. 본 연구는 windowing 변수를 fusion·FN/FP aggregation·contrastive retrieval의 교란 없이 격리하기 위해 일부러 train-LLM-only를 쓴다.
2. **예산·런타임 한계를 존중하라.** LLM 호출은 비싸고(self-hosted라면 GPU 8~12GB 제약), 대규모 sweep은 금지. chunk×series×repeat 조합을 임의로 키우지 말고 의뢰서에 명시된 범위 안에서만 실행. "더 철저히" 한다며 호출 수를 폭증시키지 말 것.
3. **범위·연구방향 결정은 agent의 권한이 아니다.** Group A 게이트(§3) 등 판단 분기에서 **자동으로 다음 단계로 넘어가거나 범위를 넓히지 말고, 멈춰서 결과를 보고**하라. 프레이밍·지표 철학·데이터셋 확장 같은 결정은 연구자가 내린다. 의뢰서 범위만 실행한다.

---

## 0. 연구 한 줄 정의와 현재 상태

**연구 정의 (석사학위 논문):**
ARGOS-style LLM 기반 시계열 이상치 rule generation에서, fixed-size chunk 입력 대신 **학습 시점(training-time)에 anomaly/에러 주변의 compact temporal evidence window를 구성(curation)**했을 때, 룰 성능을 유지하면서 **입력 비용**과 **rule-evidence traceability**를 개선할 수 있는지 분석·검증한다. Curator는 **rule training 단계에서만** 동작하며, 배포 시점에는 생성된 deterministic Python rule만 사용한다.

**현재 repo 상태 (검증 완료):**
- ARGOS fork + `segment_selection/` 모듈 + chunk-sensitivity 스크립트가 구현되어 있고, KPI 3개 series에 대해 pilot 실험(fixed vs evidence, chunk 6종, repeat 3)이 1회 수행됨.
- 엔지니어링 구조는 양호. 그러나 **평가 지표·평가 신호·variance 처리에 결함**이 있어, 현재 요약의 "evidence가 fixed를 +0.2 이긴다"는 **신뢰할 수 없다.**
- 데이터셋 파일은 repo에 포함돼 있지 않음(`datasets/`에 `dataset.py`만 존재). 실행 전 KPI 데이터를 `scripts/prepare_kpi_dataset.py`로 준비해야 함.

---

## 1. 연구 방향 (왜 이렇게 고치는지 — 맥락)

### 1.1 논문의 척추 (단일)
**"동등 성능 유지 + 입력 비용 감소 + 더 높은 rule-evidence consistency."**
성능(F1)은 main claim이 아니라 **preservation constraint**다. 성능이 개선되면 보너스, 동등하면 비용·traceability로 논문이 선다. 성능 향상 논문으로 쓰지 않는다.

### 1.2 Core method = Event-bounded curation + Matched reference (동시 주입)
- **Event-bounded window:** window 길이를 `chunk_size`의 고정 비율(0.25/0.5/1.0)로 자르지 말고, **anomaly event의 자연스러운 범위**(deviation 시작 → normal 복귀)에 적응시킨다. 즉 spike는 짧게, level-shift/drift는 길게. 이것이 morphology-adaptive를 RL 없이 deterministic하게 얻는 핵심이다.
- **Matched reference (동시 주입):** compact anomaly window는 normal context가 부족해 룰이 false positive를 늘리거나 일반화에 실패한다(아래 §2-4에서 데이터로 확인됨). 따라서 **anomaly window와 시간적으로 인접한 normal reference window를 프롬프트에 *함께* 주입**한다. "후보 중 하나만 고르기"가 아니다. 이것이 "compact해도 성능을 잃지 않는" 메커니즘이다.
- ARGOS와의 경계: ARGOS의 contrastive retrieval은 point-level·산발적·fusion 설정 전용이다. 본 연구의 reference는 **window-level·contiguous·local dynamics 보존·train-LLM-only**라는 점에서 겹치지 않는다. 보고서에서 이 구분을 명시한다.

### 1.3 Optional Stage-2 (지금 구현하지 않음, 향후 깊이 확보용)
- **Type-conditioned windowing:** 이상치 유형(spike / level-shift / drift)마다 필요한 window 정책이 다르다는 가설을 분석축으로. 단 type label이 필요하고 KPI/Yahoo는 유형이 빈약하므로, controlled injection 또는 typed dataset이 필요. **MVP가 신뢰 가능한 양성 결과를 낸 뒤에만** 착수.

### 1.4 절대 주장하지 않을 것 (보고서·코드 주석 공통)
- 새로운 SOTA TSAD 모델 / full ARGOS 대체
- ARTIST-style RL controller 구현 / query-conditioned adaptive selector
- Outlier Exposure 적용 / causal·root-cause explanation / LLM reasoning faithfulness 검증
- "모든 데이터셋에 일반화됨"

---

## 2. 현재 코드의 검증된 결함 (수정 근거)

다음은 코드·결과를 직접 검사해 확인한 사실이다. 각 작업(§3)은 여기에 대응한다.

1. **(치명적) 평가 지표가 이진 룰 출력에 부적합하다.**
   - 룰 출력은 `agent/review_agent.py`의 `eval()`에서 `scores = np.zeros(..., dtype=int)` — **이진(0/1)**이다.
   - `eval_metrics/point_f1.py`의 `PointF1.calc`는 `sklearn.metrics.precision_recall_curve` 후 **test 라벨로 F1을 최대화하는 threshold를 고른다(oracle threshold).** 이진 출력에 적용하면 결과는 `max(룰 그대로 F1, "전부 anomaly로 찍기" F1)`이 된다.
   - 재현 검증 결과: **"아무것도 안 찍는 룰"과 "전부 찍는 룰"이 동일하게 F1 ≈ 0.014**를 받는다. 즉 결과 CSV의 `0.0145` 바닥값은 룰 성능이 아니라 **지표 바닥(예측-전부 baseline)**이며, flooding 룰과 do-nothing 룰을 구별하지 못한다.
   - 결론: 현재 `test_f1`은 배포될 룰의 성능이 아니다. PA를 금지했지만, 이 **oracle-threshold-on-test 역시 PA와 같은 계열의 낙관적 프로토콜**이다.

2. **(치명적) test 평가 신호가 퇴화돼 있다.**
   - 모든 결과 행에서 `recall`이 정확히 `0.4999...`로 상수다(룰과 무관). `test_f1`은 `{0.0145, 0.08, 0.16, 0.33, 0.53, 0.66}` 같은 이산값으로만 점프한다.
   - 정황상 이 KPI series들의 test split에는 anomaly **event가 극소수(추정 ~2개 동일 길이 burst)**뿐이며, F1이 사실상 "burst를 0/1/2개 잡았나"의 3~4값 신호다. → 단일 series에서 ±0.2 차이는 통계적으로 무의미할 수 있다.

3. **(치명적) LLM 룰 생성 variance가 효과를 삼킨다.**
   - 예: `07927` fixed chunk=1000의 test_f1 = `{0.0145, 0.0145, 0.432}`. 단일 repeat이 바닥↔0.43으로 튄다. repeat=3, temperature 미고정으로는 해석 불가.

4. **(구조적) Matched-reference 메커니즘이 구현됐으나 죽어 있다.**
   - `segment_selection/candidate_generator.py`는 `nearby_normal_reference` 후보를 만들지만, `segment_selection/selector.py`는 **최고 utility 후보 하나만** 선택한다(tie-break도 `anomaly_density`). reference는 density가 낮아 utility가 낮으니 **선택되지 않는다.**
   - 결과적으로 작은 window에서 `anomaly_density=1.0`(anomaly only, normal context 0) window가 선택돼 성능이 붕괴한다(요약 문서 self-note와 일치). 즉 §1.2의 핵심 메커니즘이 코드에서 작동하지 않는다.

5. **(방법론) "best chunk"를 test로 고른다(leakage).**
   - 요약 문서가 `fixed best test F1` / `evidence best test F1`로 best chunk를 **test 성능으로** 선택했다. `val_f1` 컬럼이 이미 있으므로 val로 선택해야 한다.

6. **(재현성) 모델이 비고정·비공개 경로.**
   - 요약 기준 `gpt-5.4-mini` (Codex OAuth)로 실행됨 — 작성자 스스로 "stable public baseline 아님"이라 명시. 어떤 주장도 pinned 공개 모델로 재실행 필요.

---

## 3. 작업 목록 (우선순위 순)

> 각 작업: **[유형]** BUGFIX(측정 유효성 복구) / DESIGN(연구 방법 변경) / REPRO(재현성) / OPTIONAL.
> 각 작업에 **수정 파일**, **변경 내용**, **수용 기준(acceptance)**을 명시한다.

### Group A — 측정 정정 (P0, 가장 먼저, 새 실험 확장 금지)

#### A1. [BUGFIX] 이진 룰용 고정-결정 지표 추가 + oracle-threshold 주력 사용 중단
- **파일:** `eval_metrics/`에 신규 `point_f1_fixed.py`(또는 `binary_f1.py`); `agent/review_agent.py`의 `eval_scores_by_metrics` 호출부.
- **변경:**
  - 룰의 이진 예측(`scores >= 0.5`)으로 **threshold 탐색 없이** precision/recall/F1을 계산하는 지표를 구현하고, 이를 **primary point-level 지표(`point_f1_fixed`)**로 보고.
  - 기존 `point_f1.py`(best-threshold)는 삭제하지 말고 컬럼명을 `point_f1_besthr_oracle`로 바꿔 **"낙관적 상한, 주력 아님"**으로만 병기.
  - `point_f1pa`, `event_f1pa`는 계산은 유지하되 **"PA — 참고용, 주장 근거로 사용 금지"**로 라벨링.
- **수용 기준:** "전부 anomaly로 찍는 룰"과 "아무것도 안 찍는 룰"의 `point_f1_fixed`가 서로 다르고 둘 다 낮게(0에 가깝게) 나온다. 동일 입력에 대해 동일 출력(deterministic) 확인.

#### A2. [BUGFIX] 비-PA event/range 지표 추가
- **파일:** `eval_metrics/affiliation_f1.py`(신규); `requirements.txt`.
- **변경:** **affiliation-based precision/recall/F1**(Huet et al., 2022)을 range 지표로 추가한다. `pip install affiliation-metrics` 사용 권장(직접 재구현 시 버그 위험). 대안으로 **composite F1**(point-level precision × event-level recall의 조화평균; precision은 PA로 부풀리지 않음)을 구현해도 됨.
- **수용 기준:** flooding 룰이 affiliation/composite에서 낮은 precision으로 페널티를 받는다(즉 PA처럼 부풀지 않는다).

#### A3. [BUGFIX/분석] test split의 anomaly 통계 로깅 및 series 적격성 판정
- **파일:** `scripts/aggregate_chunk_sensitivity.py`, `scripts/prepare_kpi_dataset.py`(또는 데이터 점검 노트북).
- **변경:** 각 split(train/val/test)에 대해 **anomaly 포인트 수, anomaly event 수, anomaly 비율**을 계산해 `stats.json`/summary에 기록. test event 수가 적은(예: `< 5 events` 또는 anomaly point가 매우 적은) series는 **"per-series F1 주장 부적격"**으로 플래그.
- **수용 기준:** 각 series의 test anomaly event 수가 표로 출력된다. 부적격 series는 명시적으로 표시된다.

#### A4. [BUGFIX] variance 통제 — temperature 고정 + repeat 확대 + CI 보고
- **파일:** `scripts/run_chunk_sensitivity.py`(인자 추가), `agent/*.py`의 모든 `LLM(...)` 호출부, `scripts/aggregate_chunk_sensitivity.py`.
- **변경:**
  - `--temperature` 인자를 추가하고 Detection/Repair/Review의 LLM 호출 temperature를 이 값으로 전달. **실험 기본값 0.0.**(현재 일부 경로는 `temperature=0.7` 하드코딩 — 모두 인자화.)
  - `--repeats` 기본값을 **≥ 5(권장 10)**로 상향.
  - aggregation이 chunk_size별로 **mean ± std + bootstrap 95% CI**를 산출. 방법 간 비교는 **CI가 분리될 때만** "차이 있음"으로 보고.
- **수용 기준:** summary에 mean±CI가 출력된다. temperature=0에서 동일 설정 재실행 시 variance가 0.7 대비 줄어든다.

#### A5. [BUGFIX] "best chunk" 선택을 val 기준으로 (leakage 제거)
- **파일:** summary 생성 로직(요약 문서 생성 스크립트 또는 `aggregate` 후처리).
- **변경:** chunk_size 선택은 **mean val_f1** 기준으로만. 선택이 고정된 뒤 해당 설정의 **test 지표를 한 번만** 보고. `*best test F1*` 식의 test 기준 선택 전면 금지.
- **수용 기준:** summary에 "selected by val" 문구와 함께, val로 고른 chunk와 그때의 test 성능이 분리 보고된다.

#### A6. [REPRO] 모델 고정
- **파일:** `scripts/run_chunk_sensitivity.py`(`--llm_engine` 기본값), `config/agent.yaml`, `metadata.json` 기록부.
- **변경:** **pinned 공개 스냅샷 모델** 사용(예: `gpt-4o-mini-<날짜 스냅샷>` 또는 `config/agent.yaml`의 `self_hosted_llm_list`에 있는 오픈모델). 각 run의 `metadata.json`에 정확한 모델 문자열·temperature·top_k·max_iter를 기록.
- **수용 기준:** Codex OAuth/비고정 모델 경로가 실험 기본에서 제거된다. metadata만으로 재현 설정이 복원 가능하다.

> **Group A 완료 후 필수 게이트:** fixed-chunk만으로 **재실행**하여 (a) `point_f1_fixed`가 flood/do-nothing을 구분하는지, (b) test event 수가 충분한 series가 있는지, (c) variance가 통제됐는지 확인한다. 셋 중 하나라도 실패하면 §3 이후로 진행하지 말고 보고한다.

---

### Group B — Core method 정렬 (P1, Group A 통과 후)

#### B1. [DESIGN] Matched reference 동시 주입 (pick-one → inject-together)
- **파일:** `segment_selection/selector.py`, `segment_selection/candidate_generator.py`, `agent/detection_agent.py`(`build_detection_agent_v3_prompt` 및 query 조립부, 예: `final_query = "##### DATA\n" + current_data_str`), `agent/prompts/detection.py`.
- **변경:**
  - curation 출력이 **(anomaly evidence window, matched normal reference window) 쌍**을 반환하도록 변경. selector는 anomaly window를 고르되, reference는 경쟁 후보가 아니라 **함께 전달되는 컨텍스트**로 처리.
  - detection 프롬프트에 `##### NORMAL REFERENCE` 섹션을 추가해 reference window를 주입(코드에 이미 `##### POSITIVE/NORMAL DATA` 선례 있음 — 동일 패턴 재사용 가능).
- **수용 기준:** evidence 모드 프롬프트에 anomaly window와 normal reference가 동시에 포함된다. trace에 두 window의 index 범위가 모두 기록된다.

#### B2. [DESIGN] Event-bounded 후보 생성 추가
- **파일:** `segment_selection/candidate_generator.py`.
- **변경:** 기존 `chunk_size*{0.25,0.5,1.0}` centered window에 더해, 각 anomaly event `[s,e]`에 대해 **event-bounded window** `[s - m_pre, e + m_post]`를 생성(margin은 event 길이에 비례하거나 최소값 보장 — 예: `m = max(min_margin, k*(e-s))`). window 길이가 event 형태에 적응되게 한다.
- **수용 기준:** 동일 chunk budget에서 spike형 event와 shift형 event의 선택 window 길이가 서로 다르게 나온다(trace로 확인).

#### B3. [DESIGN] Utility 가중치 재조정 + normal-context floor
- **파일:** `segment_selection/utility_scorer.py`(`DEFAULT_WEIGHTS`), `configs/segment_selector_default.yaml`, `segment_selection/selector.py`(tie-break).
- **변경:**
  - `anomaly_density` 가중(현재 0.30, 최대)을 낮추고(예: 0.10), `normal_contrast`/`anomaly_coverage`를 상향.
  - **하드 제약:** `anomaly_density == 1.0`(normal 0%) 또는 normal 비율이 임계 미만인 후보는 **선택에서 배제**(floor). selector tie-break에서 density 우선을 제거.
- **수용 기준:** 작은 budget에서도 density=1.0 window가 선택되지 않는다.

---

### Group C — 비교 실험 재설계 (P1, B 이후)

#### C1. [DESIGN] Counterfactual 비교군 고정
- **파일:** `scripts/run_chunk_sensitivity.py`, aggregation.
- **변경:** 다음을 동일 budget·동일 평가 프로토콜로 비교:
  `fixed_chunk` / `best_fixed_chunk(val로 선택)` / `same-length random window` / `anomaly-centered(density-only)` / `event-bounded + matched reference(제안)`.
- **수용 기준:** 모든 비교군이 §4 지표로 동시에 산출되고, mean±CI로 보고된다. random 대비 제안의 우위가 CI로 판별된다.

#### C2. [분석] 다수 series 집계 / 적격 series 선정
- **파일:** aggregation, 데이터 준비 스크립트.
- **변경:** A3에서 적격으로 판정된(anomaly event가 충분한) KPI series를 **다수(예: ≥ 8개)** 사용하고, per-series cherry-pick 대신 **series에 걸친 분포(mean±CI) 또는 micro-average**로 보고.
- **수용 기준:** 결론이 단일 series가 아니라 다수 series 분포로 뒷받침된다.

---

### Group D — 선택 작업 (P2, 이후)

#### D1. [OPTIONAL] FN/FP 모드 sanity check
train-LLM-only에서 확인된 효과가 ARGOS의 FN/FP 설정에서도 유지되는지 소규모 확인(주력 주장 아님).

#### D2. [OPTIONAL] Type-conditioned windowing (Stage-2)
이상치 유형별 window 정책 분석. type label 필요 → controlled injection 또는 typed dataset 도입 후. **MVP 양성 결과 전에는 착수 금지.**

---

## 4. 평가 프로토콜 규약 (Metric Contract)

모든 실험은 다음을 따른다.

- **Primary (주장 근거):**
  - `point_f1_fixed` — 이진 예측 고정 결정 point-level F1 (threshold 탐색 없음).
  - `affiliation_f1` (또는 composite F1) — 비-PA range 지표.
  - **Cost:** detection 입력 token 수, prompt rows, **evidence compression ratio = fixed_chunk_len / curated_window_len**.
- **Secondary (참고용, 주장 근거 금지):** `point_f1_besthr_oracle`, `point_f1pa`, `event_f1pa`.
- **금지:**
  - PA 계열 지표를 주력/결론 근거로 사용 금지.
  - test 라벨로 threshold 또는 chunk/하이퍼파라미터 선택 금지(oracle-on-test 금지).
  - VUS-PR/AUROC는 **연속 score 전용** — 이진 룰 출력에는 적용하지 않는다(적용하려면 룰을 score화하는 별도 정의가 필요하며, 그 전엔 보고하지 않는다).
- **보고 단위:** chunk_size·method별 **mean ± std + 95% CI**. 비교는 CI 분리 시에만 유효.

---

## 5. 실행 순서·재현 설정

1. 환경 구성(`requirements.txt`, `pyproject.toml`), KPI 데이터 준비(`scripts/prepare_kpi_dataset.py`).
2. **Group A 전부 완료 → 게이트 통과 확인.**
3. fixed-chunk 재실행(정정된 지표·temp=0·repeat≥5)으로 baseline 재수립.
4. Group B(core method) 구현 → Group C 비교 실험.
5. Group D는 선택.

재현 고정값(권장 기본): `temperature=0.0`, `repeats≥5`, pinned 공개 모델, chunk_sizes는 우선 `{250, 1000, 2500}`로 축소 후 필요시 확대. 모든 run에 `metadata.json`(모델·temp·top_k·max_iter·seed) 기록.

---

## 6. 산출물

```
eval_metrics/point_f1_fixed.py          # A1
eval_metrics/affiliation_f1.py          # A2 (또는 composite)
scripts/run_chunk_sensitivity.py        # A4/A6 인자 추가
scripts/aggregate_chunk_sensitivity.py  # A3/A4/A5 mean±CI, val 선택, anomaly stats
segment_selection/candidate_generator.py# B2 event-bounded
segment_selection/selector.py           # B1/B3
segment_selection/utility_scorer.py     # B3
configs/segment_selector_default.yaml   # B3
agent/detection_agent.py, agent/prompts/detection.py  # B1 reference 주입

results/  (정정 후 재생성: mean±CI 포함)
reports/
  00_measurement_fix_report.md          # A 무엇을 왜 고쳤고, 정정 전후 숫자가 어떻게 달라졌는가
  01_chunk_sensitivity_report.md        # 정정된 지표 기준, val 선택, mean±CI
  02_core_method_report.md              # event-bounded + matched reference 비교
  03_failure_or_null_report.md          # 효과 없음/실패 케이스 분석
```

각 보고서는 **정정 전후 비교**를 반드시 포함한다(특히 `00`: 기존 `0.0145` 바닥값·recall=0.5·oracle threshold가 어떻게 바뀌었는지).

---

## 7. 성공·중단 기준

**성공(아래 중 하나 이상, CI 분리 기준):**
- A. event-bounded+reference가 best-fixed 대비 동등 성능에서 **입력 비용 유의하게 감소**.
- B. 동등 비용에서 **성능 유의 개선**.
- C. 동등 성능·비용에서 **rule-evidence consistency 개선**.
- D. **Null result + insight:** "이 setting에서 temporal curation은 무의미하며 그 이유는 X" — 이것도 유효한 학위논문 결과. 숨기지 말고 분석.

**중단/보고 전환:**
- Group A 게이트 실패(지표가 여전히 flood/do-nothing 구분 못 함, 적격 series 없음, variance 통제 불가).
- 정정된 지표 기준으로 모든 방법이 동일 CI 안 → "windowing은 병목이 아님"으로 보고하고 D 방향 정리.

---

## 8. 절대 하지 말 것 (요약)

- test 라벨로 threshold/하이퍼파라미터/chunk 선택(oracle-on-test).
- PA 지표를 결론 근거로 사용.
- curator를 배포(runtime) 경로에 포함하거나, test 라벨을 curation에 사용.
- 정정 전 결과(특히 `0.0145` 바닥·recall=0.5)를 그대로 인용해 성능 주장.
- SOTA/full-ARGOS 대체/RL controller/Outlier Exposure/causal·faithfulness 주장.
- 단일 series cherry-pick으로 일반화 주장.

**보고는 정직하게. Null/실패 결과도 그대로 기록한다.**