# P2b 순차 결정 루프 결과

## 1. 무엇을 왜 만들었나

`roarm-heightmap-v1` 관측을 받아 행동을 고르고, 교체 가능한 실행기에 넘긴 뒤, 새 높이 지도를 다시 관측하는 유한 반복 루프를 만들었다. 목적은 합성 더미에서 정책 비교·지표 집계·재현성·시각 재생을 먼저 고정하고, 이후 DEME 실행기와 예측 모델을 같은 경계에 연결하는 것이다.

고정 계약은 그대로 유지했다.

- 높이 지도: `(76, 38)`, `cell_m=0.005`, `origin=(0.125, -0.190)`, `frame=roarm_base`, `agg=max`, 빈 셀 수치 `0.0`과 별도 `valid` 마스크
- 행동: `[x_m, y_m, dir_x, dir_y]`, `x ∈ [0.125, 0.315)`, `y ∈ [-0.190, 0.190)`, 방향은 단위 벡터
- 주 지표: 총 시간, 총 스쿱 횟수, 실패율; 이동거리는 보조

## 2. 실행 순서와 종료 조건

실행 순서는 아래와 같다.

1. `ScoopExecutor.reset(initial_heightmap)`
2. `observe()`로 계약 검증된 높이 지도 취득
3. 선택적 `PredictionProvider`가 평균·분산·위험 조정 점수를 `DecisionPredictions`로 구성
4. `ScoopPolicy.select_action(observation, DecisionContext)` 호출
5. 행동 범위·유한값·방향 단위벡터 검증
6. `executor.execute(action)` 호출
7. 반환된 새 높이 지도와 지표 검증
8. 누적 적재량·시간·실패·이동거리 갱신 후 목표량 판정

반복 상한은 240회다. 목표량 도달 외에 유효 재료 셀 소진, 연속 실패 5회, 총 실패 20회, 정책 예외, 범위 밖 행동, 실행기 예외, 관측 계약 위반을 각각 명시적 종료 사유로 기록한다.

## 3. 실행기 교체 경계

추상 `ScoopExecutor`는 세 메서드만 요구한다.

- `reset(initial_observation)`
- `observe() -> Heightmap`
- `execute(action) -> ExecutionResult`

`run_episode`는 실행기 내부 상태나 해석적 배열을 직접 읽지 않는다. 자체 검사에서는 별도 `_OneShotExecutor`를 같은 루프에 꽂아 1회 목표 달성을 확인했다. 이후 DEME 어댑터도 위 세 메서드와 `ExecutionResult`만 맞추면 정책·종료·지표·Rerun 층을 재사용할 수 있다.

현재 `AnalyticExecutor`는 방향을 반영한 60 × 58 mm 타원 발자국에서 최대 10 mm를 깎고, 인접 셀 높이차를 안식각 기반으로 완화한다. 완화는 남은 체적을 보존하며, 깎인 체적에 합성 밀도 550 kg/m³를 곱해 적재량으로 환산한다. 이 값들은 개발용 합성 실행기 파라미터이며 실물 재료 측정값으로 사용하지 않는다.

## 4. 정책 비교 결과

동일 초기 더미, seed 457, 초기 합성 질량 `0.142124410 kg`, 목표 비율 0.65(`0.092380867 kg`)로 실행했다. 네 정책 모두 목표량에 도달했고 실패 시도는 0회였다.

| 정책 | 총 스쿱 횟수 | 총 시간 [s] | 실패율 | 이동거리 [m] | 적재량 [kg] | 종료 |
|---|---:|---:|---:|---:|---:|---|
| `greedy_high` | 13 | 89.291384 | 0.000000 | 6.072846 | 0.094394219 | `target_reached` |
| `greedy_low` | 37 | 254.526152 | 0.000000 | 17.381537 | 0.093056524 | `target_reached` |
| `center_out` | 27 | 182.960080 | 0.000000 | 11.990020 | 0.094483847 | `target_reached` |
| `random` | 17 | 118.311958 | 0.000000 | 8.327989 | 0.093034331 | `target_reached` |

횟수는 `13/37/27/17`로 정책에 따라 실제로 달랐다. 독립 `run_repeat`의 결정론 payload는 primary와 canonical JSON 바이트가 동일하며 SHA-256은 `19a5e1bee4d074806508444652c6718cd8780eca34de2fa698329c1bd73a82ac`다.

## 5. 불확실성 투입 자리

`DecisionContext`가 아래 네 항목을 매 정책 호출에 전달한다.

- `predictor`: 이후 `model_scoop_predictor` 인스턴스 슬롯
- `predictions.means`: 후보별 예측 평균
- `predictions.variances`: 후보별 예측 분산
- `predictions.risk_adjusted_scores`: LCB/UCB 계열 위험 조정 점수

`risk_config`와 `risk_score_fn`도 같은 context에 들어간다. 인터페이스 전용 `ModelPolicySlot`은 이를 소비하도록 정의했으며, 규칙 정책은 해당 필드를 무시한다. 자체 검사는 분산 `[0.01, 0.04]`, predictor sentinel, risk 설정, 점수 벡터가 변경 없이 정책에 도달했는지 실행 중 확인한다.

## 6. 게이트와 Rerun

- `GATES.md`: **8/8 ALL MET**, abandoned 0
- 동일 시드 재실행: canonical payload 일치
- 보호 파일: heightmap·predictor·DEME·pellet·상태 원장·relay diff 0
- Rerun SDK/CLI: `0.34.1`
- RRD footer: PASS
- 비시스템 엔티티: 정확히 13개, 누락·초과 0
- 타임라인: 정확히 `blueprint`, `decision_step`, `log_time`
- RBL footer: PASS
- 헤드리스 스크린샷: PASS
- RRD SHA-256: `c9cbf6257ef5a01656d83cb2665abfddf3627c66969a309f8b438b5a3d81cffb`

RRD에는 매 결정 스텝의 높이 지도, 선택 위치, 제거 체적 중심, 행동 방향, 누적 적재량·시간·실패·이동거리·횟수, 정책 이벤트가 들어 있다. 육안검수에서는 3D 잔여 능선과 국부 함몰, 선택점과 제거 중심의 대응, 지표 누적과 정책 경계 리셋을 확인했다. 헤드리스 화면 우측 상단의 message-proxy 경고도 `visual_inspection.json`에 기록했으며, 기하·그래프·이벤트 행은 화면에 렌더됐다.

## 7. 여전히 판정할 수 없는 것

- DEME 입자 접촉·그랩 메시·고정 궤적의 실제 실행 결과
- 실물 펠릿 물성, 실물 적재량·시간·실패율
- 학습 데이터 미제공 상태의 모델 정책 성능과 위험 계수
- 고정 경로의 전 작업공간 도달성 및 실로봇 안전성
- 분석 실행기의 실패율 0을 실기 신뢰성으로 해석하는 것

이번 산출물은 루프·인터페이스·관측 가능성 검증까지이며, 학습·DEME 스쿱·로봇 제어는 실행하지 않았다.

## 8. 만든 파일

- `sim_scripts/p39_decision_loop.py`
- `claudedocs/runtime_logs/decision_track/p2b_decision_loop/GATES.md`
- `claudedocs/runtime_logs/decision_track/p2b_decision_loop/REPORT.md`
- `.../run_primary/summary.json`, `comparison.md`, `traces/*.json`
- `.../run_primary/timeline.rrd`, `timeline.rbl`, `rerun_validation.json`
- `.../run_primary/inspection.png`, `decision_snapshot.png`, `visual_inspection.json`
- `.../run_repeat/summary.json`, `comparison.md`, `traces/*.json`
