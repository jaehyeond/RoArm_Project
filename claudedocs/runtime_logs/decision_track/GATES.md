# GATES.md — decision_track 게이트 대장

이 폴더(`claudedocs/runtime_logs/decision_track/`)의 산출물이 **믿을 만한지**를
기계로 검사하는 규칙 목록이다. 사람이 눈으로 훑는 대신 명령 한 줄로 재검사한다.

> 범위: P41(오라클 n=30) 과 P42(관측 예산) 산출물. P2b 의 게이트는 별도 파일
> `p2b_decision_loop/GATES.md` 에 있으며 이 문서가 대체하지 않는다.

## 읽는 법

**게이트(gate)** = 산출물이 특정 조건을 만족하는지 프로그램이 판정하는 검사.
PASS/FAIL 두 값만 낸다.

게이트를 두 종류로 나눈다. 섞으면 "아직 주장이 안 선다"가 "코드가 깨졌다"처럼 보인다.

| kind | 뜻 | FAIL 이면 |
|---|---|---|
| **validity** | 실험이 망가지거나 조작되지 않았는가 | **결과를 인용하면 안 된다.** 고쳐서 다시 돌려야 한다 |
| **claim** | 프로포절의 주장이 이 데이터로 성립하는가 | 결과 자체다. 고칠 대상이 아니라 **보고할 사실**이다 |

`p42 gate` 의 종료코드는 **validity 만** 본다 (claim FAIL 은 고장이 아니므로).

**blind_spot** = 이 게이트가 **못 보는 것**. 게이트가 PASS 해도 여전히 틀릴 수 있는
방향을 적는다. 게이트 목록에서 가장 중요한 칸이다 — PASS 를 "검증됨"으로
읽지 않게 하는 유일한 장치다.

**회귀 주입(regression injection)** = 게이트가 실제로 FAIL 을 낼 수 있는지 확인하려고
일부러 산출물을 망가뜨려 보는 것. 한 번도 FAIL 을 못 낸 게이트는 게이트가 아니라
장식이다. 아래 모든 게이트에 대해 실제로 주입해서 FAIL 을 확인했고, 그 명령과
관측된 출력을 함께 적었다.

## 실행 환경

```bash
cd ~/orca/workspaces/RoArm_Project/decision-oracle       # 이 worktree
PY=/home/cgxr/miniconda3/envs/isaaclab/bin/python        # numpy 1.26.0 / scipy 1.15.3
```

## 게이트 일람

| ID | kind | 무엇을 지키는가 | 구현 |
|---|---|---|---|
| G1 | validity | k=1 관측 예산은 P41 오라클과 **완전히 같은 실험**이어야 한다 | `p42 gate` |
| G2 | claim | 각 k 에서 방법②(형상)가 방법①(양만)을 부호검정으로 이기는가 | `p42 gate` |
| G3 | validity | 세 정책이 실제로 **서로 다른 행동**을 하는가 | `p42 gate` |
| G4 | validity | 관측 예산 k 를 실제로 지켰는가 (몰래 다시 보지 않았는가) | `p42 gate` |
| G5 | validity | 세 정책이 **같은 목표질량**으로 비교됐는가 | `p42 gate` |
| G6 | claim | P41 n=30 에서 오라클이 greedy 를 이기는 것이 우연이 아닌가 | `p42 gate` |
| G7 | claim | P41 n=30 에서 **형상 예측의 추가 이득**이 우연이 아닌가 | `p42 gate` |
| G8 | validity | 정책이 후보를 못 찾아 죽은 행(`policy_error`)이 없는가 | `p42 gate` |

## 전체 게이트 통과 명령

```bash
$PY sim_scripts/p42_observation_budget.py gate \
  --p42  claudedocs/runtime_logs/decision_track/p42_observation_budget/p42_observation_budget.json \
  --p41  claudedocs/runtime_logs/decision_track/p41_oracle_n30/p41_oracle.json \
  --p41b-stats claudedocs/runtime_logs/decision_track/p41_oracle_n30/p41b_stats.json \
  --out  claudedocs/runtime_logs/decision_track/p42_observation_budget
```

---

## G1 — `G1_k1_reproduces_p41` (validity)

**판정 규칙**: p42 산출물의 `k=1` 행 전건이, 같은 (정책, 시드)의 P41 행과
`(total_scoops, failures, target_reached)` 3-튜플까지 일치해야 한다.
대조 가능한 k=1 행이 0건이면 게이트를 적용하지 못한 것이므로 **FAIL 처리**(fail-closed).

**왜 필요한가**: p42 는 p39 의 `run_episode` 를 쓰지 않는다. p39 는 매 스텝
**진짜 관측**을 정책에 넘기는데 관측 예산을 걸려면 바로 그 지점을 바꿔야 하고,
p39 는 수정 금지이기 때문이다. 그래서 p42 는 종료 계약(목표 적재 / 최대 시도 /
연속·누적 실패 한도 / 재료 고갈)을 **다시 구현**했다. 다시 구현한 루프가 원본과
같은 실험인지 보증하는 것이 이 게이트다. k=1 은 "매 스쿱마다 다시 본다" 이므로
정의상 P41 과 같은 조건이다.

**FAIL 회귀 주입**: belief 모형의 안식각을 4도 틀리게 준다. 그러면 오라클의
선행탐색이 다른 실행기 위에서 돌아가므로 k=1 이어도 P41 과 갈라져야 한다.

```bash
$PY sim_scripts/p42_observation_budget.py run \
  --out /tmp/gate_inject/g1_bias --n-seeds 3 --observe-every 1 --model-repose-bias-deg 4
$PY sim_scripts/p42_observation_budget.py gate \
  --p42 /tmp/gate_inject/g1_bias/p42_observation_budget.json \
  --p41 claudedocs/runtime_logs/decision_track/p41_oracle_n30/p41_oracle.json
```

**관측된 FAIL**:
```
[FAIL] (validity) G1_k1_reproduces_p41 — k=1 행 9건 대조, 불일치 3건.
  oracle_1/seed457: p42(17, 5, False) != p41(14, 0, True);
  oracle_2/seed458: p42(17, 5, False) != p41(14, 0, True);
  oracle_2/seed459: p42(10, 5, False) != p41(14, 0, True)
```

**blind_spot**
- **P41 자체가 틀렸으면 둘이 같이 틀린 채로 PASS 한다.** 이 게이트는 "두 구현이
  일치한다"만 보고 "물리가 맞다"는 전혀 보지 않는다. 대리모델의 붕괴 규칙이
  진짜 입자 거동과 다르다는 사실은 이 게이트로 절대 드러나지 않는다.
- 3-튜플만 본다. 같은 스쿱 수·같은 실패 수라도 **다른 자리를 팠을 수** 있다.
  행동 궤적(선택 xy, 방향)은 대조하지 않는다.
- `--model-repose-bias-deg` 를 0 이 아니게 돌린 산출물에는 애초에 적용되지 않는다.
  민감도 실행(bias ±4도) 산출물은 이 게이트의 보호를 받지 못한다.

---

## G2 — `G2_shape_gain_significant_k<k>` (claim)

**판정 규칙**: 각 k 에서 `oracle_2 > oracle_1` 부호검정의 단측 p < 0.05.
승패 규칙은 **완주 우선, 둘 다 완주면 스쿱 수가 적은 쪽 승, 같으면 무**
(`p41b_oracle_stats.sign_test`).

**부호검정이란**: 시드마다 두 정책을 짝지어 승/패/무를 세고, "동전 던지기라면
이만큼 이길 확률"을 이항분포로 직접 계산한다. 분포 가정이 필요 없어 스쿱 수처럼
분포를 모르는 정수 지표에 안전하다. 무승부는 세지 않는다(표준 규약).

**구조적 하한을 함께 보고한다**: 무승부를 뺀 유효 표본 `n_eff` 에서 **전승**이어도
p 는 `0.5^n_eff` 아래로 내려가지 않는다. `n_eff < 5` 면 어떤 짝지은 부호 기반
검정으로도 0.05 를 넘길 수 없다. 즉 이 경우의 FAIL 은 "효과가 없다"가 아니라
**"이 표본으로는 가를 수 없다"**는 뜻이다. 게이트 출력에
`best_possible_p_at_this_n_effective` 로 같이 찍힌다.

**FAIL 회귀 주입**: 시드를 3개로 줄여 유효 표본을 죽인다.

```bash
$PY sim_scripts/p42_observation_budget.py run \
  --out /tmp/gate_inject/g2_lowN --n-seeds 3 --observe-every 1 inf
$PY sim_scripts/p42_observation_budget.py gate \
  --p42 /tmp/gate_inject/g2_lowN/p42_observation_budget.json
```

**관측된 FAIL**:
```
[FAIL] (claim) G2_shape_gain_significant_k1   — 승1 패0 무2 n_eff=1 p=0.5   (최소 가능 p=0.5)
[FAIL] (claim) G2_shape_gain_significant_kinf — 승2 패0 무1 n_eff=2 p=0.25  (최소 가능 p=0.25)
```

**blind_spot**
- **대리모델 안에서의 유의성이다.** 실물에서 재현된다는 보증이 전혀 없다.
- 시드 30개는 같은 능선 생성기(`p39.synthetic_pile`)의 미세 변형이다
  (중심 ±2mm, ±4mm 흔들기). **더미 개수·형상 종류·용기 위치 같은 장면 다양성을
  재는 표본이 아니다.** 따라서 p 값은 "이 더미 계열 안에서의 유의성"이다.
- 무승부가 많으면 효과가 실재해도 검정력이 먼저 죽는다. FAIL 을 "형상 예측 무용"으로
  읽으면 안 된다.
- 승패 규칙이 **완주를 스쿱 수보다 앞에** 둔다. 완주는 했지만 시간·이동거리가 나쁜
  정책을 이 게이트는 벌하지 않는다.

---

## G3 — `G3_policies_are_distinct` (validity)

**판정 규칙**: 같은 (k, seed) 칸에서 세 정책의 스쿱 수가 **완전히 같은 칸**이
전체 칸 수보다 적어야 한다. 전 칸이 같으면 같은 정책을 세 번 돌린 것이다.

**왜 필요한가**: 정책 분기를 잘못 배선해 셋 다 같은 코드를 타면, 표는 멀쩡해 보이는데
비교는 아무 의미가 없다. 실제로 리팩터링 중 가장 조용히 발생하는 사고다.

**FAIL 회귀 주입**: 세 정책을 전부 greedy_high 로 강제한다.

```bash
$PY sim_scripts/p42_observation_budget.py run \
  --out /tmp/gate_inject/g3_identical --n-seeds 3 --observe-every 3 --force-identical-policies
$PY sim_scripts/p42_observation_budget.py gate \
  --p42 /tmp/gate_inject/g3_identical/p42_observation_budget.json
```

**관측된 FAIL**:
```
[FAIL] (validity) G3_policies_are_distinct — 동일 (k,seed) 격자 3칸 중
  세 정책의 스쿱 수가 완전히 같은 칸 3칸.
```

**blind_spot**
- **스쿱 수만 본다.** 서로 다른 자리를 골랐는데 우연히 스쿱 수가 같은 경우와,
  정말 같은 정책인 경우를 구분하지 못한다. 행동 궤적을 비교하지 않는다.
- 반대로 "충분히 다르다"도 못 본다. 두 정책이 거의 같고 한 칸만 달라도 PASS 한다.
- 정책이 서로 다르다는 것과 **의미 있게 다르다**는 것은 별개다.

---

## G4 — `G4_observation_budget_respected` (validity)

**판정 규칙**: 모든 행에서 기록된 실제 재관측 횟수 `observations_used`
== `ceil(total_scoops / k)`.

**왜 필요한가**: 이 실험의 전부가 "관측을 못 하게 막았다"는 전제다. 정책이 어딘가에서
진짜 상태를 다시 들여다보면 실험이 통째로 무효다. 카운터로 그것을 막는다.

**FAIL 회귀 주입**: 예산을 무시하고 매 스텝 재관측하게 만든다.

```bash
$PY sim_scripts/p42_observation_budget.py run \
  --out /tmp/gate_inject/g4_leak --n-seeds 3 --observe-every 3 --leak-true-observation
$PY sim_scripts/p42_observation_budget.py gate \
  --p42 /tmp/gate_inject/g4_leak/p42_observation_budget.json
```

**관측된 FAIL**:
```
[FAIL] (validity) G4_observation_budget_respected — 전 9행 중 관측 횟수 != ceil(스쿱/k) 인 행 9건.
  greedy_high/k=3/seed457: 21 != 7; oracle_1/k=3/seed457: 14 != 5; oracle_2/k=3/seed457: 14 != 5
```

**blind_spot**
- **횟수만 센다.** belief 배열이 **어떤 경로로** 진짜 상태와 같아졌는지는 못 본다.
  `--model-repose-bias-deg 0` 이면 oracle_2 의 belief 는 예측 모형이 곧 실행기라서
  언제나 진짜 상태와 일치하지만, 관측 카운트는 정상이므로 이 게이트는 **PASS 한다.**
  bias=0 의 oracle_2 곡선이 k 에 둔감한 것은 이 사각지대의 직접적 결과다.
- 실패 피드백(`FailingExecutor.failure_log`)은 관측으로 세지 않는다. "빈 채로
  올라왔다"는 하중 신호는 히트맵 재관측이 아니라고 본 설계 판단이며, 이 게이트가
  검증하는 대상이 아니다.
- 관측의 **비용**(스캔 시간, 버킷 가림, 분진)은 모델에도 게이트에도 없다.

---

## G5 — `G5_comparison_is_symmetric` (validity)

**판정 규칙**: 산출물 전 행의 `target_mass_kg` 값 집합의 크기가 1.
한 정책에만 다른 목표를 줬으면 FAIL.

**왜 필요한가**: "우리 방법이 이겼다"의 가장 흔한 사고 원인은 물리가 아니라
**설정 비대칭**이다. 한쪽에만 쉬운 목표를 주면 어떤 표도 이긴다.

**FAIL 회귀 주입**: greedy_high 에게만 목표질량 90% 를 준다.

```bash
$PY sim_scripts/p42_observation_budget.py run \
  --out /tmp/gate_inject/g5_unfair --n-seeds 3 --observe-every 3 --unfair-target-scale 0.9
$PY sim_scripts/p42_observation_budget.py gate \
  --p42 /tmp/gate_inject/g5_unfair/p42_observation_budget.json
```

**관측된 FAIL**:
```
[FAIL] (validity) G5_comparison_is_symmetric — 정책별 목표질량 집합
  {'greedy_high': {0.09}, 'oracle_1': {0.1}, 'oracle_2': {0.1}}.
```

**blind_spot**
- **목표질량 한 축만 본다.** 실패 모델 파라미터(`min_fill_fraction`,
  `avalanche_margin_deg`), 초기 더미, `top_k`, `dug_radius_m` 처럼 정책마다 따로
  줄 수 있는 다른 축은 검사하지 않는다.
- **"공정한 설정" ≠ "현실적인 설정".** 세 정책에 똑같이 비현실적인 조건을 줘도 PASS 한다.
- greedy_high 에만 붙은 실패-셀 회피(`avoid_radius_m`)와, 형상 예측이 없는 정책에만
  붙은 dug 기억은 **의도된 비대칭**이다. 이 게이트는 그 비대칭을 잡지 않는다
  (잡으면 안 된다 — 없으면 baseline 이 자멸해 비교가 조작이 되므로).

---

## G6 / G7 — P41 n=30 통계 게이트 (claim)

**판정 규칙**
- **G6** `G6_oracle2_beats_greedy` : `oracle_2 > greedy_high` 부호검정 p < 0.05
- **G7** `G7_shape_beats_mass_only` : `oracle_2 > oracle_1` 부호검정 p < 0.05

입력은 `p41b_stats.json` (= `p41b_oracle_stats.py` 산출물).

**왜 나눴는가**: G6 은 "예측이 교수님 안을 이기는가"(기여 1), G7 은 "**형상까지**
예측하는 것이 양만 예측하는 것보다 낫는가"(기여 2)다. 프로포절에서 값이 다른 두
주장이고, 실제로 **결과도 다르게 나온다.**

**FAIL 회귀 주입**: 시드를 앞 3개만 써서 유효 표본을 죽인다.

```bash
$PY sim_scripts/p41b_oracle_stats.py \
  --input claudedocs/runtime_logs/decision_track/p41_oracle_n30/p41_oracle.json \
  --out /tmp/gate_inject/g6_subsample3 --subsample 3
$PY sim_scripts/p42_observation_budget.py gate \
  --p42 claudedocs/runtime_logs/decision_track/p42_observation_budget/p42_observation_budget.json \
  --p41b-stats /tmp/gate_inject/g6_subsample3/p41b_stats.json
```

**관측된 FAIL**: (아래 "현재 판정" 절에 실제 출력)

**blind_spot**
- G2 의 blind_spot 전부가 그대로 적용된다(대리모델 한정 유의성, 생성기 단일성,
  검정력 소멸, 완주 우선 순위).
- **시드 수를 사후에 늘려 p 를 낮추는 것을 막지 못한다.** 이 게이트는 주어진
  산출물만 본다. p-해킹 방지는 게이트가 아니라 규약으로 지킨다 —
  시드 30개(457~486)는 실행 **전에** 고정했고, 늘리려면 목표 n 을 먼저 선언하고
  전량을 새로 돌려 그 결과만 보고해야 한다.
- 단측 검정이다. "oracle_2 가 더 나쁠 가능성"은 이 게이트가 재지 않는다
  (참고용 양측 p 는 `p41b_stats.json` 에 함께 기록).

---

## G8 — `G8_no_policy_error_rows` (validity)

**판정 규칙**: 산출물 어느 행도 `terminal_reason` 이 `policy_error` 로 시작하지 않아야 한다.

**왜 필요한가**: 정책은 belief 위에서 후보 셀을 고르는데, "이미 판 자리"(dug) 제외
마스크와 "실패한 자리" 제외 마스크가 활성 셀을 **전부** 덮으면 `_active_cells` 가
비고 정책이 예외로 죽는다. 그 행은 `target_reached=False` 로 남아 겉보기에는
"정책이 목표를 못 채웠다"처럼 보이지만, 실제로는 **물리가 아니라 마스킹 아티팩트**다.
이걸 못 걸러 내면 "이 조건에서도 기준선이 무너진다"는 **거짓 강건성 주장**이 나온다.

**FAIL 회귀 주입**: dug 제외 반경을 발자국 전폭(58mm)으로 키우고 k=inf 로 돌린다.
마스크가 누적되어 지도를 통째로 덮는다.

```bash
$PY sim_scripts/p42_observation_budget.py run \
  --out /tmp/gate_inject/g8_dug058 --n-seeds 3 --observe-every inf --dug-radius-m 0.058
$PY sim_scripts/p42_observation_budget.py gate \
  --p42 /tmp/gate_inject/g8_dug058/p42_observation_budget.json
```

**관측된 FAIL** (실제 산출물 `p42_dugradius_058` 에서, n=30·k∈{1,inf}):
```
[FAIL] (validity) G8_no_policy_error_rows — 전 180행 중 policy_error 로 끝난 행 60건
  (정책별 {'greedy_high': 30, 'oracle_1': 30}).
```
→ 이 게이트가 `p42_dugradius_058` 실행을 실제로 무효 판정했다. 게이트가 없었다면
"58mm 에서도 greedy 완주율 0.000" 을 강건성 근거로 보고할 뻔했다.
`oracle_2` 는 dug 목록을 쓰지 않으므로 30행 모두 정상이다.

**알려진 원인**: `p42_observation_budget.py` 의 폴백이 **실패-셀 마스크에만** 걸려 있고
**dug 마스크에는 없다**. 3줄 수정이면 되지만, 실행 중이던 실험의 의미를 바꾸지 않으려고
이번에는 고치지 않았다 — 후속 작업 항목.

**blind_spot**
- **`policy_error` 만 센다.** 마스킹이 지도를 **거의** 다 덮어 정책이 억지로 나쁜 칸을
  고른 경우(예외는 안 남)는 잡지 못한다. **PASS 라도 dug 반경이 결과를 왜곡하지
  않았다는 보증이 아니다** — 반경 민감도를 따로 돌려 봐야 한다.
- 예외 문자열만 본다. 어떤 정책이 왜 후보를 잃었는지, 몇 스텝째였는지는 기록하지 않는다.
- `p42` 는 실패 **사유**(`avalanche` / `empty_grab`) 분포를 행에 남기지 않는다.
  따라서 이 게이트도, 다른 어떤 게이트도 "무엇에 걸려 죽었는지"를 검증하지 못한다.

---

## 게이트 전체가 못 보는 것 (공통 blind spot)

게이트를 전부 PASS 시켜도 다음은 여전히 검증되지 않는다. 이 목록을 빼고
"게이트 통과"를 보고하면 그 보고가 거짓이 된다.

1. **대리모델의 물리.** `AnalyticExecutor`/`FailingExecutor` 의 붕괴 처리는 진짜 입자
   물리가 아니다. 안식각을 넘은 이웃 쌍의 높이 차를 정해진 비율로 나눠 갖는 기하
   완화 규칙일 뿐이고, 실제 사면 붕괴·아칭·다짐·입자 분리와 대응하지 않는다.
   **최종 확인은 DEME 시뮬레이션과 실물 스쿱이 한다.**
2. **오라클은 학습이 아니라 커닝이다.** 정책이 후보의 결과를 실행기 사본으로 미리
   본다. 따라서 oracle_1/oracle_2 의 모든 수치는 "학습이 이렇게 된다"가 아니라
   **학습이 도달할 수 있는 상한**이다. 실제 학습 정책은 반드시 이보다 나쁘다.
   게이트는 이 상한을 낮은 값으로 검증하지 못한다.
3. **PP 물성 미실측.** `angle_of_repose_deg = 32.0` 은 임시값이고 `--model-repose-bias-deg`
   의 ±4도 역시 임의값이다. **어떤 수치도 폴리프로필렌 값으로 인용해서는 안 된다.**
   실측이 들어오면 P40/P41/P42 결과가 전부 바뀔 수 있다.
4. **장면 다양성.** 시드는 단일 능선 생성기의 미세 변형이다. 더미 개수, 형상 종류,
   용기 위치, 조명, 센서 잡음은 전혀 흔들지 않았다.
5. **관측 비용의 부재.** k 는 재관측 **횟수**만 제한한다. 실제 크레인에서 재관측이
   비싼 이유(스캔 시간, 버킷 가림, 분진, 운전 리듬)는 모델에 없고, 시간 지표에
   관측 비용이 들어 있지 않다.
6. **하드웨어 실패 양식.** 도구 충돌, 입자 끼임, 서보 스톨, 그랩 미폐쇄 같은 실물
   실패는 실패 모델에 없다. heightmap 기하로 판정 가능한 두 가지(붕괴·빈그랩)뿐이다.
7. **엔드이펙터 부재.** 현재 인벤토리에 스쿱/그랩·입자 재료·배출 용기가 없다.
   이 게이트들이 전부 PASS 해도 실물에서 한 번도 퍼 본 적이 없다는 사실은 변하지 않는다.

---

## 현재 판정 (2026-09-01)

(이 절은 산출물이 갱신될 때마다 함께 갱신한다. 아래 수치의 출처는 각 행의 경로다.)

### 본 산출물 — `p42_observation_budget/` (bias 0, k=1/3/5/inf, n=30, 360행)

판정 파일: `claudedocs/runtime_logs/decision_track/p42_observation_budget/p42_gate_report.json`

```
[PASS] (validity) G1_k1_reproduces_p41              k=1 행 90건 대조, 불일치 0건
[FAIL] (claim   ) G2_shape_gain_significant_k1      승4 패0 무26 n_eff=4  p=0.0625
[PASS] (claim   ) G2_shape_gain_significant_k3      승21 패0 무9  n_eff=21 p=4.768e-07
[PASS] (claim   ) G2_shape_gain_significant_k5      승23 패0 무7  n_eff=23 p=1.192e-07
[PASS] (claim   ) G2_shape_gain_significant_kinf    승25 패0 무5  n_eff=25 p=2.980e-08
[PASS] (validity) G3_policies_are_distinct          120칸 중 세 정책 동일 0칸
[PASS] (validity) G4_observation_budget_respected   360행 중 위반 0건
[PASS] (validity) G5_comparison_is_symmetric        목표질량 전부 0.1 kg
[PASS] (claim   ) G6_oracle2_beats_greedy           승30 패0 무0 p=9.313e-10
[FAIL] (claim   ) G7_shape_beats_mass_only          승4 패0 무26 p=0.0625
[PASS] (validity) G8_no_policy_error_rows           360행 중 policy_error 0건
VALIDITY_ALL_PASS=True   CLAIM_PASS=4/6
```

**validity 8종 전부 PASS** → 이 산출물의 결과는 인용해도 된다.

**claim 2종 FAIL 은 같은 사실 하나다**: **k=1(매 스쿱 재관측)에서는 형상 예측의
*추가* 이득이 n=30 으로 통계적으로 잡히지 않는다** (4승 0패 26무, p=0.0625).
유효 표본이 4라 **전승이어도 0.0625 아래로 내려갈 수 없다** — 검정을 바꿔서 해결되는
문제가 아니라 표본 문제다. 같은 불일치율(약 13%)이 이어진다면 시드 약 38개가 필요하다.
**시드를 사후에 늘려 다시 계산하지 않았다**(p-해킹 방지).
같은 비교가 **k≥3 에서는 전부 PASS** 로 뒤집힌다 (p=4.8e-07 이하).

### 민감도 산출물

| 산출물 | 설정 | validity | claim (G2) |
|---|---|---|---|
| `p42_observation_budget_bias_minus4/` | belief 안식각 −4도 | 전부 PASS | k=1 ❌(p=0.0547) · k=3/5/inf ✅ |
| `p42_observation_budget_bias_plus4/` | belief 안식각 +4도 | 전부 PASS | k=1 ❌(p=0.867, 5승 8패로 역전) · k=3/5/inf ✅ |
| `p42_dugradius_008/` | dug 반경 8mm | 전부 PASS | k=1 ❌ · kinf ✅ |
| `p42_dugradius_058/` | dug 반경 58mm | **G8 FAIL** (180행 중 60행 `policy_error`) | — |

🔴 **`p42_dugradius_058` 의 k=inf 행은 무효다.** 마스킹 아티팩트이므로 이 실행을
근거로 아무것도 주장하지 않는다. 나머지 네 산출물은 validity 전부 PASS 다.

### P41 n=30 통계 산출물 — `p41_oracle_n30/`

| 게이트 | 결과 |
|---|---|
| G6 `oracle_2` > `greedy_high` | **PASS** — 30승 0패 0무, p=9.313e-10 |
| G7 `oracle_2` > `oracle_1` | **FAIL** — 4승 0패 26무, p=0.0625 (n_eff=4, 하한 0.0625) |

출처: `claudedocs/runtime_logs/decision_track/p41_oracle_n30/p41b_stats.json` / `.md`
