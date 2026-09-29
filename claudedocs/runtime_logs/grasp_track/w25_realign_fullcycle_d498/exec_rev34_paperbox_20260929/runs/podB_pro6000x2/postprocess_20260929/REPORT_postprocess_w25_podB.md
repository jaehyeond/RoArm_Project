# W25-A podB 원자료 회계 재계산 보고 (raw-accountant, 1단계)

- 대상 원자료: `…/exec_rev34_paperbox_20260929/runs/podB_pro6000x2/run_01/` (14 파일, NPZ 1.77 GB)
- 출력 폴더: `…/runs/podB_pro6000x2/postprocess_20260929/`
- 실행 조건: **CPU 전용, 새 물리 0, GPU/DEME/Isaac/Rerun 0, 원자료 쓰기 0**
- 이 단계의 실제 경과 시간(wall-clock): 약 32분 (2026-09-29 16:47:53 KST 시작 → 17:20 KST)
- 결론 범위: **생산 회계 재현 여부 + 항목별 PASS/FAIL 까지.** "전체 사이클 성공" 선언은 하지 않는다(D490 `claudedocs/DECISIONS_ACTIVE.md:316`).

용어: **실제 경과 시간(wall-clock)** = 시계로 잰 시간. **파생(derived)** = 원자료를 고치지 않고 다시 계산해 만든 값.
**규약(contract)** = 라벨·전환·정착을 어떻게 세는지 미리 적어 둔 문서. **cadence** = 저장 프레임이 얼마나 촘촘한가.

---

## 0. 한눈에 보는 결과

| 항목 | 결과 |
|---|---|
| P1 보존(원자료 무수정) | **PASS** — 14/14 sha256 이 회수 영수증과 일치, 작업 전·후 동일(`diff PRE POST` 0줄) |
| P2 생산식 재분류 재현 | **PASS** — 18,627,675 라벨 칸 중 **불일치 0**, 클래스별 전부 0 |
| P3 독립 검사기 재현 | **PASS** — 같은 18,627,675 칸 **불일치 0**, unittest 4/4 |
| P4 정착 창 재계산 | **6항목 중 1 FAIL** — cadence 계약(≥6프레임·≤0.05 s) **미충족** |
| P5 원자료 스키마 27항목 | **22 PASS / 5 FAIL** |
| P6 등록 규약 45항목 | **35 PASS / 2 FAIL / 8 판정불가**, 그중 hard_fail FAIL 1 |
| P7 W25 전용 관측 | 채터링 3회 후 `retries_exhausted`, 흘림 25알 전부 transport 단계 |
| P8 W19 비교 | 수치 병기함. **"같은 입력·다른 장비" 짝은 비교 불가**(podA 원자료 없음) |

---

## 1. P1 보존 영수증

| 시점 | 파일 수 | 영수증 불일치 | 판정 | 파일 |
|---|---|---|---|---|
| 작업 전 | 14 | 0 | PASS | `PRESERVATION_BEFORE.json` |
| 작업 후 | 14 | 0 | PASS | `PRESERVATION_AFTER.json` |

`logs/PRE.txt` 와 `logs/POST.txt` 의 `diff` 가 0줄이다. NPZ `af313f05…5032`, JSON `72d238fe…0aa1`, timeline `e15c1e7c…fd1b2` 를 포함해 14개 전부 `RETRIEVAL_RECEIPT.json` 값과 같다.

## 2. 식 무수정 증거 (rev34_copy)

`rev34_copy/REVISION_PIN.json` — 51개 파일 사본 전부 원본 sha 와 일치(`n_bad: 0`), `EXEC_PIN.json` 기록 sha 와도 일치.

- `rev34/src/inventory_geometry.py` = `a49e1ba9…0327` = **W14 rev31 원본과 바이트 동일** → 분류식 정본이 W19 때와 같은 파일이다.
- 메인 repo `sim_deme_scoop_s1.py` = `2e40f7ed…7933` = W14 핀과 일치 → 구 전개식도 동일.
- 식은 한 줄도 고치지 않았다. 경로 문제는 **래퍼**(`tools/derive_w25.py`)로만 우회했고, 동결 파일 편집 0건이라 `DIFF_paths_only.patch` 는 필요 없었다.

### 2-1. 동결 `derive_v2.py` 를 그대로 돌렸을 때 (기록용)

`rev34_copy/rev34/src/derive_v2.py` 를 바이트 사본 그대로 두 번 실행했고, 두 번 다 **rc=1** 로 멈췄다. 로그: `derived_v2_w25/frozen_tool_attempt/`.

1. `FileNotFoundError` — `--rev29-derived` 기본값이 W13 전용 경로(`…/rev34_copy/derived/w13_cycle_seed460_rev29_derived.npz`)라 W25 에는 그 파일이 없다.
2. 존재하는 파일을 넘겨 그 지점을 지나가게 하자 `SystemExit: 입력 해시 불일치 raw` — `derive_repaired_raw.py:37-43` 의 `EXPECTED_SHA256["raw"] = 529f422e…6b0f`(W13 원자료)와 W25 원자료 `af313f05…5032` 가 다르기 때문이다.

즉 동결 드라이버는 **W13 입력에 고정**돼 있다. 그래서 식(분류·전개·전환 함수)은 사본 모듈에서 그대로 import 하고, 경로·해시·메모리 처리만 하는 래퍼를 썼다(W19 후처리 `tools/derive_w19.py` 와 같은 방식).

## 3. P2 — 전 프레임 재분류 vs 생산 기록

- 프레임 275 × 입자 67,737 = **라벨 칸 18,627,675**
- **rev34(=rev31 식, ERRATUM_04 받침면) 재계산 vs 원자료 기록: 불일치 0**
- 클래스별 불일치도 전부 0 (`source/receiving_bin/tool_residual/spill/in_flight/ambiguous` 모두 0)
- 혼동행렬이 완전 대각선이다: 대각 `[17,492,380 / 58,143 / 35,290 / 3,091 / 3,660 / 1,035,111]`, 비대각 0
- 생산 JSON `decisions[*].counts` 14개 태그 전부가 원자료 라벨 bincount 와 동일, 각 태그의 재계산 불일치도 0
- `delivery.inventory_final`(JSON) == 원자료 마지막 프레임 bincount == 재계산 값:
  `source 63,402 / receiving_bin 585 / tool_residual 14 / spill 25 / in_flight 0 / ambiguous 3,711` (합 67,737)
- 전환 인덱스: 기록 `[1, 26, 4681, 4970, 12468, 12643, 13052, 14219, 16780, 17155, 20774]` == 규약(phase-only) 재계산값. rev28 legacy 규칙으로 재현하면 34개가 나와 기록과 다르다(= 기록이 규약 쪽을 따랐다는 확인).
- 대조용 **rev29 strict floor** 로 다시 세면 1,900,749 칸이 달라진다(주로 `source → ambiguous` 1,861,767). 이는 규약 판(받침면 vs 엄격 하한)의 차이이지 오류가 아니다.
- 소요: 66.3 s. 산출: `derived_v2_w25/w25_podB_seed460_rev34_derived.npz` + `DERIVED_V2_MANIFEST.json`

> **비주장**: 재현 0 불일치는 "기록이 규약대로 계산됐다"는 뜻이다. 배출량이 옳다/성공했다는 뜻이 아니다.

## 4. P3 — 독립 검사기

`tests/independent_check_v2.py`(sha `bfd51c11…9f4c`, import = `itertools`·`math`·`numpy` 뿐)는 생산 모듈(`inventory_geometry`, `sim_deme_scoop_s1`, `w13_*`)·`scipy` 를 **전혀 import 하지 않고** 규약 문구에서 다르게 구현한 것이다(회전 = Hamilton 곱, 봉쇄 = 축정렬 경계상자, 전환 = `groupby`). W19 후처리에서 쓰던 파일의 바이트 사본이라 W25 결과를 보고 맞춰 쓸 수 없었다.

| 검사 | 결과 |
|---|---|
| 전 275프레임 생산식 vs 독립식 | **불일치 0 칸 / 18,627,675 칸** |
| 전환 인덱스 독립 재계산 | 기록과 동일, ERRATUM_04 §2 경계 프레임 `i-1` 누락 0 |
| 원자료·파생 sha | 회수 영수증·manifest 와 일치 |
| `decisions[*].counts` JSON vs 원자료 | 불일치 0 |
| unittest | `Ran 4 tests … OK` (91.8 s) |

결과: `tests/RESULTS_allframes_w25.json`, 로그 `tests/allframes.stderr.txt`.

## 5. P4 — 정착 창 재계산 (**FAIL 1**)

동결 파라미터: `settlement_window_s = 0.25`, `settlement_frame_dt_s = 0.05`, `particle_frame_dt_s = 0.1`,
`settle_speed_max_m_s = 0.005`, `settle_move_max_m = 0.001`.

| 항목 | 판정 | 수치 |
|---|---|---|
| 창 재계산 == 생산 기록 | PASS | 9개 키 전부 일치 |
| **cadence 계약 (≥6프레임 · ≤0.05 s)** | **FAIL** | 프레임 **5개**, 최대 간격 **0.100025 s** |
| 확정 배출 재계산 == 생산 | PASS | 585 |
| 가능 배출 재계산 == 생산 | PASS | 665 (585 + 밴드 안 80 + 테두리 위 0) |
| `exact_single_value_allowed` 플래그 | PASS | `false` (585 ≠ 665 와 일치) |
| 기록 라벨 기준 vs rev34 재계산 라벨 기준 | PASS | 둘 다 stable 585 / settled 585 |

창 프레임 시각: `22.702677 / 22.802702 / 22.802702 / 22.902727 / 22.902727 s`
(같은 sync 의 결정 스냅샷 때문에 시각이 두 쌍 겹친다 → 실질 서로 다른 시각은 3개, 간격 0.100025 s).

창 안 안정 입자의 최대 속력 0.000383 m/s(판정선 0.005), 최대 중심 이동 0.0104 mm(판정선 1 mm)로 **여유는 크다**.
그러나 **저장 간격이 0.1 s 라 0.05 s 계약을 채울 수 없다** — 여유가 크다는 사실이 cadence 계약 충족을 대신하지 않는다. **사후 허용값을 만들지 않았고, FAIL 로 그대로 둔다**(D485 `:309`).

> **두 층 분리**: 이것은 *기록 계약*의 한계다(저장 간격 0.1 s). *물리 실패*가 아니다.
> 그래서 배출량은 **585~665알(11.85~13.47 g) 구간**으로만 읽고, "585알 확정" 같은 단일값 인용은 금지된다(`exact_single_value_allowed: false`).

산출: `derived_v2_w25/SETTLEMENT_WINDOW_RECOMPUTE.json`.

## 6. P5 — 원자료 스키마 27항목 (22 PASS / **5 FAIL**)

`schema_check/RAW_SCHEMA_CHECK_W25.json` · verdict `RAW_SCHEMA_CHECK_FAIL_5_OF_27`

PASS 쪽 주요 확인: dense 배열 22,128행 17종 모양/dtype, sparse 275×67,737 배열, `inventory_labels` 정확 일치,
metadata 필수 28키(**`w25_frame` 4개 하위키 전부 포함**), ERRATUM_03 행 정체성(`particle_frame_row == arange`, sync 인덱스 비감소, 중복 6개),
ERRATUM_04 §2 경계 프레임, 전환 인덱스, `sync_t_s` 엄격 증가 + 입자 프레임 시각 오차 0.0,
접촉 49,648,108행(역할 매핑 `fixed:0/door:1/tray:2/bin:3`, 인덱스 범위 정상),
ERRATUM_02 용기 반경 의미(`circumradius`, apothem = circumradius·cos(π/48) 오차 <1e-15),
공구 공동·소스 경계·임계 선언, 설치 수치 입력 11키, 정준 템플릿·질량, 속도 5/20 심각도, rc0↔stderr 대조, 재시도 0,
사전 해시 영수증(선언 15개 재계산 불일치 0; 사전 검증 606건 불일치 0), EXEC_PIN 바이트 동일 77/77, 재고 전수·배타.

### FAIL 5건

| # | 항목 | 실측 | 비고 |
|---|---|---|---|
| 1 | `metadata_time_mapping_abs_s_and_geometry_epsilon_m` | 두 키 **모두 없음** | 규약 `RAW_SCHEMA_REQUIRED.md:71-72`. W19 A 도 같은 FAIL |
| 2 | `erratum04_floor_rule_and_contract_version_declared_in_raw_metadata` | `classify_floor_rule`·`classify_contract_version`·`classify_source_floor_rule`·`classify_revision`·`classify_predicate_module` **5개 전부 `null`** | 규약 `ERRATUM_04.md:37-38`. **라벨은 ERRATUM_04 식으로 계산됐는데(P2/P3 확인) 원자료 메타데이터가 그 규약을 선언하지 않는다.** ERRATUM_04 는 `:10-11` 에서 "명시적으로 선언한 revision 에만 효력"이라 적혀 있다 |
| 3 | `erratum01_visual_mapping_one_row_per_saved_particle_frame` | `visual_mapping.json` **없음** | 재생(2단계) 산출물. 저장 프레임이 12개 단계를 전부 덮는 것은 확인됨. 없는 것은 FAIL 로 남김 |
| 4 | `final_manifest_path_bytes_sha256_for_every_output` | `MANIFEST.json`/`FINAL_MANIFEST.json` **없음** | 회수 영수증에는 sha 만 있고 byte count 가 없다. W19 도 같은 FAIL |
| 5 | `delivery_layers_separate_and_settlement_cadence_contract` | 5프레임·0.100025 s | P4 와 같은 사실 |

## 7. P6 — 등록 규약(criteria) 45항목 (35 PASS / **2 FAIL** / 8 판정불가)

`schema_check/CRITERIA_CHECK_W25.json` · 규약 파일 `criteria_w25_paperbox_cap32h.json` (sha `ba46b045…f15d`, `frozen_before_production: true`)

### 상한 관련 (요청 항목)

| 항목 | 규약값 | 실측 | 판정 |
|---|---|---|---|
| `runner.physics_wall_cap_s` | 115,200 s | 영수증 cap 115,200 s / 실제 **36,372.797 s** (31.57 %, 여유 78,827.2 s) | PASS |
| `runner.sim_soft_wall_cap_s` | 114,000 s | argv 114,000 s / 시뮬 내부 36,311.69 s, `abort_class: null` | PASS |
| `runner.graceful_grace_s` | 1,200 s | 영수증 1,200 s, `killed: false` | PASS |
| `runner.no_retry` | 재시도 0 | `auto_retry: false`, 시도 폴더 `run_01` 하나 | PASS |
| `process.rc0_is_not_delivery` | rc0 ≠ 배출 판정 | rc 0, `abort_class: null`, `exact_single_value_allowed: false` | PASS |

### FAIL 2건

1. **`delivery.settlement_window_UNCALIBRATED`** (severity `uncalibrated_report_only`) — 5프레임·0.100025 s 로 cadence 미충족. 이 severity 는 토큰 발급을 막지 않는 **보고 대상**이다(규약 `reading_rules[0]`). 그래도 사후 완화 없이 FAIL 로 적는다.
2. **`policy.no_threshold_change_after_outcomes`** (severity **`hard_fail`**) — 규약 문구는 "criteria.json is frozen before production **and its SHA256 is recorded in the execution receipt**". 실측:
   - criteria sha `ba46b045…f15d` 는 `EXEC_PIN.json`(GPU 결과 0 시점 동결본)에 기록돼 있다.
   - 그러나 `run_01/EXECUTION_RECEIPT.json` 이 담는 sha 는 `commands_json_sha256`·`runner_self_sha256` 둘뿐이고, 그 `COMMANDS_w25_podB_pro6000x2.json` 도 criteria **경로**만 적고 sha 는 적지 않는다.
   - 동결 시점 자체는 앞선다: criteria `changed_utc 2026-09-28T20:53:49Z` < EXEC_PIN `20:57:35Z` < 실행 시작 `21:15:46Z`.
   - **문구를 글자대로 읽어 FAIL 로 남긴다.** 기록 위치가 영수증이 아니라 상위 핀 파일이라는 **관측**이며, 임계가 결과를 보고 바뀌었다는 주장은 하지 않는다.

### 판정불가 8건 (PASS 아님)

`visual.production_maps_every_particle_frame`, `visual.rrd_counts_logged_rows_not_totals`,
`visual.aborted_certificate_no_invented_pass`, `visual.actual_tool_geometry_required`,
`visual.rerun_contract`, `visual.isaac_frame_mapping`, `readiness.renderer_bounds`,
`negative_controls.required` — 재생(RRD/Isaac)·readiness·음성대조 산출물이 필요하다. 이 회계 단계의 산출 경계 밖이다.

> bridge 계열 6건은 전부 PASS: `numeric_epsilon_m`(1e-06, 최악 여유 0.0628 m),
> `orthonormal_tol`(1e-09), `max_align_rotation_deg`(limit_failures 0, 최악 회전 0.000237 rad),
> `plan_match_tol`(계획-목표 편차 0.0 m, 220/220 소비, 실패 0), `elapsed_upper_bound`(누산 0.004000999989898 ≤ 상한 0.004000999999997),
> `observed_sync_overrun`(관측 전용, 최대 |Δt−요청| 1.0e-06 s). `visual.door_mm_per_deg` 도 PASS(2.1845 / 2.0045 mm/deg, 금지값 7.7822 아님).

## 8. P7 — W25 전용 관측 (원시 사실만)

`W25_RAW_OBSERVATIONS.json`

### 8-1. 채터링(실물 절차 재현)
파라미터: 임계 서보 3.6°, 재열기 서보 8.0°(관절 5.5°), 최대 3회. 단위: `servo_deg = joint_deg + 2.5`.

| 회 | 읽은 값 (관절/서보) | 재열기 정지 (관절/서보, 사유) | 재닫기 정지 (관절/서보, 사유) | 힌지 모멘트 |
|---|---|---|---|---|
| 0 | 3.4047 / 5.9047 | 5.5 / 8.0, `reached_open_end` | **3.5875 / 6.0875**, `servo_stall` | 1.7795 N·m |
| 1 | 3.5874 / 6.0874 | 5.5 / 8.0, `reached_open_end` | **3.4885 / 5.9885**, `servo_stall` | 1.7778 N·m |
| 2 | 3.4884 / 5.9884 | 5.5 / 8.0, `reached_open_end` | **3.3107 / 5.8107**, `servo_stall` | 1.7674 N·m |
| 3 | 3.3108 / 5.8108 | — | — (`retries_exhausted`) | — |

세 번 모두 임계 3.6° 아래로 내려가지 못했고 마지막 행은 `retries_exhausted` 다. 재닫기 서보각은 6.0875 → 5.9885 → 5.8107 로 회당 0.099°, 0.178° 줄어든다. **원인은 주장하지 않는다.**

### 8-2. 결정 시점 문 관절각 (생산 JSON == 원자료 배열, 오차 ≤1e-4°)

| 태그 | sync | sim t [s] | 관절 [°] | 서보 [°] |
|---|---|---|---|---|
| `close_stop` | 7,074 | 7.684674 | **3.404724** | 5.904724 |
| `lift_end` | 12,642 | 8.929542 | **3.310830** | 5.810830 |
| `reclose_end` | 13,051 | 8.970851 | **2.390452** | 4.890452 |
| `release_before` | 14,218 | 13.640018 | **2.392391** | 4.892391 |

문 정지 기록은 총 11건, 최종 관절각 0.0160°(서보 2.5°). `reclose` 정지 사유는 **`pinch_guard`** (접촉력 3.0113 N > 보호선 3.0 N), 힌지 모멘트 0.8480 N·m — W19 A 의 `servo_stall` 과 사유가 다르다(관측).

### 8-3. 흘림(spill) 25알의 이탈 단계
- 한 번이라도 spill = 25, 최종 spill = 25 (rev34 재계산도 25, 동일 ID 집합)
- **이탈 단계별: `transport` 25알 (다른 단계 0)**
- 최초 spill 프레임 144~163(t 11.5035~13.2039 s), 서브페이즈는 `place_retract_base90`. 직전 라벨은 대부분 `in_flight`.
  프레임별 분포 `{144:2, 145:1, 147:4, 148:1, 149:2, 150:2, 151:5, 154:2, 155:1, 156:2, 158:2, 163:1}`
- 최종 z 는 1.2487~1.4855 mm (`spill_rest_z_m` 20 mm 보다 아래), 최종 x 는 −0.35~−0.13 m 에 5~6개 군집

### 8-4. `bridge_clearance`
`CLEARANCE_CERTIFIED`, `pass: true`, sync 13,051, sim t 8.970851 s, 인증에 소비한 물리 step 0(`certify_consumed_zero_physics: true`, 전후 모두 13,052).
사전검사 220/220 통과·실패 0·계획-목표 편차 0.0 m, 최악 여유 0.0628 m(`tray_wall_plus_y`, y축, door). 관절 한계 위반 0. 문-고정부 잔차 8.89e-09 m(보고용).

### 8-5. `w25.frame`
`box_frame_convention: "A"`, `box_anchor: "declared_box_center"`,
`R_robot_box = [[0,1,0],[-1,0,0],[0,0,1]]`, `t_robot_m = [0.25, 0.0, -0.2577501314177868]`,
상자 바닥이 바닥에서 19.631 cm. 메타데이터와 생산 JSON 의 `R_robot_box` 가 동일.
트레이 선언 310×220×230 mm 이 npz 발자국과 xy 차 0.0 mm, 선언 밖 구 0개·윗단 위 0개, `mismatch_reasons: []`.
`fixtures.declared_not_measured: true` — 치수·자리는 **선언이며 실측이 아니다**.

## 9. P8 — W19 A 와 수치 병기 (관측만)

`W25_VS_W19_COMPARISON.json`

| 값 | **W25-A podB (rev34, 종이상자)** | **W19-A (rev32)** |
|---|---|---|
| 알 개수 | 67,737 | 20,000 |
| sync / 저장 프레임 | 22,128 / 275 | 16,813 / 288 |
| 시뮬 시간 | 22.9027 s | 24.8073 s |
| 실행 실제 경과 시간 | 36,372.797 s | 20,977.836 s |
| 확정 배출 | **585알 (11.8506 g)** | 272알 (5.51 g) |
| 가능 배출 | **665알 (13.4712 g)** | 327알 (6.6242 g) |
| 최종 재고 | src 63,402 / bin 585 / tool 14 / spill **25** / fly 0 / amb 3,711 | src 19,266 / bin 272 / tool 11 / spill **124** / fly 0 / amb 327 |
| 닫기 정지 관절각 | 3.4047° (`servo_stall`) | 3.2135° (`servo_stall`) |
| 재닫기 관절각 | **2.3905°** (`pinch_guard`) | **3.0536°** (`servo_stall`) |
| 채터링 | 3회 → `retries_exhausted` | 없음(rev32 에 절차 없음) |
| 정착 cadence | **미충족**(5프레임·0.100025 s) | **미충족**(5프레임·0.100025 s) |
| 재분류 불일치 | 0 / 18,627,675 칸 | 0 / 5,760,000 칸 |

> **인과 금지**: 두 실행은 알 개수·더미 형상·상자 선언·절차(채터링/표면 개방/항상 재닫기)·좌표계 규약·장비가 **동시에** 다르다.
> 배출량 차이를 어느 한 변수의 결과로 읽을 수 없다(n≥3 필요, D490 `:316`).
>
> **"같은 입력·다른 장비" 짝은 이 시점에 비교 불가.** `runs/podA_4090/` 에는 스모크 3종(`smoke_01~03`)과 부트스트랩 로그만 있고 `run_01` 원자료가 없다 → `COMPARISON_NOT_POSSIBLE_YET`.

## 10. 시각 진단과 실제 검수

PNG 5장 `diagnostics/01~05`, 측정값 `diagnostics/DIAGNOSTIC_MEASUREMENTS.json`,
**실제로 열어 본 관찰** `inspection.json` (D324 `AGENTS.md:140` — 생성 성공은 검수가 아니다).

주요 관찰: 용기 안 585알이 z 0.0035~0.014 m 층(테두리 0.073 m 에 한참 못 미침), ambiguous 80알이 안바닥·안벽 margin 밴드에 몰림,
spill 16알이 apothem 바깥 바닥면 z≈1.3 mm, 정착 창 속력/이동 분포가 판정선에서 두 자릿수 떨어져 있음,
문 각도 톱니 3회가 전부 5.5°→3.3~3.6° 로 임계(관절 환산 1.1°) 위에 머묾, 흘림 25알의 최초 시각이 11.5~13.2 s 한 구간에만 몰림.

**판독 결함도 적었다**: `02_inventory_per_frame.png` 에서 source(63,402) 곡선이 그려진 y 범위 위로 벗어나 보이지 않는다 — 그림만으로 source 추이를 읽으면 안 되고 `per_frame_counts_recorded` 를 봐야 한다. `03`·`04` 는 제목/라벨 겹침이 있다.

**RRD 생략 사유(D341 `AGENTS.md:152`)**: 이 단계는 라벨/인덱스 파생의 코드·배열·해시 감사이고 기하·포즈·접촉·궤적에 대한 새 판정을 내리지 않는다. 재생은 별도 단계(replay-renderer)가 같은 원자료로 수행한다.

## 11. FAIL 요약과 남은 승인 경계

**FAIL 총 8건** (P4 1 + P5 5 + P6 2 — 정착 cadence 는 같은 사실이 세 곳에 나타난다)

1. 정착 창 cadence 계약 미충족 — 5프레임(<6), 최대 간격 0.100025 s(>0.05 s). `settlement_frame_dt_s` 는 0.05 로 선언돼 있으나 실제 저장 간격 `particle_frame_dt_s` 는 0.1 이다.
2. 원자료 메타데이터에 ERRATUM_04 선언 문자열 5개가 전부 없음(라벨은 그 식으로 계산됐음에도).
3. 메타데이터 `time_mapping_abs_s` · `geometry_epsilon_m` 없음.
4. `visual_mapping.json` 없음(재생 단계 산출).
5. 바이트 수까지 포함한 최종 매니페스트 없음.
6. `policy.no_threshold_change_after_outcomes` — criteria sha 가 실행 영수증이 아니라 EXEC_PIN 에 기록됨(hard_fail severity).

**승인 경계 — 사용자 몫으로 남는 것**

- 배출량은 **585~665알 (11.85~13.47 g) 구간**으로만 인용 가능하다. 정착 cadence 계약이 미충족인 한 "585알 확정" 단일값 승격은 할 수 없다(`exact_single_value_allowed: false`).
- 저장 간격을 줄이는 것은 **새 revision + 새 criteria 파일**이 필요하다(등록 규약 `reading_rules[3]`: 결과를 본 뒤 값 변경 금지). 이 보고서는 그런 변경을 제안하지 않는다.
- "전체 사이클 성공" 판단은 재생(2단계)·독립 감사(3단계)를 거친 뒤 **사용자**가 한다(D490 `:316`).
- podA(4090)와의 장비 짝 비교는 podA `run_01` 회수 후에만 가능하다.

---

### 비주장

- 이 폴더는 **파생**이다. `run_01` 원자료의 어떤 바이트도 바꾸지 않았다(PRE==POST 증명).
- 재현 불일치 0 은 "기록이 규약대로 계산됐다"는 뜻이며 배출 성공·물리 타당성의 증거가 아니다.
- ambiguous 3,711알은 규약("모든 구체가 margin 안쪽")의 기하적 결과이며 물리 실패가 아니다.
- 어떤 FAIL 도 사후 허용값으로 완화하지 않았다(D485 `:309`).
- 차이는 관측이며 원인으로 읽지 않는다(D490 `:316`).
- 새 물리·GPU·DEME·Isaac·Rerun 실행 0, 설치 0, 버전관리 커밋/푸시 0.

### 산출물 색인

```
postprocess_20260929/
  PRESERVATION_BEFORE.json / PRESERVATION_AFTER.json   P1
  rev34_copy/{rev34,rev29}/… + REVISION_PIN.json       식 무수정 증거 (51 파일, n_bad 0)
  derived_v2_w25/w25_podB_seed460_rev34_derived.npz    P2 파생 라벨·전환·혼동행렬
                 DERIVED_V2_MANIFEST.json
                 SETTLEMENT_WINDOW_RECOMPUTE.json      P4
                 frozen_tool_attempt/                  동결 derive_v2 크래시 기록
  tests/{independent_check_v2.py,test_allframes_w25.py,w25_paths.py,RESULTS_allframes_w25.json}  P3
  schema_check/RAW_SCHEMA_CHECK_W25.json               P5 (27항목)
               CRITERIA_CHECK_W25.json                 P6 (45항목)
  W25_RAW_OBSERVATIONS.json                            P7
  W25_VS_W19_COMPARISON.json                           P8
  diagnostics/01~05*.png + DIAGNOSTIC_MEASUREMENTS.json
  inspection.json                                      실제 육안 검수 기록
  manifest.json                                        경로·바이트·sha256
  tools/  logs/
```
