# W25-A podA 원자료 회계 재계산 보고 (raw-accountant, 1단계)

- 대상 원자료: `…/exec_rev34_paperbox_20260929/runs/podA_4090/run_01/` (14 파일, NPZ 1.77 GB)
- 출력 폴더: `…/runs/podA_4090/postprocess_20260930/`
- 절차 정본: podB 후처리 보고서 `…/runs/podB_pro6000x2/postprocess_20260929/REPORT_postprocess_w25_podB.md` 의 P1~P8.
  **도구는 그 폴더 `tools/*.py` 8개의 바이트 사본**이며, 경로 인자만 podA 로 바꿔 실행했다(§2-2 sha 대조표).
- 실행 조건: **CPU 전용, 새 물리 0, GPU/DEME/Isaac/Rerun 0, 원자료 쓰기 0, 설치 0, 버전관리 커밋/푸시 0**
- 이 단계의 실제 경과 시간(wall-clock): 약 24분 (2026-09-30 10:00:40Z 시작 → 10:24Z)
- 결론 범위: **생산 회계 재현 여부 + 항목별 PASS/FAIL 까지.** "전체 사이클 성공" 선언은 하지 않는다(D490 `claudedocs/DECISIONS_ACTIVE.md:316`).

용어: **실제 경과 시간(wall-clock)** = 시계로 잰 시간. **파생(derived)** = 원자료를 고치지 않고 다시 계산해 만든 값.
**규약(contract)** = 라벨·전환·정착을 어떻게 세는지 미리 적어 둔 문서. **cadence** = 저장 프레임이 얼마나 촘촘한가.
**sync** = 시뮬레이터가 물리를 돌린 뒤 제어값을 주고받는 한 스텝. **판정불가** = 이 단계의 산출 경계 밖이라 PASS/FAIL 을 매기지 않는 항목.

---

## 0. 한눈에 보는 결과

| 항목 | 결과 | 출처 |
|---|---|---|
| P1 보존(원자료 무수정) | **PASS** — 14/14 sha256 이 회수 영수증과 일치, 작업 전·후 동일(`diff PRE POST` 0줄) | `PRESERVATION_BEFORE.json` · `PRESERVATION_AFTER.json` · `logs/PRE.txt`·`POST.txt` |
| P2 생산식 재분류 재현 | **PASS** — 18,559,938 라벨 칸 중 **불일치 0**, 클래스별 전부 0 | `derived_v2_w25/DERIVED_V2_MANIFEST.json` |
| P3 독립 검사기 재현 | **PASS** — 같은 18,559,938 칸 **불일치 0**, unittest 4/4 OK | `tests/RESULTS_allframes_w25.json` |
| P4 정착 창 재계산 | **6항목 중 1 FAIL** — cadence 계약(≥6프레임·≤0.05 s) **미충족** | `derived_v2_w25/SETTLEMENT_WINDOW_RECOMPUTE.json` |
| P5 원자료 스키마 27항목 | **22 PASS / 5 FAIL** | `schema_check/RAW_SCHEMA_CHECK_W25.json` |
| P6 등록 규약 45항목 | **35 PASS / 2 FAIL / 8 판정불가**, 그중 hard_fail FAIL 1 | `schema_check/CRITERIA_CHECK_W25.json` |
| P7 W25 전용 관측 | 채터링 3회 후 `retries_exhausted`, 흘림 123알 전부 transport 단계 | `W25_RAW_OBSERVATIONS.json` |
| P8 podA↔podB 비교 | **수치 병기 완료**(같은 입력·다른 장비 짝). 판정 문구 없음 | `W25_PODA_VS_PODB_COMPARISON.json` |
| (부) podA vs W19 A | podB 도구 그대로 재실행 — 단 내부 라벨 문자열은 podB 용이다(§9-3) | `W25podA_VS_W19_COMPARISON.json` |

**FAIL 총 8건**(P4 1 + P5 5 + P6 2 — 정착 cadence 는 같은 사실이 세 곳에 나타난다). FAIL 항목 집합은 podB 와 **문자 단위로 같다**(§9-1 마지막 줄).

---

## 1. P1 보존 영수증

| 시점 | 파일 수 | 영수증 불일치 | 판정 | 파일 |
|---|---|---|---|---|
| 작업 전 (10:00:40Z) | 14 | 0 | PASS | `PRESERVATION_BEFORE.json` |
| 작업 후 (10:23:17Z) | 14 | 0 | PASS | `PRESERVATION_AFTER.json` |

`logs/PRE.txt` 와 `logs/POST.txt` 의 `diff` 가 **0줄**이다.
원자료 NPZ `8b341be1…0a13e`, 생산 JSON `49374b1f…b38e`, timeline `9566d1c5…8581` 를 포함해 14개 전부
`runs/podA_4090/RETRIEVAL_RECEIPT.json` 값과 같다.

> 관측 하나: `run_01/_obj/` 4개(bin·door·fixed·tray OBJ)의 sha256 은 podB run_01 의 같은 파일과 **바이트 동일**하다
> (`56b3e5b0…`, `68db266c…`, `f477a77d…`, `dabd7c21…`). 기하 내보내기가 같은 입력에서 같은 바이트를 냈다는 관측이며, 원인은 주장하지 않는다.

## 2. 식 무수정 증거

### 2-1. rev34_copy

`rev34_copy/REVISION_PIN.json` — 51개 파일 사본 전부 원본 sha 와 일치(`n_bad: 0`), `EXEC_PIN.json` 기록 sha 와도 일치.
podB 의 `rev34_copy/` 와 재귀 `diff` 결과 `REVISION_PIN.json` 을 빼면 **완전 동일**하다.

- `rev34/src/inventory_geometry.py` = `a49e1ba9…0327` = **W14 rev31 원본과 바이트 동일** → 분류식 정본이 W19 때와 같은 파일이다.
- 메인 repo `sim_deme_scoop_s1.py` = `2e40f7ed…7933` = W14 핀과 일치.
- 식은 한 줄도 고치지 않았다. 경로 문제는 **래퍼**(`tools/derive_w25.py`, podB 바이트 사본)로만 우회했고, 동결 파일 편집 0건이라 `DIFF_paths_only.patch` 는 필요 없다.

### 2-2. 도구 바이트 사본 sha256 대조 (podB → podA)

`sha256sum -c` 로 8/8 `OK`. 테스트 3개도 바이트 사본이다.

| 파일 | sha256 | 대조 |
|---|---|---|
| `tools/derive_w25.py` | `2d5d2e2dda64c6fe1e5f41b6de49d6dafb4127ed18378d0b9ba77d65ae9c1119` | OK |
| `tools/settlement_w25.py` | `c24ec023cf2225b9415957078a3e8f87d552100cf9e8d8b621f0e8cc19afe07d` | OK |
| `tools/schema_check_w25.py` | `4856e24319ca5c9bb40a3261456134e700f9b3fdcbdc5cca36c6f61295938b8b` | OK |
| `tools/observations_w25.py` | `cc6159266557b79f27286be85cb87d07f7e1e85e4d6fecc922b051c450e7d07c` | OK |
| `tools/diagnostics_w25.py` | `5041783a2fa74e158b5053237d678015b3b98603ae81cdc32586fbcf436aefd2` | OK |
| `tools/compare_w19_w25.py` | `a34ab7f8b15058f40dcf3850765d1802fc1884d9d3e762158d50e914e1fbe1d5` | OK |
| `tools/mk_revision_pin.py` | `32ffbad8a5c4d4014a5c16cae3aa80c0a5ee6adb623f02039974e2e172d6299e` | OK |
| `tools/mk_manifest.py` | `4734e6fc43823c3286ec8aa182c6b60b3a562d4d2ac44cd618089c1a8baaa6fc` | OK |
| `tests/independent_check_v2.py` | `bfd51c114f81824c9ba5a4b96b0f838985790902b18c7dcd9512fef2bcff9f4c` | OK |
| `tests/test_allframes_w25.py` | (podB 사본, sha 는 manifest.json) | OK |
| `tests/w25_paths.py` | (podB 사본, sha 는 manifest.json) | OK |

`tools_new/` 에만 새로 쓴 파일이 2개 있다. **동결 식은 건드리지 않는다.**

- `tools_new/preservation.py` — podB 는 P1 을 인라인으로 했고 `tools/` 에 파일이 없어 같은 형식으로 새로 썼다.
- `tools_new/compare_podA_podB.py` — podB 도구 `compare_w19_w25.py` 는 W25↔W19 전용이고 그 파일 `:3-4` 가
  "podA 가 아직 실행 중이라 이 시점에 비교 불가"라고 적고 있다. P8 의 장비 짝 비교를 하려면 새 파일이 필요했다.
  하는 일은 이미 기록된 배열/JSON 값을 읽어 나란히 적고 빼는 것뿐이다.

### 2-3. 동결 `derive_v2.py` 를 그대로 돌렸을 때 (기록용, podB §2-1 재현)

`rev34_copy/rev34/src/derive_v2.py` 바이트 사본을 두 번 실행했고 두 번 다 **rc=1** 로 멈췄다.
로그: `derived_v2_w25/frozen_tool_attempt/`.

1. `FileNotFoundError` — `--rev29-derived` 기본값이 W13 전용 경로(`…/rev34_copy/derived/w13_cycle_seed460_rev29_derived.npz`)라 W25 에는 그 파일이 없다.
2. 존재하는 파일을 넘겨 그 지점을 지나가게 하자 `SystemExit: 입력 해시 불일치 raw` — `derive_repaired_raw.py` 의 `EXPECTED_SHA256["raw"]`(W13 원자료)와 podA 원자료 `8b341be1…0a13e` 가 다르기 때문이다.

즉 동결 드라이버는 **W13 입력에 고정**돼 있다. 그래서 식(분류·전개·전환 함수)은 사본 모듈에서 그대로 import 하고,
경로·해시·메모리 처리만 하는 래퍼를 썼다(podB 와 같은 방식, 그 래퍼도 바이트 사본이다).

## 3. P2 — 전 프레임 재분류 vs 생산 기록

- 프레임 **274** × 입자 67,737 = **라벨 칸 18,559,938** (podB 는 275 프레임)
- **rev34(=rev31 식, ERRATUM_04 받침면) 재계산 vs 원자료 기록: 불일치 0**
- 클래스별 불일치도 전부 0 (`source/receiving_bin/tool_residual/spill/in_flight/ambiguous` 모두 0)
- 혼동행렬이 완전 대각선이다: 대각 `[17,461,085 / 24,977 / 25,724 / 15,021 / 5,020 / 1,028,111]`, 비대각 합 **0**
- 생산 JSON `decisions[*].counts` 14개 태그 전부가 원자료 라벨 bincount 와 동일(P3 `decision_counts_json_vs_raw_mismatch: []`)
- `delivery.inventory_final`(JSON) == 원자료 마지막 프레임 bincount == 재계산 값:
  `source 63,636 / receiving_bin 251 / tool_residual 14 / spill 123 / in_flight 1 / ambiguous 3,712` (합 67,737)
- 전환 인덱스: 기록 `[1, 26, 4681, 4948, 13118, 13293, 13414, 14581, 16856, 17231, 20850]` == 규약(phase-only) 재계산값.
  rev28 legacy 규칙으로 재현하면 **32개**가 나와 기록과 다르다(= 기록이 규약 쪽을 따랐다는 확인).
- 대조용 **rev29 strict floor** 로 다시 세면 **1,880,535** 칸이 달라진다
  (주로 `source → ambiguous` 1,855,862, 다음으로 `receiving_bin → ambiguous` 24,344). 이는 규약 판(받침면 vs 엄격 하한)의 차이이지 오류가 아니다.
- 소요: 87.178 s. 산출: `derived_v2_w25/w25_podB_seed460_rev34_derived.npz`(sha `fde7ae31…091a`) + `DERIVED_V2_MANIFEST.json`

> **파일 이름 주의**: 파생 NPZ 파일명이 `w25_podB_…` 인 것은 `tools/derive_w25.py` 안에 이름이 박혀 있고
> 그 도구를 **바이트 사본으로 두었기 때문**이다. 내용은 podA 원자료에서 나온 값이다(`provenance_json.source_raw_npz` 가 podA 경로).
> 도구를 고치지 않는다는 계약을 이름보다 우선했다.

> **비주장**: 재현 0 불일치는 "기록이 규약대로 계산됐다"는 뜻이다. 배출량이 옳다/성공했다는 뜻이 아니다.

## 4. P3 — 독립 검사기

`tests/independent_check_v2.py`(sha `bfd51c11…9f4c`, import = `itertools`·`math`·`numpy` 뿐)는 생산 모듈
(`inventory_geometry`, `sim_deme_scoop_s1`, `w13_*`)·`scipy` 를 **전혀 import 하지 않고** 규약 문구에서 다르게 구현한 것이다
(회전 = Hamilton 곱, 봉쇄 = 축정렬 경계상자, 전환 = `groupby`). W19 후처리 파일의 바이트 사본이라 podA 결과를 보고 맞춰 쓸 수 없었다.

| 검사 | 결과 |
|---|---|
| 전 274프레임 생산식 vs 독립식 | **불일치 0 칸 / 18,559,938 칸** |
| 전환 인덱스 독립 재계산 | 기록과 동일, ERRATUM_04 §2 경계 프레임 `i-1` 누락 0 |
| 원자료·파생 sha | 회수 영수증·manifest 와 일치, `n_sync` 22,204 == 생산 JSON |
| `decisions[*].counts` JSON vs 원자료 | 불일치 0 |
| unittest | `Ran 4 tests … OK` (127.359 s) |

결과: `tests/RESULTS_allframes_w25.json`, 로그 `tests/allframes.stderr.txt`.

## 5. P4 — 정착 창 재계산 (**FAIL 1**)

동결 파라미터: `settlement_window_s = 0.25`, `settlement_frame_dt_s = 0.05`, `particle_frame_dt_s = 0.1`,
`settle_speed_max_m_s = 0.005`, `settle_move_max_m = 0.001`.
cadence 계약 원문은 `exec_rev34_paperbox_20260929/rev34/src/sim_w13_full_cycle.py:1005`
— `"cadence_ok": bool(len(widx) >= 6 and max(...) <= 0.05 + 1e-9)`.

| 항목 | 판정 | 수치 |
|---|---|---|
| 창 재계산 == 생산 기록 | PASS | 9개 키 전부 일치(`n_differing_keys: 0`) |
| **cadence 계약 (≥6프레임 · ≤0.05 s)** | **FAIL** | 프레임 **5개**, 최대 간격 **0.100025 s** |
| 확정 배출 재계산 == 생산 | PASS | **251** |
| 가능 배출 재계산 == 생산 | PASS | **297** (251 + 밴드 안 46 + 테두리 위 0) |
| `exact_single_value_allowed` 플래그 | PASS | `false` (251 ≠ 297 과 일치) |
| 기록 라벨 기준 vs rev34 재계산 라벨 기준 | PASS | 둘 다 stable 251 / settled 251 |

창 프레임 시각: `22.600547 / 22.700572 / 22.724578 / 22.800597 / 22.824603 s`,
간격 `0.100025 / 0.024006 / 0.076019 / 0.024006 s`.
(podB 와 달리 podA 는 같은 시각이 겹치는 쌍이 없다 — 서로 다른 시각 5개다. 그래도 최대 간격 0.100025 s 는 같다.)

창 안 안정 입자의 최대 속력 0.00023046 m/s(판정선 0.005), 최대 중심 이동 0.015148 mm(판정선 1 mm)로 **여유는 크다**.
그러나 **저장 간격이 0.1 s 라 0.05 s 계약을 채울 수 없다** — 여유가 크다는 사실이 cadence 계약 충족을 대신하지 않는다.
**사후 허용값을 만들지 않았고, FAIL 로 그대로 둔다**(D485 `:309`).

> **두 층 분리**: 이것은 *기록 계약*의 한계다(저장 간격 0.1 s). *물리 실패*가 아니다.
> 그래서 배출량은 **251~297알(5.08~6.02 g) 구간**으로만 읽고, "251알 확정" 같은 단일값 인용은 금지된다(`exact_single_value_allowed: false`).

산출: `derived_v2_w25/SETTLEMENT_WINDOW_RECOMPUTE.json`.

## 6. P5 — 원자료 스키마 27항목 (22 PASS / **5 FAIL**)

`schema_check/RAW_SCHEMA_CHECK_W25.json` · verdict `RAW_SCHEMA_CHECK_FAIL_5_OF_27`

PASS 쪽 주요 확인:
dense 배열 22,204행 17종 모양/dtype, sparse 274×67,737 배열, `inventory_labels` 정확 일치,
metadata 필수 28키(`w25_frame` 4개 하위키 전부 포함), ERRATUM_03 행 정체성(`particle_frame_row == arange`,
sync 인덱스 비감소, 중복 **16개**), ERRATUM_04 §2 경계 프레임, 전환 인덱스,
`sync_t_s` 엄격 증가 + 입자 프레임 시각 오차 0.0,
접촉 **49,517,594**행(역할 매핑 `fixed:0/door:1/tray:2/bin:3`, sync 인덱스 1~22,203 범위 정상),
ERRATUM_02 용기 반경 의미(`circumradius`), 공구 공동·소스 경계·임계 선언, 설치 수치 입력, 정준 템플릿·질량(0.020257 g/알),
속도 5/20 심각도(최대 2.3335 m/s, 경고 초과 0), rc0↔stderr 대조(stderr 0 바이트, `timed_out:false`, `killed:false`),
재시도 0(`auto_retry:false`, 시도 폴더 `run_01` 하나), 사전 해시 영수증(선언 15개 재계산 불일치 0; 사전 검증 **606건** 불일치 0),
EXEC_PIN 바이트 동일 **77/77**, 재고 전수·배타(총합 틀린 프레임 0, 최종 ambiguous 비율 5.48 %), w25 절차·좌표계 기록 존재.

### FAIL 5건

| # | 항목 | 실측 | 규약 인용 |
|---|---|---|---|
| 1 | `metadata_time_mapping_abs_s_and_geometry_epsilon_m` | 두 키 **모두 없음** | `RAW_SCHEMA_REQUIRED.md:71-72` "`time_mapping_abs_s`, `geometry_epsilon_m`, particle density, canonical pile absolute path and SHA-256, and every frozen scientific criterion path/hash." W19 A·W25 podB 도 같은 FAIL |
| 2 | `erratum04_floor_rule_and_contract_version_declared_in_raw_metadata` | `classify_floor_rule`·`classify_contract_version`·`classify_source_floor_rule`·`classify_revision`·`classify_predicate_module` **5개 전부 `null`** | `RAW_SCHEMA_REQUIRED_ERRATUM_04.md:37-38` 이 두 키의 문자열을 지정한다. 같은 문서 `:10-11` 은 "effective only for producer or derived revisions that explicitly declare `RAW_SCHEMA_REQUIRED + ERRATUM_04`; it does not apply retroactively". **라벨은 ERRATUM_04 식으로 계산됐는데(P2/P3 확인) 원자료 메타데이터가 그 규약을 선언하지 않는다.** |
| 3 | `erratum01_visual_mapping_one_row_per_saved_particle_frame` | `visual_mapping.json` **없음** | 재생(2단계) 산출물. 저장 프레임이 12개 단계를 전부 덮는 것은 확인됨(`all_12_phases_covered: true`). 없는 것은 FAIL 로 남긴다 |
| 4 | `final_manifest_path_bytes_sha256_for_every_output` | `MANIFEST.json`/`FINAL_MANIFEST.json` **없음** | 회수 영수증이 14개 파일 sha 를 모두 덮지만 byte count 가 없다. W19·podB 도 같은 FAIL |
| 5 | `delivery_layers_separate_and_settlement_cadence_contract` | 5프레임·0.100025 s (확정 251 / 가능 297 / `exact_single_value_allowed:false`) | P4 와 같은 사실 |

## 7. P6 — 등록 규약(criteria) 45항목 (35 PASS / **2 FAIL** / 8 판정불가)

`schema_check/CRITERIA_CHECK_W25.json` · 규약 파일 `criteria_w25_paperbox_cap32h.json`
(sha `ba46b045…f15d`, `frozen_before_production: true`)

### 상한 관련

| 항목 | 규약값 | 실측 | 판정 |
|---|---|---|---|
| `runner.physics_wall_cap_s` | 115,200 s | 영수증 cap 115,200 s / 실제 **52,707.941 s** (45.75 %, 여유 62,492.1 s) | PASS |
| `runner.sim_soft_wall_cap_s` | 114,000 s | argv 114,000 s / 시뮬 내부 52,603.59 s, `abort_class: null` | PASS |
| `runner.graceful_grace_s` | 1,200 s | 영수증 1,200 s, `killed: false`, `group_alive_after: false` | PASS |
| `runner.no_retry` | 재시도 0 | `auto_retry: false`, 시도 폴더 `run_01` 하나 | PASS |
| `process.rc0_is_not_delivery` | rc0 ≠ 배출 판정 | rc 0, `abort_class: null`, `exact_single_value_allowed: false`, 확정 251 / 가능 297 | PASS |

**타임아웃 ≠ 성공 대조(D486 `:311`)**: `RUN_STATUS.json` 은 `state: completed_rc0`, `timed_out: false`, `killed: false`,
`signals_received: []` 이고 `run_paperbox_full_cycle.stderr.txt` 는 **0 바이트**다. 영수증의 자동 종료 분류와 stderr 가 어긋나지 않는다.

### FAIL 2건

1. **`delivery.settlement_window_UNCALIBRATED`** (severity `uncalibrated_report_only`) — 5프레임·0.100025 s 로 cadence 미충족.
   이 severity 는 토큰 발급을 막지 않는 **보고 대상**이다(규약 `reading_rules[0]`). 그래도 사후 완화 없이 FAIL 로 적는다.
2. **`policy.no_threshold_change_after_outcomes`** (severity **`hard_fail`**) — 규약 문구는
   "criteria.json is frozen before production **and its SHA256 is recorded in the execution receipt**". 실측:
   - criteria sha `ba46b045…f15d` 는 `EXEC_PIN.json`(GPU 결과 0 시점 동결본)에 기록돼 있다(`size 37334`).
   - 그러나 `run_01/EXECUTION_RECEIPT.json` 이 담는 sha 는 `commands_json_sha256`·`runner_self_sha256` 둘뿐이고,
     그 `COMMANDS_w25_podA_4090.json` 도 criteria **경로**만 적고 sha 는 적지 않는다.
   - 동결 시점 자체는 앞선다: criteria `changed_utc 2026-09-28T20:53:49Z` < 실행 시작 `2026-09-28T21:13:47Z`
     (`criteria_frozen_before_run: true`).
   - **문구를 글자대로 읽어 FAIL 로 남긴다.** 기록 위치가 영수증이 아니라 상위 핀 파일이라는 **관측**이며,
     임계가 결과를 보고 바뀌었다는 주장은 하지 않는다.

### 판정불가 8건 (PASS 아님)

`visual.production_maps_every_particle_frame`, `visual.rrd_counts_logged_rows_not_totals`,
`visual.aborted_certificate_no_invented_pass`, `visual.actual_tool_geometry_required`,
`visual.rerun_contract`, `visual.isaac_frame_mapping`, `readiness.renderer_bounds`,
`negative_controls.required` — 재생(RRD/Isaac)·readiness·음성대조 산출물이 필요하다. 이 회계 단계의 산출 경계 밖이다.

> bridge 계열은 PASS: `numeric_epsilon_m`(1e-06), 최악 여유 **0.06283244 m**(`tray_wall_plus_y`, y축, door),
> 사전검사 220/220 통과·실패 0·계획-목표 편차 0.0 m, 관절 한계 위반 0, 인증에 소비한 물리 step 0
> (`certify_consumed_zero_physics: true`, 전후 모두 13,414).

## 8. P7 — W25 전용 관측 (원시 사실만)

`W25_RAW_OBSERVATIONS.json`

### 8-1. 채터링(실물 절차 재현)
파라미터: 임계 서보 3.6°, 재열기 서보 8.0°(관절 5.5°), 최대 3회. 단위: `servo_deg = joint_deg + 2.5`.

| 회 | 읽은 값 (관절/서보) | 재열기 정지 (관절/서보, 사유) | 재닫기 정지 (관절/서보, 사유) | 힌지 모멘트 |
|---|---|---|---|---|
| 0 | 3.3576 / 5.8576 | 5.5 / 8.0, `reached_open_end` | **3.1577 / 5.6577**, `servo_stall` | 1.7747 N·m |
| 1 | 3.1577 / 5.6577 | 5.5 / 8.0, `reached_open_end` | **3.2116 / 5.7117**, `servo_stall` | 1.7797 N·m |
| 2 | 3.2116 / 5.7116 | 5.5 / 8.0, `reached_open_end` | **3.3062 / 5.8062**, `servo_stall` | 1.7689 N·m |
| 3 | 3.3062 / 5.8062 | — | — (`retries_exhausted`) | — |

세 번 모두 임계 3.6°(서보) 아래로 내려가지 못했고 마지막 행은 `retries_exhausted` 다.
재닫기 서보각은 5.6577 → 5.7117 → 5.8062 로 회당 **+0.054°, +0.095°** 늘어난다. **원인은 주장하지 않는다.**

### 8-2. 결정 시점 문 관절각 (생산 JSON == 원자료 배열, 오차 ≤1e-4°)

| 태그 | sync | sim t [s] | 관절 [°] | 서보 [°] |
|---|---|---|---|---|
| `close_stop` | 7,073 | 7.598773 | **3.357609** | 5.857609 |
| `lift_end` | 13,292 | 8.909392 | **3.306213** | 5.806213 |
| `reclose_end` | 13,413 | 8.921613 | **3.034077** | 5.534077 |
| `release_before` | 14,580 | 13.590780 | **3.035641** | 5.535641 |

문 정지 기록은 총 11건, 최종 관절각 0.0160°(서보 2.5160°).
`reclose` 정지 사유는 **`servo_stall`**(힌지 모멘트 1.7753 N·m, 최대 단일 접촉 1.7578 N).
— podB 의 `pinch_guard` 와 사유가 다르다(관측).

### 8-3. 흘림(spill) 123알의 이탈 단계
- 한 번이라도 spill = 123, 최종 spill = 123 (rev34 재계산도 123, 동일 ID 집합)
- **이탈 단계별: `transport` 123알 (다른 단계 0)**
- 최초 spill 프레임 143~165(t 11.4022~13.4027 s), 서브페이즈는 `place_retract_base90`. 직전 라벨은 대부분 `in_flight`.
  프레임별 분포 `{143:8, 144:3, 145:3, 146:3, 147:9, 148:5, 149:1, 150:6, 151:16, 152:20, 153:10, 154:10, 155:16, 156:1, 157:4, 158:3, 163:1, 164:3, 165:1}`
- 최종 z 는 1.2485~1.9793 mm (`spill_rest_z_m` 20 mm 보다 아래), 최종 x 는 −0.35~−0.08 m 에 퍼져 있다

### 8-4. `bridge_clearance`
`CLEARANCE_CERTIFIED`, `pass: true`, sync 13,413, sim t 8.921613 s, 인증에 소비한 물리 step 0.
사전검사 220/220 통과·실패 0·계획-목표 편차 0.0 m, 최악 여유 0.06283244 m(`tray_wall_plus_y`, y축, door).
관절 한계 위반 0. 문-고정부 잔차 1.493e-09 m(보고용).

### 8-5. `w25.frame`
`box_frame_convention: "A"`, `box_anchor: "declared_box_center"`,
`R_robot_box = [[0,1,0],[-1,0,0],[0,0,1]]`, `t_robot_m = [0.25, 0.0, -0.2577501314177868]`,
상자 바닥이 바닥에서 19.631 cm. 메타데이터와 생산 JSON 의 `R_robot_box` 가 동일.
트레이 선언 310×220×230 mm 이 npz 발자국과 xy 차 0.0 mm, `mismatch_reasons: []`.
`fixtures.declared_not_measured: true` — 치수·자리는 **선언이며 실측이 아니다**.

## 9. P8 — podA ↔ podB 수치 병기 (관측만)

`W25_PODA_VS_PODB_COMPARISON.json`

### 9-0. 같은 입력이라는 증거 (영수증 argv 대조)

| 항목 | podA | podB | 동일 |
|---|---|---|---|
| 동결 시뮬 스크립트 | `rev34/src/sim_w13_full_cycle.py` | 같음 | 예 |
| `--params` | `rev34/params_w25_paperbox.json` | 같음 | 예 |
| `--pile` | `pile_…n67737_rho0p503_seed460.npz` (sha `31dd2897…83c1`) | 같음 | 예 |
| `--seed` | 460 | 460 | 예 |
| `--numeric-evidence` | `numeric_inputs_w25_paperbox_n67737.json` | 같음 | 예 |
| `--max-wall-s` | 114000 | 114000 | 예 |
| hostname(컨테이너) | `0bf27bcbe95b` | `19c0ea4352b0` | 아니오 (다른 pod) |
| GPU | RTX 4090 ×1 | RTX PRO 6000 ×2 | 아니오 |

### 9-1. 주요 수치 병기

| 값 | **podA (4090 ×1)** | **podB (PRO 6000 ×2)** | podA−podB | podA/podB |
|---|---|---|---|---|
| 알 개수 | 67,737 | 67,737 | 0 | 1.000 |
| sync 수 | 22,204 | 22,128 | +76 | 1.00343 |
| 저장 입자 프레임 | 274 | 275 | −1 | 0.99636 |
| 시뮬 시간 | 22.824603 s | 22.902727 s | −0.078124 s | 0.99659 |
| **실행 실제 경과 시간(wall-clock)** | **52,707.941 s** | **36,372.797 s** | +16,335.144 s | **1.4491** |
| 시뮬 내부 wall | 52,603.59 s | 36,311.69 s | +16,291.9 s | 1.4487 |
| **확정 배출** | **251알 (5.0846 g)** | **585알 (11.8506 g)** | −334알 | **0.4291** |
| **가능 배출** | **297알 (6.0164 g)** | **665알 (13.4712 g)** | −368알 | 0.4466 |
| 최종 재고 | src 63,636 / bin 251 / tool 14 / **spill 123** / fly 1 / amb 3,712 | src 63,402 / bin 585 / tool 14 / **spill 25** / fly 0 / amb 3,711 | — | — |
| 들린 알(`lift_end` tool_residual) | **388** | **397** | −9 | 0.97733 |
| `lift_end` in_flight / ambiguous | 4 / 3,990 | 5 / 3,978 | −1 / +12 | — |
| `close_stop` tool_residual | 441 | (podB 보고서 미기재) | — | — |
| `reclose_end` tool_residual | 417 | 417 | 0 | 1.000 |
| 닫기 정지 관절각 | 3.3576° (`servo_stall`) | 3.4047° (`servo_stall`) | −0.0471° | — |
| **문 재닫기(reclose) 정지각** | **3.0341° / 서보 5.534°, `servo_stall`** | **2.3905° / 서보 4.8905°, `pinch_guard`** | +0.6436° | — |
| 채터링 | 3회 → `retries_exhausted` | 3회 → `retries_exhausted` | — | — |
| 채터 재닫기 각 추이 | 3.1577 → 3.2116 → 3.3062 (증가) | 3.5875 → 3.4885 → 3.3107 (감소) | — | — |
| 정착 창 프레임 / 최대 간격 | 5 / 0.100025 s (**미충족**) | 5 / 0.100025 s (**미충족**) | 0 / 0 | — |
| 정착 창 안정 입자 최대 속력 | 0.00023046 m/s | 0.00038295 m/s | — | 0.6018 |
| 정착 창 안정 입자 최대 이동 | 0.015148 mm | 0.010357 mm | — | 1.4626 |
| 흘림(spill) | 123 | 25 | +98 | 4.92 |
| 구덩이 제거 부피 | 51.8095 cm³ | 53.7693 cm³ | −1.9598 | 0.96355 |
| 구덩이 최대 깊이 | 17.3357 mm | 19.9085 mm | −2.5728 | 0.87077 |
| 재분류 불일치 | 0 / 18,559,938 칸 | 0 / 18,627,675 칸 | — | — |
| 회계 FAIL 집합 | P5 5건·P6 2건 | 동일 5건·2건 (`failures` 배열 문자 단위 동일) | — | — |

### 9-2. 단계별 실제 경과 시간(wall-clock)

`sync_wall_elapsed_s` 를 단계 경계에서 차분한 값. 단위 s.

| 단계 | podA sync 행 | podB sync 행 | podA wall | podB wall | podA/podB |
|---|---|---|---|---|---|
| initial_home | 1 | 1 | 28.2 | 11.2 | 2.528 |
| settle | 25 | 25 | 238.7 | 91.1 | 2.620 |
| approach | 4,655 | 4,655 | 12,066.5 | 4,773.1 | 2.528 |
| descend | 267 | 289 | 2,289.6 | 1,023.8 | 2.236 |
| close | 8,170 | 7,498 | 5,402.3 | 2,368.1 | 2.281 |
| lift | 175 | 175 | 1,474.3 | 812.0 | 1.816 |
| reclose | 121 | 409 | 59.4 | 120.4 | 0.493 |
| transport | 1,167 | 1,167 | 9,959.0 | 8,240.2 | 1.209 |
| discharge | 2,275 | 2,561 | 2,802.1 | 2,463.8 | 1.137 |
| discharge_wait | 375 | 375 | 3,220.6 | 2,444.1 | 1.318 |
| close_after_discharge | 3,619 | 3,619 | 3,458.6 | 4,352.3 | 0.795 |
| return_home | 1,354 | 1,354 | 11,604.0 | 9,611.4 | 1.207 |

### 9-3. podA vs W19 A (podB 도구 그대로 재실행 — 라벨 주의)

`W25podA_VS_W19_COMPARISON.json`. podB 도구 `compare_w19_w25.py` 의 바이트 사본을 podA JSON 으로 돌린 것이라
**파일 안의 문자열이 podB 용 그대로**다. 아래 두 문자열은 podA 파일에 대해 **무효**이니 읽지 말 것:

- `rows[0].tag = "W25-A podB_pro6000x2 (rev34, paper box, n=67,737)"` → 실제 내용은 **podA** 수치다.
- `same_input_different_hardware.verdict = "COMPARISON_NOT_POSSIBLE_YET"` → 도구에 박힌 상수다.
  실제 짝 비교는 §9-1·§9-2 와 `W25_PODA_VS_PODB_COMPARISON.json` 에 있다.

수치(관측): podA 251알/297알 vs W19 A 272알/327알, 최종 spill 123 vs 124, reclose 3.0341° vs 3.0536°,
정착 cadence 양쪽 다 미충족.
W19 는 알 개수·더미 형상·상자 선언·절차·좌표계 규약이 모두 달라 짝 비교가 아니다.

### 9-4. 이 절의 비주장

- 두 실행은 **같은 동결 스크립트·params·pile·seed** 를 썼고 **GPU 장비/개수와 컨테이너만 다르다**.
  그래도 위 차이를 "장비 때문"이라고 읽지 않는다 — DEM 접촉 해가 비결정적이라는 것이 W21 에서 이미 관측됐고,
  인과 주장에는 n≥3 이 필요하다(D490 `:316`). 이 표는 **관측 2점**일 뿐이다.
- 배출 251 vs 585, 흘림 123 vs 25 는 같은 입력에서 나온 **변동폭의 크기**를 보여주는 관측이다.
  어느 쪽이 "옳은 값"인지 이 단계에서 정하지 않는다.
- 두 실행 모두 정착 cadence 계약 미충족이므로 배출 수치는 **구간으로만** 읽는다.

## 10. 시각 진단과 실제 검수

PNG 5장 `diagnostics/01~05`, 측정값 `diagnostics/DIAGNOSTIC_MEASUREMENTS.json`,
**실제로 열어 본 관찰** `inspection.json` (D324 `AGENTS.md:140` — 생성 성공은 검수가 아니다).

주요 관찰: 용기 안 251알이 z 0.0035~0.009 m 의 한 겹 층(테두리 0.073 m 에 한참 못 미침),
ambiguous 46알이 안바닥·안벽 margin 밴드에 몰림, spill 93알이 apothem 바깥 바닥면 z≈1.2~2.0 mm,
정착 창 속력/이동 분포가 판정선에서 한두 자릿수 떨어져 있음, 문 각도 톱니 3회가 전부 5.5°→3.16~3.31° 로
임계(관절 환산 1.1°) 위에 머묾, 흘림 123알의 최초 시각이 11.4~13.4 s 한 구간에만 몰림.

**바이트 사본 도구에서 나온 판독 결함 3건**(도구는 고치지 않았다 — `inspection.json` `tool_byte_copy_artifacts_found_while_inspecting`):

1. `tools/diagnostics_w25.py:88` — 그림 02 제목이 `"…(275 frames)…"` 로 **podB 값 하드코딩**. podA 는 274 프레임이다.
2. `tools/diagnostics_w25.py:154-155` — `close_stop_deg = door[7074]`, `reclose_deg = door[13051]` 로 **podB 의 sync 인덱스**가 박혀 있다.
   podA 의 정지는 sync 7073·13413 이므로 `DIAGNOSTIC_MEASUREMENTS.json` 의 04 항목 두 값(3.359859 / 3.454896)은
   **podA 정지각이 아니다 → 이 두 칸은 판정불가**. 정본은 `W25_RAW_OBSERVATIONS.json` 의 `door_final.stops`(3.3576° / 3.0341°).
3. `tools/diagnostics_w25.py:150` — 확대 창이 `sync 6500..13200` 로 고정돼 podA 의 reclose 정지(sync 13413)가 아래 패널에서 잘린다.

그 밖의 판독 결함: 그림 02 에서 source(63,636) 곡선이 y 범위 위로 벗어나 보이지 않는다
(그림만으로 source 추이를 읽으면 안 되고 `per_frame_counts_recorded` 를 봐야 한다);
그림 02 의 t≈8.9 s 태그 라벨 3개가 겹침; 그림 01·03 제목이 폭에서 잘림; 그림 04 아래 패널 라벨이 제목과 겹침.

**RRD 생략 사유(D341 `AGENTS.md:152`)**: 이 단계는 라벨/인덱스 파생의 코드·배열·해시 감사이고
기하·포즈·접촉·궤적에 대한 새 판정을 내리지 않는다. 재생은 별도 단계(replay-renderer)가 같은 원자료로 수행한다.

## 11. FAIL 요약과 남은 승인 경계

**FAIL 총 8건** (P4 1 + P5 5 + P6 2 — 정착 cadence 는 같은 사실이 세 곳에 나타난다)

1. 정착 창 cadence 계약 미충족 — 5프레임(<6), 최대 간격 0.100025 s(>0.05 s).
   `settlement_frame_dt_s` 는 0.05 로 선언돼 있으나 실제 저장 간격 `particle_frame_dt_s` 는 0.1 이다.
2. 원자료 메타데이터에 ERRATUM_04 선언 문자열 5개가 전부 없음(라벨은 그 식으로 계산됐음에도).
3. 메타데이터 `time_mapping_abs_s` · `geometry_epsilon_m` 없음.
4. `visual_mapping.json` 없음(재생 단계 산출).
5. 바이트 수까지 포함한 최종 매니페스트 없음.
6. `policy.no_threshold_change_after_outcomes` — criteria sha 가 실행 영수증이 아니라 EXEC_PIN 에 기록됨(hard_fail severity).

**판정불가로 남긴 것**

- P6 규약 8항목(재생·readiness·음성대조 산출물 필요).
- `DIAGNOSTIC_MEASUREMENTS.json` 의 `04_door_angle_and_chatter.close_stop_deg`·`reclose_deg`
  — 바이트 사본 도구에 podB sync 인덱스가 박혀 있어 podA 에 대해 의미가 없다(§10-2). 도구는 고치지 않았다.

**승인 경계 — 사용자 몫으로 남는 것**

- podA 배출량은 **251~297알 (5.08~6.02 g) 구간**으로만 인용 가능하다. 정착 cadence 계약이 미충족인 한
  "251알 확정" 단일값 승격은 할 수 없다(`exact_single_value_allowed: false`).
- 저장 간격을 줄이는 것은 **새 revision + 새 criteria 파일**이 필요하다(등록 규약 `reading_rules[3]`: 결과를 본 뒤 값 변경 금지).
  이 보고서는 그런 변경을 제안하지 않는다.
- "전체 사이클 성공" 판단은 재생(2단계)·독립 감사(3단계)를 거친 뒤 **사용자**가 한다(D490 `:316`).
- podA↔podB 차이(배출 251 vs 585, 흘림 123 vs 25, 실제 경과 시간 1.449배)를 **어떤 원인으로 읽을지**는
  이 단계의 산출이 아니다. 같은 입력의 반복은 현재 n=2 이며 인과에는 n≥3 이 필요하다(D490 `:316`).

---

### 비주장

- 이 폴더는 **파생**이다. `run_01` 원자료의 어떤 바이트도 바꾸지 않았다(PRE==POST 증명).
- 재현 불일치 0 은 "기록이 규약대로 계산됐다"는 뜻이며 배출 성공·물리 타당성의 증거가 아니다.
- ambiguous 3,712알은 규약("모든 구체가 margin 안쪽")의 기하적 결과이며 물리 실패가 아니다.
- 어떤 FAIL 도 사후 허용값으로 완화하지 않았다(D485 `:309`).
- 차이는 관측이며 원인으로 읽지 않는다(D490 `:316`).
- 새 물리·GPU·DEME·Isaac·Rerun 실행 0, 설치 0, 버전관리 커밋/푸시 0, 다른 폴더 쓰기 0.

### 산출물 색인

```
postprocess_20260930/
  PRESERVATION_BEFORE.json / PRESERVATION_AFTER.json   P1
  rev34_copy/{rev34,rev29}/… + REVISION_PIN.json       식 무수정 증거 (51 파일, n_bad 0)
  derived_v2_w25/w25_podB_seed460_rev34_derived.npz    P2 파생 라벨·전환·혼동행렬 (파일명은 도구 바이트 사본 탓, 내용은 podA)
                 DERIVED_V2_MANIFEST.json
                 SETTLEMENT_WINDOW_RECOMPUTE.json      P4
                 frozen_tool_attempt/                  동결 derive_v2 크래시 기록 (rc1 x2)
  tests/{independent_check_v2.py,test_allframes_w25.py,w25_paths.py,RESULTS_allframes_w25.json}  P3
  schema_check/RAW_SCHEMA_CHECK_W25.json               P5 (27항목)
               CRITERIA_CHECK_W25.json                 P6 (45항목)
  W25_RAW_OBSERVATIONS.json                            P7
  W25_PODA_VS_PODB_COMPARISON.json                     P8 (장비 짝, tools_new)
  W25podA_VS_W19_COMPARISON.json                       P8 부록 (podB 도구 바이트 사본, 라벨 주의 9-3)
  diagnostics/01~05*.png + DIAGNOSTIC_MEASUREMENTS.json
  inspection.json                                      실제 육안 검수 기록 + 도구 하드코딩 결함 3건
  tools/    podB tools/*.py 바이트 사본 8개
  tools_new/ preservation.py, compare_podA_podB.py (새로 쓴 2개, 동결 식 미접촉)
  logs/PRE.txt POST.txt
  manifest.json                                        경로·바이트·sha256
```
