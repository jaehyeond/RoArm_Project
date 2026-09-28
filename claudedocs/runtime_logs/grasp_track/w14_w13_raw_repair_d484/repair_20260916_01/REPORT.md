# W14 — W13 run_01 원자료 판정 결함 2개의 새 revision(rev29) 수정과 CPU 검수

작성: 2026-09-16. 이번 case의 신규 변수: [원자료 단계 전환/재고 분류 구현 정정]. 물리 조건 신규 변수 0. **새 물리·GPU·렌더·학습·실물 0.** 동결 rev28/run_01/post03은 바이트 단위로 보존했고(§5), 수정은 이 폴더의 rev29 사본에만 있다. run_01 원자료의 규약 FAIL 판정은 그대로다 — 이 폴더의 라벨/전환 인덱스는 **파생(derived)** 산출이며 소급 PASS가 아니다.

## 1. 무엇을 왜

독립 감사 `REV28_PRODUCTION_PARTIAL_RAW_AUDIT_01.json`(14/16)의 FAIL 2개가 대상이다.

1. **전환 인덱스**: 규약 `RAW_SCHEMA_REQUIRED.md:52`는 `transition_sync_index`가 dense `sync_phase_code`가 바뀐 행과 정확히 같아야 한다고 한다(phase 12개 → 11개). rev28 `enter()`는 subphase 진입에서도, `record_t0()`는 행 0에서도 인덱스를 남겨 25개가 기록됐다.
2. **더미 안 분류의 바닥 식**: 선언(`inventory_geometry.INSIDE_RULE`: 모든 구체가 margin 2.5 mm 안쪽)과 달리 z 하한을 구 **최상단**(`z+r > floor−margin`)으로 비교해 바닥에 닿거나 살짝 뚫은 알도 source로 셌다. 규약대로면 구 **최하단**(`z−r > floor+margin`)이어야 한다. 생산자의 자기검증기 `verify_w13_self.reclass`도 같은 식을 갖고 있어 사전 검수가 이를 걸러내지 못했다.

## 2. 절차 (관찰 가능한 순서)

1. 부팅 문서·9/14 재개 문서 §2/§4·감사 JSON·규약/에라타·root 재현(`ROOT_PARTIAL_RAW_SPOT_REPRO_01.json`)을 읽고, run_01 NPZ/JSON·pile NPZ·rev28 26파일·메인 `sim_deme_scoop_s1.py` 해시를 기록값과 대조했다(전부 일치, `PRESERVATION_BEFORE.json`).
2. rev28 전체를 `rev29/`로 복사한 뒤 3파일만 고쳤다(`rev29/DIFF_rev28_to_rev29_src.patch`, 93줄):
   - `src/sim_w13_full_cycle.py`: `record_t0()`의 `trans_idx.append(0)` 제거, `enter()`는 `phase != state["phase"]`일 때만 append. subphase 상태 갱신·강제 입자 프레임 저장은 그대로.
   - `src/inventory_geometry.py`: `in_src` z 하한을 `(S[:,:,2] − Rr) > box[2,0] + marg`로 정정. `SOURCE_FLOOR_RULE`/`REVISION="rev29"` 선언과 `semantics_metadata` 항목 2개 추가. near 밴드·다른 축·우선순위·margin·속도창 불변.
   - `src/verify_w13_self.py`: 자기검증 oracle `reclass`의 같은 바닥 식 정정 + `transitions_phase_only_exact` 검사 추가.
   - 새 파일 `src/raw_transitions.py`(규약 규칙 + 결함 재현용 legacy 규칙), `src/derive_repaired_raw.py`(동결 원자료 → 파생 NPZ). params/criteria/numeric_inputs/COMMANDS는 rev28과 바이트 동일(`rev29/REVISION_PIN.json`: 29파일, 변경 3, 신규 3, 미변경 23).
3. `derive_repaired_raw.py`로 run_01 원자료(읽기 전용)에서 283프레임×20,000알을 rev28 함수(동결 경로에서 import)와 rev29 함수로 각각 재분류하고, phase-only 전환을 재계산해 `derived/w13_cycle_seed460_rev29_derived.npz` + `DERIVED_MANIFEST.json`을 만들었다(252.5 s, stderr 0바이트).
4. CPU 테스트 18개(`tests/`, unittest, pytest 미설치라 stdlib): 결함 재현(rev28 FAIL) → rev29 PASS → 독립식 대조. 독립 검사기 `tests/independent_check.py`는 생산 모듈·scipy·감사 함수를 import하지 않고 Hamilton 곱 회전·clump AABB containment·groupby 전환으로 따로 구현했다(생산: scipy 회전행렬·per-sphere 식; 감사: cross-product 식).
5. 단일 프레임 진단 PNG 3장을 만들고 실제로 열어 보았다(§4). RRD는 생략했다(§6 사유).

## 3. 수치

**전환 인덱스** (`tests/test_transitions.py` 6/6 PASS):

| 항목 | 값 |
|---|---|
| 기록(rev28) 25개 | `[0, 1, 26, 556, 885, 1119, 1186, 1187, 4810, 5161, 7166, 7300, 7336, 7337, 7563, 7892, 8275, 8604, 8835, 10881, 11256, 14875, 15106, 15435, 15818]` |
| rev28 `enter()/record_t0()` 함수를 AST로 추출해 기록 (phase, subphase) 열로 대역 재생 | 25개, 기록과 **정확히 일치**(결함 재현) → 규약과 불일치 = FAIL |
| rev29 같은 대역 재생 | 11개 `[1, 26, 4810, 5161, 7166, 7300, 7336, 8835, 10881, 11256, 14875]` = 독립식(groupby) = 감사 기대값 → PASS |
| 초과 14개의 정체 | 행 0 + subphase-only 전환 13개(각각 phase 코드 동일·subphase만 변경임을 확인) |
| 12 phase 커버 | 전환 11개 + 행 0의 phase 집합 = 0..11 |

**재고 분류** (`tests/test_containment.py` 9/9, `tests/test_allframes_regression.py` 3/3 PASS, 504 s):

| 항목 | rev28(기록) | rev29(strict) | 독립식 |
|---|---:|---:|---:|
| PF0/ID8 라벨 | source | ambiguous | ambiguous |
| PF0/ID8 구 최하단 / 최상단 | −0.000001232287 m / 0.002312257 m (요구 > 0.0025) | 동일 좌표 | 동일 |
| 전 프레임 재계산 vs 기록 불일치 | **0** (parity) | **1,507,161** (= 감사 값) | rev29와 **0** |
| 최종 프레임(PF282) source/bin/tool/spill/in_flight/ambiguous | 19,712/0/0/73/0/215 | 14,350/0/0/73/0/5,577 | 동일 |
| 재닫기 프레임(PF107) tool_residual | 144 | 144 | — |
| 144 cohort의 최종 라벨 | 133/0/0/7/0/4 | 132/0/0/7/0/5 | 동일 |
| 매 프레임 라벨 합 = 20,000·코드 0~5 | ✓ | ✓ | ✓ |
| 파생 NPZ `inventory_code_rev29_strict` array-sha | — | `d617b71c…eab781` (테스트 재계산과 일치) | — |

합성 단위 사례(둘 다 실행, 규약 기대값 대조): 바닥에 정확히 놓인 알 → rev28 source(FAIL)/rev29 ambiguous(PASS); 중심 z=−1 mm 관통 알 → rev28 source/rev29 ambiguous; 최하단 = floor+margin 정확히 → rev29 ambiguous(strict `>`); 완전 안쪽·측벽 밴드·바닥 아래 spill·이동(≥ 0.03924 m/s) in_flight·90° 회전 알은 두 판 모두 규약대로.

**규약대로 고친 결과의 의미(판정 변경 아님)**: 바닥에 놓인 층은 이제 source가 아니라 ambiguous다(PF282에서 5,362알, 전 프레임 5,264~5,362). 이는 "모든 구체가 floor+2.5 mm 안쪽"이라는 사전 규약의 직접 결과이지 물리 판단이 아니다. 규약을 바닥 접촉을 허용하도록 바꿀지는 별도 결정이다(§7).

## 4. 진단 그림 (실제 열어 본 관찰)

- `diagnostics/pf0_id8_floor_side_view.png`: ID8의 7개 원(XZ 측면)이 z=0 바닥선에 닿아 있고(최하단 −0.0012 mm), 가장 낮은 원의 윗점 2.312 mm는 +2.5 mm(녹색 점선, rev29 기준)보다 낮지만 −2.5 mm(적색 점선, rev28이 비교한 선)보다 높다 — rev28 predicate가 참, strict predicate가 거짓임이 그림으로 확인된다.
- `diagnostics/inventory_counts_timeline.png`: source 수는 기록 19,8xx→19,1xx→19,7xx, rev29 14,5xx→13,8xx→14,35x로 같은 모양이고 약 5,300의 일정 오프셋(바닥층). ambiguous는 기록 183~580, rev29 5,450~5,900. 실선 11개(phase 전환)와 점선 25개(기록)가 구분된다.
- `diagnostics/phase_transitions_strip.png`: 25개 회색 ▼ 중 approach 안 4개·transport 안 5개·return_home 안 3개·1186/1187·7337이 subphase 전환이고, 검은 ▲ 11개가 phase 계단의 오름 지점과 일치한다.

## 5. 원본 보존과 해시

- run_01 `w13_cycle_seed460.npz` SHA256 `529f422e…c46b0f`, `.json` `e482b939…d9340`, pile `659d6b0b…8812`: 작업 전(`PRESERVATION_BEFORE.json`, 64파일)과 테스트 종료 시(`test_03_frozen_inputs_unchanged_after_work`) 모두 일치. rev28 26파일 = `REVISION_PIN.json`과 일치. 최종 사후 해시는 `PRESERVATION_AFTER.json`.
- 동결 폴더에 새 파일을 만들지 않았다(`python -B`, `sys.dont_write_bytecode`).
- 파생 NPZ SHA256 `cdf11a36…cc667d` (800,671 bytes). 원시 좌표/ID/쿼터니언/속도/시간은 파생 NPZ에 복사하지 않고 원본을 가리킨다(`provenance_json`).

## 6. 한계

- rev29 sim 코드는 **물리를 다시 돌려 검증한 것이 아니다**. `enter()/record_t0()` 수정은 기록된 (phase, subphase) 열에 대한 대역 재생으로 검증했다. 다음 실제 실행(GPU)은 별도 승인이다.
- 바닥 층 ambiguous 전환(5,3xx알)은 규약의 결과다. 규약 자체(바닥 margin)를 바꿀지는 사용자 결정이며 여기서 바꾸지 않았다.
- **추가 관찰(수정 범위 밖)**: 규약 `RAW_SCHEMA_REQUIRED.md:42` "모든 phase 전환 sync에 입자 프레임"에서 인덱스 규약이 모호하다. 생산 `enter()`는 전환 **직전 행(i−1)** 에 강제 프레임을 남기고(경계 순간의 물리 상태), 전환 인덱스 i는 새 phase의 첫 행이다. 11개 전환 모두 i−1에 프레임이 있고, 정확히 i에 프레임이 있는 것은 7336(reclose_end 결정 행)뿐이다. 기록 25개도 마찬가지로 i에 프레임이 있는 것은 0·1186·7336뿐이다. 생산자 자기검증 `transitions_have_particle_frames`(`set(tr) ⊆ pf`)는 run_01에서 실행되지 않았고(러너가 step1 뒤 정지), 실행됐다면 FAIL이었을 것이다. rev29는 저장 동작을 바꾸지 않았고 테스트는 실제 규약(i−1)을 고정했다. 규약 문구를 "i−1(경계 상태)"로 명시할지, 생산자를 "새 phase 첫 sync 뒤 저장"으로 바꿀지는 별도 결정 대상이다.
- RRD 생략 사유(D341): 이 case는 같은 원시 좌표에 대한 라벨/인덱스 파생의 코드·배열·해시 감사이며 새 기하·접촉·궤적 판정을 내리지 않는다. 단일 프레임 진단은 §4의 PNG로 남겼다.
- 독립 Codex 감사(Orca, gpt-5.6-sol high)의 결과는 `orca/` 및 감사 worktree `.../w14_w13_raw_repair_d484/repair_20260916_01/audit/`에 별도 기록한다(세션 문서 참조).

## 7. 다음 승인 경계

1. 규약 결정: 바닥 접촉 알을 source로 볼 것인가(규약 개정) / 현행 strict 유지. 전환 프레임 인덱스 규약(i vs i−1) 명시.
2. 재생 3결함(DOOR_STOP PF282 오연결·결정 PNG 더미·Isaac 관절 출처 338/283)의 새 revision 수정 범위 브리핑 → GPU 재렌더는 명령/입력/시간상한 제시 후 승인.
3. 144 cohort 운반 보유 원인 조사(별도 case). 이번 strict 라벨로도 cohort는 최종 132 source/7 spill/5 ambiguous로 용기 도착 0이며, 원인은 확정하지 않았다.
4. rev29로 실제 물리를 다시 돌리는 것은 dt 실행안(`claudedocs/research/dt_expansion_plan_20260916/DT_EXPANSION_PLAN.md`)과 별개로 승인 필요.

## 8. 추가(같은 날 저녁) — 사용자 결정에 따른 규약 v2(ERRATUM_04)와 rev30 파생

사용자 결정(2026-09-16 저녁): 1번 = 바닥(상자·용기)을 받침면으로 취급, 2번 = 전환 프레임은 i−1 규약으로 문구 명시. 이번 case의 신규 변수: [바닥=받침면 규약 v2 파생 라벨]. 새 물리 0.

- **초안** `contract/RAW_SCHEMA_REQUIRED_ERRATUM_04_DRAFT.md`: 상자 바닥과 용기 안바닥의 하한을 구 최하단 > 바닥−2.5 mm(파묻힘 허용 = 동결 margin 재사용)로, 옆벽·상자 윗면·테두리·공구 입구는 strict 유지, near 밴드·우선순위·속도창·spill 불변, 전환 i는 i−1 경계 프레임 필수(정확히 i는 선택). 감사 worktree의 계약 폴더 등록은 Codex 감사가 수행.
- **rev30** (rev29 사본, 2파일 변경: `inventory_geometry.py`의 `in_src`/`in_bin` z 하한 두 줄 + 의미 상수, `verify_w13_self.py`의 같은 두 식 + `transitions_have_boundary_frames`(i−1) 검사; params 등 바이트 동일). `rev30/DIFF_rev29_to_rev30_src.patch`, `rev30/REVISION_PIN.json`.
- **파생 v2** `derived_v2/w13_cycle_seed460_rev30_derived_v2.npz` + `DERIVED_V2_MANIFEST.json`(260.1 s CPU, stderr 0):

| 항목 | 기록(rev28) | rev29 strict | **rev30 v2** |
|---|---:|---:|---:|
| PF282 source/bin/tool/spill/in_flight/ambiguous | 19,712/0/0/73/0/215 | 14,350/0/0/73/0/5,577 | **19,712/9/0/73/0/206** |
| vs 기록 불일치(전 프레임) | 0 | 1,507,161 | **964** |
| vs rev29 불일치 | | | 1,508,125 |
| PF0/ID8 | source | ambiguous | source |
| 144 cohort 최종 | 133/0/0/7/0/4 | 132/0/0/7/0/5 | 133/0/0/7/0/4 |
| 감사 용기 후보 11개 중 receiving_bin | 0 | 0 | **9** (16778·17457은 ambiguous) |

- **964개 차이의 정체**: 전부 용기 쪽이다(ambiguous→receiving_bin 948, ambiguous→in_flight 16; 프레임 168~282, ID 10개). 상자 바닥 규칙 변경은 이 run의 어떤 프레임에서도 기록 라벨을 바꾸지 않았다(source 곡선 기록=rev30 완전 일치, `diagnostics_v2/inventory_counts_v2.png` 실제 확인). 즉 rev28의 "최상단" 버그는 이 run에서는 실질 영향이 없었고, 문제는 rev29 strict가 만들었던 바닥층 ambiguous와 용기 첫 층을 셀 수 없던 구조였다.
- **용기 9알(0.182 g)의 의미와 한계**: 최종 프레임에서 속도 ≤ 5.5e-6 m/s로 정지해 있고 용기 바닥 위에 있다. 그러나 동결 정착 판정(0.25 s 창·프레임 간격 0.05 s)은 run_01에서 cadence 미충족이므로 "정착 배출 9알 확정"이라고 하지 않는다. 라벨 v2 기준의 기하 분류일 뿐이다.
- **cohort 관찰(범위 밖, 후속 case 단서)**: 용기에 도달한 10개 ID는 재닫기 프레임에서 기록상 tool_residual(144개)이 아니라 **near_tool 밴드의 ambiguous**였고(z 105~120 mm, 립 근방), 운반 끝~배출 중(t 14.4~16.3 s)에 용기로 들어갔다. 즉 실제 운반된 알 집합은 144개 cohort보다 넓다. 144개의 행방 조사 시 ambiguous 밴드까지 포함해야 한다.
- **테스트**: `tests_v2/` 합성 6사례 + PF0/ID8 + 표본 4프레임(6/6 PASS, `RESULTS_containment_v2.json`); 전 프레임 rev30 = 독립식 v2(floor_rule="support") 전 프레임 rev30=독립식 v2 0 불일치·파생 NPZ/해시/동결 입력 일치 **2/2 PASS**(688.5 s, `tests_v2/RESULTS_allframes_v2.json`).
- **독립 Codex 감사 #2**(ERRATUM_04 등록 + rev30 v2 재계산): Codex #2 결과 — ERRATUM_04 **등록 완료**(`.../resume_20260913/audit/RAW_SCHEMA_REQUIRED_ERRATUM_04.md`, SHA `4290a9cf…e8e1a6`, 초안 SHA `665fb7cb…8429` 대조, 의미 불변·문구만 정리). NumPy-only 재계산(검사기 SHA `d8297814…8a0b`, 35.7 s): rev30 파생 대비 0 불일치·raw 964·rev29 1,508,125·최종 19,712/9/0/73/0/206·cohort 133/0/0/7/0/4·후보 11 중 9 재현, diff 2파일 예상 밖 0, 보존 0 불일치, 정착 비주장 명시 — 모두 PASS. 단 **종합 FAIL**: rev30 메타데이터 선언 문자열 2개(`classify_floor_rule`, `classify_contract_version`)에 설명 괄호가 붙어 등록 계약의 exact 문구와 불일치. → 조치: **rev31** = rev30 + 두 문자열만 exact 문구로(`rev31/DIFF_rev30_to_rev31_src.patch`, 라벨 식 불변), `derived_v2_rev31/` 재파생 → rev30 라벨 배열과 **bit-identical** 확인(`np.array_equal` True). 같은 Codex 터미널에 rev31 재검증(task 후속) 배정: (Codex #3 dispatch `ctx_3c3e01c02dd5`, task `task_dce9f1711fea`, 같은 터미널 `term_c29b5655-…852` 재사용) **rev31 재검증 PASS**(`audit/W14_REV31_V2_INDEPENDENT_AUDIT_02.json`, E1 exact 문자열 일치·E2 변경 2파일/`classify_spheres` AST 동일·E3 5,660,000 라벨 rev30과 bit-identical(배열 SHA `a9ad7356…e46e`)·E4 raw/pile/rev28~rev31 핀 보존 0 불일치; 새 터미널 `ctx_bfd27bf3fd05`, 첫 재사용 시도 `ctx_3c3e01c02dd5`는 codex-interactive-prompt로 시작 실패). → **규약 v2(ERRATUM_04) 파생 정본 = rev31 + `derived_v2_rev31/`**; rev30 파생은 라벨 동일하나 선언 문자열 불일치 기록으로 보존.
