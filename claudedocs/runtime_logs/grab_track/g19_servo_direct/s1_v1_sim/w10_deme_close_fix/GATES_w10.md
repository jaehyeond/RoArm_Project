# GATES_w10 — 사전 등록 (2026-09-10 밤, 실행 전에 작성)

W10 = 렌즈 클럼프 더미(W7, 20,000알)에서 실물 절차 "문을 서보 토크 정지까지 닫기" 가 4/4 발산한 문제(W8 §3-B)를
① 발산 기작 특정 → ② 폐합 속도 절반 → ③ 강성 사다리 순으로 푼다. 중앙 셀(0,0)·seed 460 만 쓴다.
판정은 전부 셀 결과 파일(`cell_*/scoop_s1_seed460.json`, `cell_*/timeline_seed460.json`, `cell_*/diverge_event_seed460.json`)을
`gates_w10_writer.py` 가 읽어 `gates_w10.json` 에 쓴다. 이 문서는 실행 전에 고정하며 실행 후 수정하지 않는다.

> 용어: *sync* = DEME 를 몇 스텝 돌린 뒤 파이썬이 상태를 읽는 간격. *유령 접촉(ghost contact)* = 구 중심이 삼각형 면 뒤쪽(h<0)에 있는데
> 투영이 삼각형 안이라 DEME 양면 커널이 관입 r+|h| 로 잡는 것(W3b §1). *물림(squeeze)* = 두 립 사이에 알이 끼어 힘이 서서히 쌓이는 것.
> *pop* = 알 하나가 비물리적 속도로 튀는 것. *culprit* = pop 순간 최대속도 owner(알).

## 공통 고정 설정 (W8 옵션 E 그대로 = 사전 등록 설정 + 폐합 sync 1 ms + 물림 가드 3 N)

| 항목 | 값 | 출처 |
|---|---|---|
| 더미 | `pile_lens6_a4p5_b3p8_c2p5_slab_n20000_seed460.npz` (sha16 `659d6b0bc771678a`) | W7 |
| 물성(임시 = MEASURE) | mu 0.45 · Crr 0.06 · CoR 0.3 · nu 0.3 · ρ 905 · 툴 E 3e9 · **E 5e6**(①②) | W8 |
| 절차 | 립→문 서보 30°→잠김 25 mm(25 mm/s, 팔 6 N)→닫기(서보 정지 1.764 N·m)→+80 mm→재닫기 | D481 · W8 |
| 폐합 | sync 1 ms · 물림 가드 3 N · **문 하한 없음**(옵션 F 의 5° 하한 제거) | 지시서 |
| 적분 | dt 1e-5(①②) · CD 20 · 천장 0.5 m | W8 |
| 셀 | 중앙 (0, 0) mm · seed 460 | 지시서 |
| 발산 판정 | DEME abort(60 m/s 규칙) **또는** 파이썬 pop-stop: 입자 최대속도 > 20 m/s(`diag_stop_v_m_s`). ①②③ 은 DEME `error_out_vel` 을 1e4 로 올려 C++ terminate 대신 파이썬이 멈춰 상태를 남긴다 — 물리(힘·적분)는 바꾸지 않는다 | 이 문서 |
| 셀당 상한 | ①② 벽시계 5400 s · ③ 14400 s(4 h) · 300 s 무진행(stdout 갱신 없음) 스톨 가드 + 1회 재시도 | 지시서 |
| GPU | 시작 전 free ≥ 3 GB, 아니면 60 s 대기 반복 최대 30 분 | 지시서 |

## 단계별 신규 변수 (변수 사다리)

| 단계 | 셀 | 직전 단계 대비 신규 변수 | 목적 |
|---|---|---|---|
| ① 진단 | `cell_diag_c` | 진단만: q<6° 구간 sync 0.1 ms + 립 근방 링버퍼 + pop 순간 덤프 (물리 불변) | 발산 순간의 culprit·접촉 삼각형·관입 기하 확보 |
| ② D+E | `cell_DE_c` | 폐합 각속도 45 → 22.5 °/s (실물 12 °/s 방향) | 속도 의존(관성/한 sync 관입) 여부 |
| ③ 강성 | `cell_DE_E1e8_c` | E 5e6 → 1e8 · dt 1e-5 → 2e-6 (②의 나머지 동일) | 무른 구 립 면 통과(E 의존) vs 유령 접촉(E 무관, 힘 ∝ E) 분리 |
| ④ | (실행 금지) | 립 기하 변경 설계 제안만 | — |

②는 ①이 끝나면 무조건 실행한다(①은 진단이라 완주를 기대하지 않음). ③은 ②가 발산했을 때만.

## 게이트

| # | 게이트 | PASS 조건 | 비고 |
|---|---|---|---|
| G0 | 구 더미 회귀(기존 경로 불변) | 구 더미 + W3b `params_fixnorm_plunge25.json` seed 460 → 완주(diverged false) · `capture.n_in_cavity` ∈ [268, 362](= W3b 315 ±15 %, W8 §3-A 권고) · 문 정지 reason 전부 `servo_stall` | 코드 변경이 params 게이트 뒤에만 있음을 확인. 범위 밖이면 FAIL |
| G1 | 발산 기작 특정(①) | `cell_diag_c/diverge_event_seed460.json` 존재 · culprit 의 직전 3 sync 표(위치·최대 접촉점·관입·삼각형 그룹) 존재 · 기작 분류가 {ghost, squeeze, sphere_sphere, unresolved} 중 하나로 파일에서 계산됨 | 분류 규칙(사전 고정): 직전 3 sync 중 culprit 구 하나라도 툴 삼각형에 대해 h<0 ∧ 투영 안 ∧ |h|<r 이면 **ghost**; 툴 두 메시 접촉력이 3 sync 연속 상승해 pop 직전 단일 ≥ 2 N 이면 **squeeze**; culprit 최대 힘 pair 가 구–구이면 **sphere_sphere**; 그 외 **unresolved**. `unresolved` 도 G1 은 PASS(파일·표 존재가 조건) 이나 보고서에 명시 |
| G2 | 토크 정지 완주 1셀 | ②③ 중 한 셀이 diverged false · rc 0 · 상한 내 · 문 정지 reason ∈ {`servo_stall`, `reached_close_end`}(`pinch_guard` 는 완주지만 **토크 정지 아님** 으로 따로 표시) | `door_floor` 없음(하한 미사용) |
| G3 | 포획 질량 > 0 | G2 셀 `capture.mass_g > 0` | 충전율은 보고값 |
| G4 | 비교표 | G2 셀 vs W8 옵션 F(`w8_deme_scoop_lens/cell_F_c`): 포획 질량·문 정지각·물림 알 수·립 등가 힘 피크·절단면 각(4방위, rest 후처리) 이 `gates_w10.json` 에 나란히 있음 | 보고 항목. 완주 셀이 없으면 "없음" 으로 기록 |
| R | D341 Rerun 1셀 | G2 셀(없으면 ① 진단 셀) → RRD/RBL/검증 JSON/스크린샷/육안 검수 JSON | 완결 항목 |
| S | D470 source sha | 스크립트·백업·params·더미 npz·STL·design.json sha 가 결과 JSON `inputs_sha16` + `gates_w10.json` 에 있음 | |

`all_pass` = G0 ∧ G1 ∧ G2 ∧ G3. G4·R·S 는 보고/완결 항목.

## ① 진단 기록 정의 (코드 `sim_deme_scoop_s1.py` diag 블록이 정본)

- q < 6°(폐합·재폐합) 에서 sync 를 `diag_fine_sync_s` = 0.1 ms(10 스텝) 로 줄인다(문 각속도·물성 불변, q 감소량은 sync 길이에 비례).
- 매 sync 행: 최대속도 owner id·위치(세계 mm)·속도, 고정부/문 각각의 최대 접촉점·힘 벡터, 그 접촉의 최근접 구(owner, 구 번호)·최근접 삼각형(id, 그룹)·면 부호거리 h(+ = 바깥)·투영-안 여부·기하 관입(r − 최근접거리)·유령 플래그(h<0 ∧ 안 ∧ |h|<r)·Hertz 역산 관입 δ_F = (3F/(4E*√r))^(2/3).
- 삼각형 그룹 = 셸 생성 순서로 태깅: `inner`(보울 안쪽 면)·`outer`(바깥 면)·`cap_in`/`cap_out`(뺨 안/밖)·`part_lip`(x=8.1 파팅면의 립 띠, θ=0)·`part_bottom`(바닥 띠, θ=π)·`part_cap`(파팅면의 캡 단면). 검산: 접촉점 ↔ 삼각형 거리 < 2r.
- 링버퍼: 최근 12 sync 의 (t, q, 툴 노드, 툴 접촉점·힘, 립선 ±12 mm 안 클럼프의 위치·자세·속도).
- pop 이벤트 트리거: 입자 최대속도 > 2 m/s 또는 단일 접촉 > 2 N → 링버퍼 + 현재 상태 + culprit 직전 3 sync 표 + `GetContactDetailedInfo`(try, 벽시계 기록) 를 `diverge_event_seed460.{json,npz}` 로 덤프. 이후 최대속도 > 20 m/s 면 pop-stop(발산 처리).
- 발산 직전 3 sync 표 = 이벤트 JSON `culprit_last3` (sync 당 1행).

## 완주 셀 비교표 정의 (G4)

W8 옵션 F 중앙 셀(`cell_F_c/scoop_s1_seed460.json` + `crater_rest_seed460.json`) 과 같은 키: `capture.mass_g`·`n_in_cavity`·`door.stops`·`q_final_deg`·`lip_gap_final_mm`·`n_pinched_at_lip`·`forces.close_peak_lipF_N`·절단면 각(rest 후처리 `crater_rest_writer.py` 재사용, 4방위·None 사유·시컨트). 렌더 타임라인 0.05 s → `render_timeline_w10.npz`(완주 셀; 진단 셀은 `render_timeline_w10_diag.npz`).
