# 2026-08-06 (22nd) — G0b T3: 재검증 회수 → 라운드-2/3 수리·검증 → ③ prereg → Isaac attempt1~4 (파지 실패 = 충돌 자산 조 목구멍 폐색 실증)

이번 case의 신규 변수: [기존 ②(T3 물리 시행) 범위 내 — 신규 변수 축 없음]

Case `g0b_d420` 계속. 로봇 HW 0 · lerobot-train 0 · git commit/push 0 ·
**Isaac 기동 4회(attempt1~4, 전부 사전등록 tuple)**. 승인 신규 0(19th "T2/T3 진행
승인" 범위 내). 세션 진행 규칙: 실패 가능 실험 = 사전비행 v2(1회 FAIL→수리→PASS) +
Isaac 물리 시행 4회(APPROACH_FAIL / LIFT_FAIL / LIFT_FAIL / APPROACH_FAIL).

## §0 이중 세션 겹침 (기록)

부트 직후 21st 세션(별 터미널)이 살아서 `wf_3cea04db-7c2` 완주 통지를 자체 회수하고
12:36~12:39에 상태 문서(§8·D423-R1·START_HERE·LEDGER·MEMORY)를 갱신함을 관측 —
편집 충돌 없음 확인(p9는 본 세션만 편집; 21st는 상태 문서만). 21st의 회수 결론과
본 세션의 독립 회수(journal 직접 판독)가 전건 일치 확인. AGENTS.md 동시 세션 금지
규칙 위반 소지가 있는 상황이었으므로, 이후 세션 종료 시 이전 터미널 종료 확인 권장.

## §1 wf_3cea04db-7c2 회수 (10/10, 에러 0)

- 부트 시점 8/10 회수(수리 반박 8건 전부 repair_effective=true), 회귀 2렌즈는
  실행 중 관측(전사 mtime 추적, 재발사·resume 없이 완주 대기 — 17th 교훈 준수).
- 최종: **수리 유효 8/8·notFixed 0, 생존 = MAJOR 고유 2종(2 agents 교차 수치 재현)
  + MINOR 9종(중복 제거)**. 전문 = `g0b_d420/p9_reverify_wf_3cea04db_findings_raw.json`
  (21st 세션이 영속화, 본 세션 파싱 대조 일치).

## §2 라운드-2 수리 (MAJOR 2 + MINOR 다수, p9 1,605→1,714줄)

| 이슈 | 수리 |
|---|---|
| **MAJOR ② transit 명령 잔차 2.52~2.55mm**(min-tilt-in-band 선택) | 선택 밴드 분기: require_tilt=False → `TRANSIT_POS_BAND_MM=0.5` + **bias-free 위치 폴리시 2단 솔브**(`TRANSIT_POLISH_DEV_DEG=2.0`, 비악화 가드). 근거 실험 = **trust region 스윕**(12~24°에서 잔차 1.2~1.5mm 불변 → 클립이 아니라 바이어스-위치 평형이 원인; `g0b_d420/t3_trust_region_dev_sweep_evidence.md`) |
| **MAJOR ① wp002 관절 46° 점프**(슬루 피크 9.90mm vs 10mm) | 2층: `--waypoint_max_joint_dev_deg 12`(체인 trust region — hop 연속성) + **`--transit_tcp_step_gate_m 20mm`(approach phase 한정 스코핑)**. 자체 분석: trust region은 hop 길이만 줄이고 순간 \|J·q̇\|은 못 줄임 → 게이트 스코핑이 정직한 수리(물체 보호 게이트 전부 불변; 현실 effort-limited 슬루는 클램프의 ~1/6 — attempt1 실측 max_tcp_step 1.7mm로 사후 확증) |
| MINOR | `_close_all` 핸들별 try+선클리어 / app_id·recording_id tag 파생 / `_verdict(lift_path_ok)` / aggregate lift_follow sanitize / 솔버 밴드 min(3.0, gate) / docstring 3건(D-8 60s 3rd supersession·미래형 provenance·transit 서술) |
| 처분(무변경) | stall 채터 보수 창 / 세그먼트 drift 기준 / nan_seen run의 NaN 토큰 — prereg §10 등재 |

## §3 사전비행 v2 (신규, FAIL→수리→PASS)

- `g0b_d420/t3_preflight_ik_chain_v2.py` (sha `f44c5a45…4c16`; v1은 증거 보존).
  4포즈 × REACH+3체인, **마진의 게이트화**: 명령 pe≤0.6mm / dev≤14° / 비관 슬루
  approach≤14mm·회랑≤9mm / 예산<6000.
- 1차 실행 **FAIL**(중대: mid-transit pe 1.0~1.5mm — 폴리시 도입 동기 / 회랑 슬루
  게이트 7.5mm 과소교정 → 9.0mm 재교정: 8mm 명령 간격 자체가 설계점).
- 수리 후 **PASS**: worst pe 0.468~0.492mm·dev≤12.05°·슬루 9.59/8.07mm·도착 tilt
  ≤0.213°·예산 3435~3735.

## §4 라운드-3 적대 재검증 `wf_9b819983-97c` (9 agents, 에러 0, 1.107M tok)

- **수리 7/7 유효·notFixed 0·회귀 2렌즈(런타임 전문 정독/사전비행 정합) 발견 0.**
  하이라이트: R1 — 폴리시 채택 22~25/40~45wp·tilt 악화 ≤+0.261°·**plan 15해 라운드-1과
  비트 동일(T2/D421 REACH 불변)**; R2 — property test 400회 리밋/dev 위반 0; R3 —
  명령 경로 원통 진입 z=top+55.7mm 기하 논증.
- 생존 = **MINOR 3**(전문 = `g0b_d420/p9_reverify2_wf_9b819983_findings_raw.json`):
  path_ok 미직렬화 / 예산 3675 표기 stale / polish 상수 gates 미수록 → **전건 기계
  수리**(path_ok 3종 JSON 직렬화 + 3735 정정 + transit_polish_dev_deg 수록).
  4차 전면 재검증 생략(동작 무변경 — diff 자체검증+py_compile+사전비행 재PASS;
  prereg §8-6에 판단 사전 고지).
- **attempt1 최종 sha `939a5bd0639332afc2572bee8cd7a7e735ad2ed7080193d4f605114cc1e2f5fb`**
  (1,747줄).

## §5 ③ prereg 발행 → Isaac attempt1~4 (전부 사전등록 tuple, 사후 인자 변경 0)

`g0b_d420/t3_prereg.md`: 본문(attempt1) + 부록 A/B/C(attempt2/3/4 — 각각 직전
attempt의 실측이 유일한 설계 입력임을 명기). supersession 3건(D-4 스폰→seed0_S1 /
D-6→attempt3 / **D-8 episode 20→60s**), p7-parity 델타 11건(라운드-1 6 + 라운드-2 5),
접촉 화살표 생략 정당화(사용자 확인 플래그), 마찰 주 leg 0.40/0.30.

| attempt | tag / sha / 핵심 인자 | verdict | 기전 실측 |
|---|---|---|---|
| 1 | `t3_grasp` / `939a5bd0…` / 기본값(margin 0.5mm) | **APPROACH_FAIL**(descend 승계) | approach 44wp 완주(도착 tilt 0.200° PASS) → descend wp006에서 **TCP z=0.05440 정지**(오차 3.917mm 포화·tcp_step→2e-6), 물체 speed 지터 20~30배·tilt 지터 급증·drift 26μm = **개방 조 구조물이 상면 접촉**(수직 압입은 지면 지지). 슬루 실측 max 1.7mm(비관 모델 9.9mm의 1/6 — R3 예측 확증) |
| 2 | `t3_grasp2` / `1ef8a411…` / margin 5.5mm·marker 0.035(CLI 인자화+가드) | **LIFT_FAIL**(5/5 완주) | descend 통과 → **close 88.31→24° 전 각도 무접촉**(stall=NO·drift 2μm·지터 0) → marker 39°~ 발화(거리 휴리스틱 — 접촉 아님 실증) → LIFT에서 TCP 상승·물체 부동(follow −0.0mm). attach 시도 144회 전부 무력화 감시(posewrite 0) |
| 3 | `t3_grasp3` / `99c99c65…` / +descend_open 45°(D-1 부분 개정: 하강 개방 인자화, 기본=동결 OPEN) | **LIFT_FAIL** | attempt2와 동일 무접촉 시그니처 — 부분 개방으로도 닫힘 면이 TCP−5.5mm 평면 미도달 |
| 4 | `t3_grasp4` / `99c99c65…` / margin **−7.5mm**(top−7.5mm 목표) | **APPROACH_FAIL**(descend) | **정지 z=0.054394 — attempt1(88.31° 개방)과 동일** → 하강 한계는 **개방각 무관·축 근방 고정 구조물** |

- D341: 4회 전부 RRD/RBL/PNG/validation(pass=True) 생성 + **육안검수 4회 완료**
  (관찰 기록: 헤더-stdout 수치 전건 일치 / q5 계단·stall 부재·접촉 지터 시각 확증 /
  phase 이벤트 정합 / 사소: 3D 뷰 기본 카메라가 씬 하단 크롭 — 표시 전용).
- **관측 1건**: verdict FAIL인데 프로세스 exit 0 (4회 전부) — Kit `sim_app.close()`가
  `return 2` 도달 전 프로세스 종료 추정. 증거 파일은 close 전 기록 완료라 무손상.
  **판정 권위 = stdout verdict 라인 + results JSON**(prereg 부록 A1에 격하 고지).

## §6 종합 결론 (T3 판별 완성 — 물리 verdict + 자산 결함 실증)

1. **동결 attempt3 충돌 자산에서 top-down 상면 중심 파지는 기하 불가능**:
   ① TCP 하강 한계 = top+4.4mm(개방 88.31°/45° 동일 → 축 근방 고정 구조물)
   ② 그 위에서 close 88→24°/45→24° 전 구간 무접촉(파지면이 그 깊이 미도달).
2. **T1 실물 rim 핀치(상면 0~12mm 물림) 성공과 정면 모순** → sim 조/목구멍 충돌
   형상이 실물과 어긋남 = **attempt3 64-part 분해의 목구멍 폐색 잔존 실증**.
   (D420 ⑥이 local convex_hull의 폐색을 이유로 attempt3를 채택했으나, attempt3의
   목구멍 개방성 자체는 검증된 적 없었음 — 이번 3연속 시행이 그 검증이었다.)
3. 교수님 지시 ②("collision으로 잡히는 걸 찾아라")의 이행 사례: 시행 4회가 설계
   파라미터(하강 한계 4.4mm·개방 무관성·무접촉 밴드)를 실측으로 내놓았다.

## §7 다음 세션 (순서)

1. **조 충돌 형상 감사(읽기 전용)**: isaaclab env pxr로 attempt3 physics 레이어의
   link5/gripper_link 64+64 파트 정점을 링크 좌표로 덤프 → q5 45°/24° FK 변환 →
   (a) 축 근방(r<14.5mm) 최저점(= +4.4mm 바닥의 정체 파트 특정) (b) 조 면 간극-깊이
   프로파일. **재분해가 아니라 진단** — D415 ③ 저촉 없음. 산출은 `g0b_d420/`.
2. **사용자 에스컬레이션**: (a) 재분해 금지(D415 ③) 해제 여부 (b) 대안(TCP/타깃
   의미 재유도·부분 파트 비활성 등) — 자산 수리 없이는 T3 GRASP_PASS 불가가 실증됨.
3. 자산 해결 후 attempt5 재사전등록(부록 D) → T4 실물 재현 대조.

## §8 산출물

p9(1,780줄, attempt3/4 sha `99c99c65…2412`), t3_prereg.md(+부록 A/B/C),
t3_preflight_ik_chain_v2.py, t3_trust_region_dev_sweep_evidence.md,
p9_reverify2_wf_9b819983_findings_raw.json, t3_grasp{,2,3,4}_* 32파일 + stdout/stderr
8파일, 본 doc, START_HERE/LEDGER/DECISIONS(D424)/MEMORY 갱신.
