# Session 2026-07-10 (late) — D329: G0a wrong-object audit + cylinder case redefinition

Verdict: `D329_G0A_WRONG_OBJECT_CASE_REDEFINE` (사용자 승인 완료)

이번 case의 신규 변수: [물체 기하: 10cm cube -> cylinder D34 x H90 — 사용자 승인
2026-07-10, 구현/실행은 D330에서]

## Session progress rule 준수 노트

이 세션은 sim 실험을 돌리지 않았다. 명시적 정당화: D322-D328의 7세션 연속 0/10
실패에 대해 "잘못된 물체" 가설이 제기되어, 새 실험 전에 기존 증거의
결정-변경(decision-changing) 감사가 선행되어야 했다. 감사 결과 case 재정의가
승인되었고, failable 실험은 D330 원통 정렬 probe로 사전 등록되었다(아래 설계).
이 감사는 "검증이 결정을 바꾼" 사례다: D328이 권고한 큐브 대상 collision sweep
audit이 취소되고 물체 교체로 대체되었다.

## 검증 방법

- 4개 병렬 검증 에이전트(probe 내부 / D328 수치 vs 로그 / 상태문서 정합 /
  waypoint 기하 + offline repo-FK 재계산) + 메인 세션 직접 코드 리딩.
- 오프라인 FK 재계산 스크립트(검증용, repo 밖 scratchpad): repo의
  `_fk_runtime_tcp` 체인 사용, Isaac 불필요.

## 핵심 판정

### 1. G0a는 D322부터 tap 트랙 10cm 큐브로 돌았다 (CONFIRMED)

- d322 probe가 `RoArmCubeTap10cmEnvCfg`를 import
  (`sim_scripts/cube10cm_top_view_d322_grasp_g0a_alignment_probe.py:222`),
  sponge를 (0.10)^3으로 리사이즈 (`:237-240`).
- d327 동일 (`:318`, `:332-336`); mass 0.72는 probe `:336` + env 기본값
  `roarm_rl/roarm_cube_push_env.py:41,266` (CUBE10CM_MASS_KG=0.720) 이중 설정.
- 원통은 어떤 grasp probe에서도 spawn된 적 없음. 전 probe non-goals에
  "no cylinder".

### 2. 큐브는 세션 드리프트가 아니라 D322_PROMPT가 못박은 설계 (CONFIRMED, 정정)

- `claudedocs/D322_PROMPT.md:60` "불변: 기존 10cm 큐브(개구 ~45mm < 100mm —
  파지 불가, 정렬만 검증)"; `:79-80` Non-goals "원통/신규 물체 스폰" 금지.
  출처: 채팅 인수인계 문서 v5 (2026-07-08) 11절.
- 교수님 지시 #6 "원통 등 잘 잡히는 형태부터"
  (`claudedocs/direction_20260708_grasp_pivot.md:10`), 실용 개구 40-45mm
  (`:28`)와의 긴장은 D322 프롬프트 설계 단계에서 들어갔다.
- 물체 선택이 위험으로 재검토된 기록은 DECISIONS D322-D328, 세션 문서,
  BACKLOG 어디에도 없음 (invariant로만 기록).

### 3. 100mm 큐브에서 정렬 목표는 구조적으로 충돌을 요구한다 (CONFIRMED)

- 목표 TCP `(0.260, +0.044, +0.037883)` env-local은 큐브 xy footprint
  (x [0.25,0.35], y [-0.05,0.05]) 내부, 꼭대기(+0.087883) 50.0mm 아래.
- 고정 조 face proxy만 +y 면 밖 2mm standoff; moving-jaw 쪽 공간은 top 아래
  전부 solid cube. D328 세 후보 어느 것도 moving-jaw 쪽 충돌-프리 corridor를
  주지 못함 (offline FK로 재계산 확인).
- `far_side_slide`는 이름과 달리 위/반대편 경로가 아니라 큐브 중간높이 수평
  직진 경로: `(0.170,0.044,0.0379) -> (0.235,0.044,0.0379) -> target`
  (d328 `:174-175`).
- 실제 충돌 지오메트리는 coarse convex hull(D231 감사: link5 collision AABB
  46.5 x 35.5 x 120.6mm)이라 2mm standoff corridor는 사실상 0.

### 4. D328 보고 수치는 전부 로그와 일치 (감사 PASS)

- cube removed 1.512mm / present 72.178mm / commanded 0.927mm, torque max 1.0
  final 0.8, 후보 3개 IK/clearance 수치, final 0/10 내역, TCP 58.656-59.379mm,
  commit `6d8ff52` 구성 — `g0a_d328_collision_vs_drive_summary.json` +
  `g0a_d328_final_retest_trials.csv` + 세션 문서와 전부 일치.
- "큐브 제거"는 삭제가 아니라 (1.2, 0.55)로 teleport
  (`step1_cube_removed.cube_removed_info.removed_local_xyz_m`).

### 5. 신규 발견 2건 (D328 보고에 없던 계측 결함)

1. **clearance 70.000mm는 fallback 상수**: approach waypoint가 큐브 footprint
   밖이면 `collision_repair_raise_m=0.070`을 그대로 기록 (d328 `:205-206`).
   측정된 값은 `high_corridor_drop`의 20.264mm뿐. far_side_slide 채택 근거는
   공허했고, 수리 실패는 예정된 결과였다.
2. **ContactSensor 0.000N의 원인은 설정**: robot USD spawn이
   `activate_contact_sensors=False` (`roarm_rl/roarm_stack_env.py:150`,
   RoArmCubeTap10cmEnvCfg가 상속). d328 docstring 자체가 "The active env does
   not configure contact sensors" 인정 (`:82`). 미스터리가 아니라 예측 가능한
   계측 결함.

### 6. 부수 확인 (사용자 질문 답변)

- **RL 아님**: G0a 리칭은 100% 스크립트. `_solve_runtime_ik` = finite-difference
  Jacobian + damped least-squares 수치 IK (d323 probe `:169-261`). policy /
  checkpoint 로드 0건, BC teacher 명시 비활성 (d327 `:348-350`). RL은
  G-사다리상 G1b에서 처음 (`direction:19`).
- **프레임**: robot base가 env origin (0,0,0)에 spawn
  (`roarm_rl/roarm_stack_env.py:170`); env-local 좌표 = base 좌표
  ("world == base coord" 주석은 `roarm_stack_env.py:13` — 핸드오프가
  cube_push_env L13이라 한 것은 파일 오기, 실질은 동일). 물체 고정 (0.30, 0.00)
  (d327 `:914-915`, d328 `:577-578`). TCP 목표는 물체 중심 기준 상대 공식
  (2mm standoff)을 base 좌표로 표현한 것.
- **그리퍼 규약**: 이 sim에서 q LOW = OPEN, q HIGH = CLOSED
  (`roarm_stack_env.py:702-704` P6 v7 주석, grasp 조건 `q >= 0.4` `:1195`).
  G0a의 `q=0.0` 고정은 완전 열림 유지가 맞음. direction 문서의 "URDF 기준
  1.571rad"는 joint travel range를 뜻함 (URDF limits `[0, 1.571]`,
  `local_assets/roarm_m3/urdf/roarm_m3.urdf:230`).

## 사용자 결정 (2026-07-10)

- (A) G0a 물체를 D34 x H90 원통으로 바꾸는 최소 패치 설계 — 승인.
- (B) D329 DECISIONS 기록 — 승인.
- (C) 큐브 유지 + D328 collision sweep audit — 기각.

## D330 최소 패치 설계 (설계만; 구현/실행 전 재보고)

신규 파일: `sim_scripts/cyl34_top_view_d330_grasp_g0a_alignment_probe.py`
(전방 전용 — 기존 probe/env 파일 이동·개명·수정 없음. d323/d327 helper import
재사용.)

1. **Env (probe-local override만)**:
   - `env_cfg.sponge.spawn`을 `sim_utils.CylinderCfg(radius=0.017, height=0.090,
     axis="Z", ...)`로 교체. rigid/collision props와 마찰(1.5/1.2)은 기존 큐브
     spawn 값 그대로 이식.
   - mass 0.72kg **유지** — 단일 변수 변경(기하만)을 위함. 정렬-only probe는
     통과 시 물체에 닿지 않으므로 질량 무관. 실물 원통 질량 보정은 G0b 준비
     항목 (BACKLOG).
   - `env_cfg.cube_size_x/y_m = 0.034`, `cube_size_z_m = 0.090` (env 내부 정합;
     구현 시 probe 경로에서 이 필드를 큐브 기하로 소비하는 코드가 없는지 확인).
   - 물체 init pos `(0.30, 0.00, TABLE_Z + 0.045)`.
   - **계측 수리(반응적)**: `env_cfg.robot.spawn.activate_contact_sensors = True`
     probe-local override + D328 런타임 ContactSensor 재사용 → 이번에는 힘이
     실제로 보고됨. force > 0 이벤트를 접촉 증거로 기록.
2. **목표 (D325 family + D327 standoff, D=0.034로 재매개변수화)**:
   - radial offset = D/2 - tip_depth = 0.017 - 0.010 = **0.007m**
   - tangent offset = D/2 - 0.008 + 0.002 = **0.011m**
   - TCP z = 원통 중심 높이 (env-local **+0.032883**)
   - 목표 TCP env-local ≈ **(0.293, +0.011, +0.033)**; link5 +x = tangent -1,
     tool axis 자유 (D325 그대로).
3. **접근**: d327_radial 2-waypoint만 (radial 0.04m 밖 pre-approach -> target),
   step 수 d327 baseline 동일. **waypoint 탐색 금지** (사용자 지시).
4. **게이트 (구조 불변, 치수만 재매개변수화)**: TCP <=5mm, tangent <=15deg,
   gap [0, 5mm] (side point = center - tangent*R 평면 proxy 유지; 원통 곡률로
   proxy gap 2mm vs 실표면 ~3.5mm 차이는 기록만), penetration 0, contact height
   >=15mm below top (top = env-local +0.077883; 중심 높이 정렬 시 ~45mm ->
   여유 통과), displacement <5mm, 10/10.
5. **사전 등록 판정**:
   - PASS (10/10) -> `D330_G0A_CYL_ALIGNMENT_PASS` -> G0a 완료, G0b 진입 논의.
   - FAIL -> 게이트별 카운트 + (이번엔 작동하는) contact force trace + viz/Rerun
     기록. 동일 정체 시그니처(수십 mm TCP stall + force>0)면 wrong-object
     가설의 runtime-stall 부분이 반증됨 -> 올바른 물체 위에서 collision/drive
     감사 재개.
6. **산출물**: `claudedocs/runtime_logs/grasp_track/g0a_d330/` (Visualization
   DoD: target-vs-actual frame PNG + decision-time snapshot + Rerun trace).
7. **구현 전 사용자 확인 필요한 가정 2개**:
   - 원통 질량 0.72kg 유지가 맞는가 (실물 원통 질량 있으면 제공)?
   - 정렬 z = 원통 중심 높이(45mm)가 맞는가 (G0b 파지 높이는 별도)?

## Non-goals (불변)

G0b 파지/들어올림, 그리퍼 닫힘, RL/PPO, 위치 랜덤화, 렌더(단일 프레임 viz
제외), 마찰/재질 변경, VLA, RoArm 실기, B200, 기존 파일 이동/개명.

## Next steps

1. 사용자: D330 설계의 가정 2개 확인.
2. D330 probe 구현 + 10-trial 실행 (failable).
3. 결과에 따라 G0b 진입 논의 또는 원통 위 충돌 감사.
