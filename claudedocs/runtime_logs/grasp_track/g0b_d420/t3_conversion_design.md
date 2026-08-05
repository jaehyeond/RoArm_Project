# g0b_d420 — T3 전환 설계 (p7 → p9, D29×H50 물리 파지)

작성: 2026-08-05 (19th 세션). 상태 = **설계 확정 대기** (p9 저작·실행은 다음 단계,
③ prereg/hash/attestation은 p9 저작 완료 시 발행). 근거 = 워크플로우 `wf_3b176386-4ee`
G5 인벤토리 + lead 1차 소스 재검증.

## 결론 요약

교수 지시(D419)의 "전환 델타 = spawn만"은 **불충분**하다. p7 probe는 cube에서도
GRASP_PASS 전례가 없고(5/20 B200 1회 = LATCH_FAIL), 아래 결함 2건이 그 실패를
구조적으로 설명한다. p9는 p7의 골격(단계·게이트·무결성 장치)을 승계하되 다음
델타를 전부 반영해 **신규 저작**한다.

## 델타 목록 (전부 사전등록 대상)

### D-1. q5 개폐 규약 반전 [치명 — lead 1차 소스 재검증 완료]

- 동결 grasp 트랙 권위: `cyl34_top_view_d337_...open_jaw_target_gate.py:57-59`
  `Q5_OPEN_RAD=1.5413 / Q5_CLOSED_RAD=0.0` + d409 worker `:2116-2117`
  `linspace(OPEN=1.5413 → 0)` 닫힘 스윕 + 첫 접촉 `q5* = 0.5525~0.7225 rad`
  (cooked d348 기하로 증명). **sim에서 q5 큼=열림, 감소=닫힘.**
- p7/env 반대 규약: probe `GRIPPER_OPEN_DEG=0.0`, close 스윕 증가 방향
  [23→45.84°], env 주석 `roarm_stack_env.py:258` "gripper OPEN q=0.0".
- p9: APPROACH/DESCEND 중 q5 = **88.31°(1.5413rad, 동결 OPEN)**, LATCH = 하강 스윕
  (예: 88.31 → 60 → 45 → 41.40 → 39 → 37 → 35 → 33 → 31.65 → 28 → 24, 밴드
  41.40~31.65° 구간 2° 촘촘히). 5/20 LATCH_FAIL("one_sided_push")은 닫힌 조 하강
  가설로 설명 가능 — 단 그 run은 B200 v4 USD였으므로 "설명 후보"로만 기록.
- 실기(T4) 주의: 실물 SDK 매핑은 별도 확인 항목 (d322 매핑 "real 88.3° ↔ URDF
  1.571rad" = 열림-열림 정렬. env 주석의 0=open은 sim 권위와 충돌 — T4 전 재검증).

### D-2. `_grasped` marker 재정의 [치명]

- env `_grasp_condition` = |TCP−중심| < 0.025m AND q5 ≥ 0.4rad
  (`roarm_stack_env.py:1192-1195`, `:379-380`).
- 결함 2중: (a) H50 top 파지의 TCP−중심 거리 = H/2+margin = **0.0255m > 0.025m**
  → 구조적 발화 불가. (b) 각도 조건이 D-1 반전 규약에서 무의미(열림 상태도 ≥0.4rad).
- p9: marker를 probe 측 monkeypatch로 교체(사전등록) —
  `distance < 0.030m AND q5 ≤ 41.40°(첫 접촉 밴드 상단)`. 파지의 실증거는 여전히
  LIFT follow(≥6mm)이며 marker는 LATCH 단계 진입 조건일 뿐.

### D-3. 그리퍼 충돌체 — **동결 attempt3 자산 재사용** [치명 → 해법 확정, 재분해 금지]

- 로컬 USD = UrdfConverter `collider_type: convex_hull` (링크당 1 볼록껍질,
  `local_assets/roarm_m3/usd/config.yaml`) → 오목 조의 목구멍이 막힘 → 물리 파지
  기하 무효.
- **해법 (레시피 추출 에이전트 회수 완료, 2026-08-05)**: 재저작·재분해는 **불필요하고
  금지**(D415 ③ "재분해 시 동결 참조 붕괴"). 동결 파생 자산이 이미 존재:
  `claudedocs/runtime_logs/grasp_track/g0a_d344/collision_asset/attempt3/roarm_m3_fullmesh_fixed_point_parts/roarm_m3.usd`
  — root 레이어는 local_assets와 bit-동일(`a4be58e8…`), physics 레이어만 상이
  (`043a5d35aa425c2589c77a34fcf415401ce9f9e7155e84ed75f6737df79fb503`, gripper_link 64 +
  link5 64 파트, 레거시 단일 hull은 disabled). prim 경로 `/World/envs/env_0/Robot/…`
  = RoArmStackEnvCfg 레이아웃과 동일(num_envs=1).
- p9 통합: ① `os.environ["ROARM_M3_USD_PATH"] = <attempt3 경로>`를 **roarm_rl import
  전에** 설정(모듈 스코프 읽기 `roarm_stack_env.py:97-100`; 기성 패턴 =
  `p7_branch_b_cube2cm_local_grasp_close_sweep_usd_probe.py:47`) ② 실행 전 root/physics
  레이어 sha 핀 assert ③ env 생성 후 첫 step 전에 스테이지 감사 — body별 enabled
  `part_NNN` **64개** + disabled 레거시 `node_STL_BINARY_` **정확히 1개**
  (`d334:197-240` `_usd_collision_inventory` + `d349:921-934` body_checks 복사) ④ 불일치
  시 물리 스텝 없이 hard-fail(d409 `worker:1065-1069` 방식).
- 그리퍼 액추에이터 = ImplicitActuatorCfg stiffness 80 / damping 4 / effort_limit 2.5
  (`roarm_stack_env.py:182-188`) → 올바른 충돌체 하에서 닫힘 스윕은 접촉에서 **stall**
  (클리핑 아님). ContactSensor는 p7식 게이트에 불요 — 추가 시에만 64파트 기준
  `max_contact_data_count` 산정(D361/D362 전례).
- 참고: D406 SDF 파생(res256)은 `D407 … FAIL_STOP`로 격리 — **attempt3 A64 변형을 쓸 것.**

### D-4. 물체 spawn/질량/마찰

- `sim_utils.CylinderCfg(radius=0.0145, height=0.050, axis="Z")` (isaaclab 2.3.0
  `shapes_cfg.py:75-89` 실재 확인), `--object_size_m 0.029 0.029 0.050`,
  `--object_xy 0.300 0.0` (d409 동결 문맥과 비교 가능 위치).
- 질량 0.02483 / 마찰 μs 0.40 μd 0.30 rest 0.0 (본 폴더
  `t3_mass_friction_contract.md` 주 leg; p7의 1.5/1.2는 D362 전이 금지 값이라 교체).
- 로봇·지면 material + combine mode 런타임 기록 의무 (지면 = 1.0/1.0/0.0 multiply,
  `roarm_stack_env.py:114-144` — 기록만, 변경 금지).

### D-5. 좌표 기준 주석 (ground z=0 vs TABLE_Z)

- terrain plane z=0, TABLE_Z=-0.012117은 계획 상수 → spawn 후 settle에서 +12.117mm
  올라앉고 probe의 settled replan이 타깃을 재유도(내부 정합). d409 절대좌표와의
  비교 시 이 시프트 명시 의무. T2b 주소지(부속 IK 확인)가 이 실높이를 커버.

### D-6. USD 경로 가드

- env 기본값 = 반납된 B200 경로(`roarm_stack_env.py:97-100`, HARD RULE #27 위반
  경로) → p9는 `ROARM_M3_USD_PATH=local_assets/roarm_m3/usd/roarm_m3.usd` export
  필수 + `/NHNHOME` 문자열이면 즉시 abort하는 가드 추가.

### D-7. D341 Rerun 산출

- p7은 Rerun 통합 0 → p9는 스텝 타임라인(TCP·물체 pos/quat·q5·게이트 스칼라)을
  전 스텝 로깅 + verdict 스칼라 + 고정 blueprint + `validate_rerun_artifact` PASS +
  검수 PNG + 육안검수. 접촉력 화살표는 생략-정당화(게이트가 접촉력을 소비하지 않음)
  — 세션 doc에 명기. 패턴 = d355 스크립트 + `t2` 프로브와 동일.

### D-8. 승계(변경 금지) 항목

- 단계 체인 APPROACH→DESCEND→LATCH→HOLD→LIFT / verdict 세트 / 물리 게이트
  (drift 6mm·speed 0.08·tilt 12°·upright 0.95·lift_follow 6mm·target 3mm — 단
  push-grasping funnel과 drift 게이트의 긴장은 **사전등록 시점에 명시 결정**:
  attempt1은 게이트 원안 유지, funnel 관측되면 게이트 수정을 별도 leg로).
- 무결성 장치: marker-only attach(kinematic pin 무력화), posewrite watch,
  set_joint_position_target watch, `hidden_kinematic_posewrite_artifact` 보고.
- `episode_length_s`는 촘촘한 스윕 감안 10→20s 상향(사전등록에 명기).

## 실행 순서 (다음 단계)

1. d348 재저작 레시피 회수(진행 중) → p9 저작 → py_compile + 정적 검사.
2. ③ prereg/hash/attestation 발행 (p9 sha + 전체 CLI 인자 + 게이트 + D341 계약).
3. Isaac 실행 (사용자 승인 "T2/T3 진행" 기수령 — 실행 직전 tuple 요약을 브리핑에 첨부).
4. `*_FAIL` 단계 = 다음 수리 지점 (START_HERE T3 행 규약).
