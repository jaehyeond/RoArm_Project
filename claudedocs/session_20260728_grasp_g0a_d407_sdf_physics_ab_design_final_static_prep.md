# D407 설계 확정 + 정적 준비 — sdf_physics_ab_d362_remeasure

Date: 2026-07-28 심야 KST (D407 설계 착수 세션 직후 연속 세션).
이번 case의 신규 변수: **1** — `gripper_link_collision_representation_a64_to_sdf_res256_v1`
(gripper_link 충돌 표현 A64→SDF res256의 **물리 적용**).

**승인 해석 (전 세션 고정, 유저 무정정)**: 유저 "D407 설계 착수 승인" = 설계 +
정적 준비 + attestation/tuple 작성까지. **실제 물리 실행(runtime)은 tuple sha
인용 별도 명시 승인 필요** (D400-P2 3단계, DECISIONS.md:23653-23663).

**Session progress rule 정당화**: 이 세션의 실패 가능 실험 = 정적 runner의
실물 replay fixture들 (D362 trace 500-row 재적용 bit-exact, D406 derivative
USD 감사, Rerun 실물 replay, admission replay — 전부 실패 가능하며 실제로
실패 시 설계/구현 수정을 강제한다).

## 1. 이 세션이 한 일 (감사 가능 step-by-step)

1. 부트: START_HERE / DECISIONS D402-R1~D406 / LEDGER 445행 / D407 kickoff
   세션 문서(§3 설계 v1 + §6 리뷰 headline) read + git status 82 dirty 확인.
2. **4-lens 리뷰 journal 전문 회수** (wf_0375d167-bc7): blocker 2 + warnings
   21 (frozen 렌즈 1+5 / lessons 0+4 / ops 1+5 / science 0+7; dirty-count
   중복 지적 2건 포함 21). 전 렌즈 refutation_failed=true. §6 headline과
   모순 없음 확인.
3. 구현 맵 워크플로우 (wf_1125341f-e32, 4-reader): D362 harness 4,862라인
   전수 stage/게이트/리터럴 맵 + D400 체인 wrapper/freeze 패턴 + D404~D406
   static-prep 템플릿. 별도로 본 세션이 직접 read한 구간:
   d362:1-216(상수)/296-430(입력 pin)/705-941(계약)/942-1132(prepare)/
   1135-1381(env·capacity)/3860-3979(게이트)/4150-4349(worker 종단)/
   4350-4862(run·finalize·main).
4. **양 leg 자산 디렉토리 전수 해시 실측** (§3.4 표) — BLOCKER 1 수리의
   근거. 각 7파일; 6파일 bit-동일; root layer만 상이.
5. 설계 확정본 저작 (§3 — blocker 2 + warnings 21 전부 반영, §2 처분표).
6. (이후 절차는 §5~ 참조 — prereg builder/harness/정적 runner/attestation.)

## 2. 리뷰 발견 23건 전수 처분표

> 표기: [B1][B2] = blocker, [f1~f5]=frozen 렌즈 warnings, [l1~l4]=lessons,
> [o1~o5]=ops, [s1~s7]=science. 전문: wf_0375d167-bc7 journal.

| ID | 요지 | 처분 (설계 §) |
|---|---|---|
| B1 | Leg A gitignored USD 레이어 2/7만 pin | §3.4: 양 leg **각 7파일 전수 sha pin** + 6파일 A↔B bit-동일 게이트 + root-만-상이 게이트 |
| B2 | Leg A 관측성 실패 시 B 발사 여부 미정의 | §3.7: leg A는 operational·observability **어느 쪽 FAIL이든 B 미발사**, attempt 전체 FAIL_STOP (분리 sub-verdict 기록, D404 준용). 리뷰어 대안(관측성 유예) 기각 근거 명기 |
| f1 | "6파일 pin" 실제 7파일 — 카운트 리터럴 결함류 | §3.4: 개수·해시 전부 빌더가 디렉토리 열거에서 프로그램적 도출 (하드코딩 금지) |
| f2 | "physics=base와 bit-동일" 서술 오류 | §3.4: "D406 derivative의 physics.usd(043a5d35)는 **D344 base asset의 physics.usd**와 bit-동일"로 재서술 |
| f3 | 동결 폴더 in-place 소비 비변조 게이트 부재 | §3.4: post-run 양 dir 7+7 전수 rehash 게이트 (completion 전) |
| f4 | D368 인용 오프셋 (21213-21216 오기) | 본 세션 재검증: `collider_count_tipping_causality`=null은 DECISIONS.md:21249(및 21447) — 확정본 인용 정정 |
| f5 | B-leg 게이트 기대치가 오프라인 감사 유래 — D406 커버리지 경계 미명시 | §3.5·§3.10: D406 권한은 property enumeration까지; live ContactSensor binding+SDF inventory는 D407 B-leg이 사상 첫 실행임을 prereg 리스크에 명문화 |
| l1 | D407 harness 자체의 session-doc 리터럴 체크 3종이 stage K 범위 밖 | §3.8 stage K: **최종 실물 세션 doc + 최종 harness로** admission replay (순서 제약 명문화: harness 동결→doc 최종→prereg→K) |
| l2 | 2-seq-Isaac-worker 첫 구조; inter-leg 정책 미명세 | §3.6.5: leg 간 정책 전항 명세 (host admission 재수행, GPU settle, per-leg dir/프로파일) + §3.10 정적 replay 불가 잔여 목록 등재 |
| l3 | B-leg d349 topology 대체 함수 미-fixture | §3.5 게이트 4 + §3.8 stage H2: 대체 함수 core를 stage-독립으로 설계, derivative 실물 USD로 오프라인 fixture |
| l4 | stage F "sub-verdict bit-exact" 충족 불가 (D362_↔D407_ 라벨) | §3.8 stage F: 수치 bit-exact + verdict는 **DXXX 정규화 후 일치** (d362:914-918 동일 메커니즘)로 판정 기준 재정의 |
| o1 | harness 1파일 vs 4-sha tuple 부정합 | §3.6.1: **controller/worker 2파일 확정**; tuple = prereg+attestation+controller+worker (D400-P2 그대로) |
| o2 | Korean font 게이트 유지 여부 미명시 | §3.6.4: **유지** (`korean_font_exists`; sheet가 한국어 렌더). ffmpeg/libx264/opencv 게이트는 MP4 폐지와 함께 삭제 |
| o3 | dirty 81 → 실측 82 stale | §3.9: allowlist는 prereg 빌드 시점 + 실행 직전 2회 **라이브 재열거** (draft 카운트 복사 금지) |
| o4 | 300s 검수 1회에 양 leg 피검체 — 예산 미확인 | §3.6.6: required_fields leg별 사이징 + stage J dry-run에서 traversal 시간 실측·여유 확인 |
| o5 | D324 target-vs-actual frame marker 미명시 | §3.6.3: leg별 RRD에 q5 target/actual 쌍 시계열 + object 초기 스폰 참조 대비 actual pose를 필수 구성요소로 명시 |
| s1 | SDF pair contact-point 보고 가용성 미검증 | §3.6.3·§3.8 stage M2: NVIDIA 버전일치 문서 검증 수행·기록 + prereg에 B-leg에서 이 체크의 지위 분류 (fail-capable 첫 관측; 사전 보증 없음) |
| s2 | 256/pair envelope의 SDF 유효성 = convex 기반 가정 | §3.5 게이트 1: B-leg capacity 공식의 SDF 적용은 "프로젝트 가정"으로 분류, 엔진 한계 표기 금지; overflow audit가 fail-capable 검증자 |
| s3 | instanceable=false가 link5 scope에도 적용 | §3.5: link5측 delta 귀속 caveat prereg 명문화 (등록 변수 정의에 포함) |
| s4 | cook cache 프로세스 간 지속 — provenance 미기록 | §3.5: leg별 cooking/cache 카운터를 prerequisites JSON에 기록 (게이트 아님, 기록 의무) |
| s5 | dirty 82 재계산 필요 (o3와 동일 취지) | o3와 동일 처분 |
| s6 | 6번째 A64 게이트 corrected_d348_128_of_128 누락 | §3.5 게이트 6: B-leg에서 historical-evidence audit로 authority 재라벨 (계산 유지) |
| s7 | leg 간 GPU settle 부재 | §3.6.5: bounded settle (5s 주기, ≤180s, free≥8192MiB + 잔존 프로세스 0) fail-closed |

## 3. D407 확정 설계 (v2 — 이 문서가 prereg의 설계 원본)

### 3.1 무엇을 왜

- 과학 질문: D362 전도(최종 row 500: XY `60.61899778989994mm`, tilt
  `89.99777464743418°`, z `-28.000520542263985mm`; event 31/32→41/42→45/46)가
  gripper_link 충돌 표현을 A64(64 convex)→SDF res256(1 mesh)로 바꾸면
  실제로 달라지는가를 물리로 첫 재측정.
- 가설 증거 기반(인과 주장 아님 — D368 `collider_count_tipping_causality`
  =null 유지, DECISIONS.md:21249): D334 cooked hull 팽창 ~3.5-9.4mm + cook
  parity 1.46% FAIL.
- 개선 임계값(improvement gate) 사전등록 안 함 — 측정 case. `g0a_pass=false`.

### 3.2 실험 구조 — 2-live-leg A/B

- **Leg A (control)**: 동결 D362 물리 계약 verbatim + D344 attempt3 base
  asset (A64 64+64). 신규 변수 0 — 현 스택(driver 580.173.02) 위 D362 재현.
- **Leg B (treatment)**: 동일 계약 + D406 attempt1 derivative (link5 A64 64
  유지·enabled, gripper A64 64 disabled + SDF res256 mesh 1 enabled,
  양 collision scope instanceable=false). 신규 변수 1.
- supervisor(controller) 1회, worker 2회(같은 worker 파일 `--leg a`/`--leg b`,
  leg당 retry 0), 순서 고정 A→B, seed 동일 33201, dt 0.005, 200+300=500 step.
- A vs D362 원본 trace 비교는 descriptive 판독(게이트 없음)으로 격하 —
  primary contrast는 **같은 세션 A vs B**.

### 3.3 동결 계약 상속

- 물리 계약 전 항목 D362 verbatim: 원통 r0.017×h0.090m/0.72kg/마찰
  1.5/1.2/rest 0.0, spawn `[0.30000001192092896, 0, 0.03288299962878227]`,
  Q_FROZEN_OPEN_F32(q5 OPEN `1.5413000583648682`), OPEN 200+close 300,
  actuator 80/4/2.5/3.14, event 임계 0.1N/0.5mm/1.0°/연속2, IK·경로 부재.
- **science 14함수 byte-identity**: worker 파일의
  `_actuator_contract, _object_spawn_contract, _q5_telemetry, _state_row,
  _physics_step_checked, _set_closed_q5_target, _instantaneous_event_masks,
  _confirmed_robot_labels, _motion_confirmed_pair, _annotate_event_masks,
  _qualifying_robot_point_contract, _baseline_statistics, _closure_statistics,
  _q0_q4_drift_summary` 14개를 D362 모듈 대비 inspect.getsource 소스 등가로
  게이트 (`replace("D362","DXXX")`/`replace("D407","DXXX")` 정규화만 —
  d362:889-939의 `_frozen_d360_science_source_contract` 메커니즘 재사용).
- 미기록 물리 설정(gravity/solver/offset)은 같은 env 경로(roarm_rl
  RoArmCubeTap10cmEnv + d351.d332._configure_runtime_env)로 두 leg 동일 적용
  + 런타임 실측 기록 의무 (게이트 아님).
- `_baseline_statistics` 13 체크 / `_closure_statistics` verdict ladder /
  d351.d333 stage·sensor 계약 / d361 durable prefix wire 계약 전부 상속.

### 3.4 자산 + freeze (BLOCKER 1·f1·f2·f3 수리)

이 세션 실측 (파일별 sha256 첫 16 hex; **prereg에는 빌더가 전체 64 hex를
디렉토리 열거에서 프로그램적으로 도출** — 개수 리터럴 포함):

| 파일 (dir 상대) | Leg A (D344 attempt3) | Leg B (D406 derivative) | A↔B |
|---|---|---|---|
| .asset_hash (32B) | ae762fcc536a0d02 | ae762fcc536a0d02 | 동일 |
| config.yaml (663B) | 5745bbb8d9e18716 | 5745bbb8d9e18716 | 동일 |
| configuration/roarm_m3_base.usd (2,769,018B) | ea0ee8f258e93579 | ea0ee8f258e93579 | 동일 |
| configuration/roarm_m3_physics.usd (33,705B) | 043a5d35aa425c25 | 043a5d35aa425c25 | 동일 |
| configuration/roarm_m3_robot.usd (1,709B) | 2227536fcb8c9dae | 2227536fcb8c9dae | 동일 |
| configuration/roarm_m3_sensor.usd (648B) | 3f44081f42b452bc | 3f44081f42b452bc | 동일 |
| roarm_m3.usd (root) | a4be58e87b1f9790 (1,457B) | c02808abbf535c26 (3,177B) | **상이 (유일)** |

- Leg A dir: `claudedocs/runtime_logs/grasp_track/g0a_d344/collision_asset/attempt3/roarm_m3_fullmesh_fixed_point_parts/`
- Leg B dir: `claudedocs/runtime_logs/grasp_track/g0a_d406/attempt1_d405_prereg_status_literal_repair/collision_asset/roarm_m3_link5_a64_gripper_sdf_res256/`
- 게이트 (admission + post-run 각 1회):
  ① 각 dir **정확히 7파일** (초과/누락 없음, 전수 열거)
  ② 14파일 전부 pin sha 일치
  ③ 6개 non-root 파일 A↔B bit-동일
  ④ root layer만 상이 — "A↔B authored 차이 = D400 게이트 검증 opinion 집합뿐"
  주장이 게이트로 검증됨 (f2 재서술: derivative physics.usd는 **D344 base
  asset의 physics.usd**와 bit-동일; base.usd와의 동일 주장 아님)
  ⑤ post-run 재실행 (f3): worker 2회 실행 뒤 completion 전에 ①~④ 전수
  재검증 — in-place read-only 소비의 비변조 증명 (콘텐츠 sha 기준; mtime
  무관). D406 attempt1 "불변 보존" 규칙과의 양립 근거: 읽기 소비만 하며
  쓰기 0을 이 게이트가 증명.
- 추가 pin: D362 원본 evidence(트레이스/prereg/completion 등 §3.8 stage E
  목록), 동결 모듈 파일(d332/d333/d334/d349/d350/d351/d361/d360/roarm_rl 2),
  URDF, d334 sidecar. D362의 `_status_scope_ok` prefix 방식 대신 **exact
  allowlist** (D401 snapshot 방식) 채택.

### 3.5 게이트 재설계 (6곳 — s6 반영; 표는 A / B)

| # | 게이트 | Leg A | Leg B |
|---|---|---|---|
| 1 | capacity (d361 공식 `1×(1+1+64+g)×256`) | 33,280 (g=64) | **17,152** (g=1). s2: 256/pair는 설치 PhysX 5.6.1 convex-pair 검증 유래 — SDF pair 적용은 **프로젝트 가정**으로 분류(엔진 한계 표기 금지), overflow audit(D362 동일 3-token 스캔)가 fail-capable 검증자 |
| 2 | live_binding + part_counts | {link5:64, gripper_link:64} | {link5:64, gripper_link:**1**} — 기대치는 stage H 오프라인 USD 감사에서 도출; **live ContactSensor+SDF inventory는 사상 첫 실행** (f5) |
| 3 | shape inventory (`_runtime_capacity_contract`) | {sensor_cylinder:1, support_table:1, link4:1, link5:64, gripper_link:64} | 동일하되 gripper_link:**1**; backend 버퍼 shape `[[17152,1],[17152,3],[17152,3],[17152,1],[1,4],[1,4]]` |
| 4 | topology (표시+검증) | d351.d349._build_live_topology_parts (동결) | **대체 함수** (l3): link5 half = 64 A64 part 열거·enabled 검증 유지; gripper half = derivative의 SDF 입력 mesh 1개 (points/faceVertexIndices; 41,094 verts/13,698 tris, source stream hash `31aead25f7aa...5a31`) + A64 64 part **disabled** 검증. core는 (stage, prefix) 인자 함수로 분리해 stage H2에서 derivative 실물로 오프라인 fixture |
| 5 | rerun 표시층 | link5 64 + gripper 64 mesh | link5 64 + gripper **SDF mesh 1** |
| 6 | corrected_d348_128_of_128 (d362:3903-3904) | 동결 유지 | 계산 유지하되 **authority 재라벨**: "D347/D348 동결 evidence의 historical audit — B-leg live topology 증명 아님" (s6) |

- s3: instanceable=false는 link5 scope에도 적용됨 (derivative root layer
  저작; D400 검증 완료) — **link5측 B−A delta는 표현 변수에 단독 귀속 불가**
  caveat를 prereg 등록 변수 정의에 포함.
- s4: leg별 PhysX cooking/cache 카운터(가능한 실측 소스: worker 내 physx
  cooking 통계 조회 결과 또는 실패 시 "unavailable" 명시)를 per-leg
  prerequisites JSON에 기록 (게이트 아님). D406 실측: cook cache는
  프로세스 간 지속 (hit 136/miss 0) — B-leg fresh cook 미보장을 리스크 등재.
- 원통 runtime geometry type probe (read-only, 양 leg 동일 코드) — ledger
  427행 `..._RUNTIME_GEOMETRY_TYPE_PENDING` 해소. 기록 + 양 leg 동일성
  체크만 (게이트 아님).

### 3.6 실행 구조·관측성 (D341/D324)

**3.6.1 harness 형태 (o1 확정)**: 2파일 —
`sim_scripts/cyl34_top_view_d407_sdf_physics_ab_d362_remeasure_controller.py`
(supervisor; Isaac import 없음) +
`sim_scripts/cyl34_top_view_d407_sdf_physics_ab_d362_remeasure_worker.py`
(단일 leg Isaac worker; D362 파일의 leg-파라미터화 derivative). 4-sha tuple
= {prereg, attestation, controller, worker} (D400-P2 정합). controller가
worker를 leg당 1회 Popen (`--stage _worker --leg a|b`), D362 supervisor
프로토콜(로그 xb 배타 생성, start_new_session, 1s 폴링, watchdog
inactivity 300s/total 900s per leg, SIGTERM→SIGKILL) 상속.

**3.6.2 headless**: worker main()이 `args.headless = True` 강제 (D362는
False). AppLauncher headless, viewport 캡처 10종·MP4·storyboard 전부 폐지
(GUI 전용 경로 — D365 stale viewport 결함류 원천 제거). D362 대비 관측성
계약 축소를 prereg에 명시; A vs D362 descriptive 비교의 환경 차로 기록.
DISPLAY 요구 삭제 (headless); rerun CLI 스크린샷은 headless 모드.

**3.6.3 leg별 관측성 산출물** (leg 하위 dir `leg_a_a64/`, `leg_b_sdf_res256/`):
durable prefix JSONL + audit (d361 rebind — 프로파일
`D407_LEG{A|B}_ACTUAL_PHYSX_DURABLE_PREFIX_V1`, lineage 키 D362 스키마 상속),
physics trace JSON+CSV (500 rows), RRD (전 row timeline `physics_step`+
`sim_time_s`, 21 decision scalars, contact points/force arrows, **q5
target/actual 쌍 시계열 + object actual pose vs 초기 스폰 참조 = D324
target-vs-actual (o5)**), blueprint RBL (D362 blueprint 구조: Spatial3D +
TextLog + TimeSeries 3종 — TextDocumentView 아님이므로 D405 R3 1문서-1뷰
제약 비해당임을 확인·기록), headless 스크린샷 (logical `960x540` → 물리
1920×1080 exact 게이트 — D405 R2/D406 라이브 검증 경로 재사용; D362의
4800x2800 GUI 값 폐지), rerun_validation JSON (0.34.1 pin, footer verify).
- s1: `all_qualifying_robot_force_points_finite_for_observability`는 B-leg에서
  SDF pair 최초 관측 — stage M2의 NVIDIA 버전일치 문서 검증 결과를 prereg에
  기록하고, 이 체크를 "fail-capable 첫 관측 (사전 보증 없음)"으로 분류.
  과학층은 D362 계약대로 point 유무와 무관하게 force witness를 보존
  (d362:2493-2495 주석 계약).

**3.6.4 beginner sheet**: 캡처 폐지에 따라 **canonical trace 기반 PIL 오프라인
생성**으로 재설계 (v1의 "matplotlib" 표기 정정 — D326 env 불변 원칙상 신규
의존성 없이 D362 sheet의 PIL 스택 재사용). leg별 1장 + A/B 대조 1장 (한국어;
`korean_font_exists` 게이트 **유지** (o2), ffmpeg/libx264/opencv 게이트
삭제). 시계열 패널: q5, 3 force, XY/tilt/z + event 마커.

**3.6.5 inter-leg 정책 (l2·s7)**: ① leg A worker 종료 → controller가 leg A
전체 판정 (operational + observability + prefix recovery + overflow audit +
inventory) **완료 후에만** leg B 진행; ② GPU settle: 5s 주기 재확인, 최대
180s, free VRAM ≥ 8192MiB + leg A 프로세스 그룹 소멸 + RAM ≥ 10GiB 재확인
— 미충족 시 fail-closed FAIL_STOP (B 미발사); ③ leg B도 D362 worker
preflight 전항 재수행 (프로세스 분리로 모듈 오염 없음); ④ leg별 out dir·
로그·prefix 완전 분리, 산출물 교차 참조 없음; ⑤ 2-seq-Isaac-worker는 repo
첫 구조 — 정적 replay 불가 잔여 목록 등재 (§3.10).

**3.6.6 라이브 육안검수 1회 (B2·o4)**: 양 leg 완주 + 산출물 생성 후
controller가 프롬프트 마커 출력, 300s/0.25s 폴링/first-read-wins, D406
원자적 작성기 계약 (OUT_DIR 내 임시명→같은 파일시스템 rename, 주제별 명시
assert, argv-청정 감시, 백그라운드 exit 대기 금지). required_fields leg별
사이징: leg별 {jaw_or_gripper_visible, cylinder_visible,
timeseries_legible, no_text_overlap} × 2 + 대조 sheet 1 체크 — stage J
dry-run에서 전체 traversal 시간 실측, 300s 여유 확인. 검수 실패/타임아웃 =
관측성 실패 (attempt FAIL_STOP; 물리 sub-verdict는 completion에 보존).

### 3.7 판정 설계 + FAIL 분기 (B2 명문화)

- Primary: leg별 최종 row 500 `disp_xy_mm / tilt_delta_deg / z_delta_mm` +
  event onset/confirmation steps (3 label) + peak forces 3종 + **B−A delta**
  (정의식 사전 고정: 각 스칼라 delta = B값 − A값; event step은 양쪽 발생
  시에만 delta, 아니면 null + 발생 플래그 쌍).
- descriptive: `toppled := 최종 tilt_delta_deg > 45.0` (게이트 아님).
- verdict 체계:
  - leg별 physical sub-verdict: D362 5종 ladder 상속 (D407_ prefix).
  - 전체 canonical verdict:
    `D407_SDF_PHYSICS_AB_TIPPING_REMEASURE_MEASURED` (양 leg 완주 + 관측성
    + 검수 + 무결 전부 PASS) 또는
    `D407_SDF_PHYSICS_AB_TIPPING_REMEASURE_FAIL_STOP` + descriptive 분류
    {leg_a_operational, leg_a_observability, inter_leg_settle,
    leg_b_operational, leg_b_observability, manual_inspection,
    postrun_integrity} 중 최초 실패 지점.
- **B2 분기 규칙 (prereg 동결 대상)**: leg A는 operational FAIL·observability
  FAIL 어느 쪽이든 → leg B 미발사, attempt 전체 FAIL_STOP. 리뷰어 대안
  (관측성 유예/비차단) 기각 근거: ① D362 자체가 관측성을 automated verdict에
  포함(`worker_observability_artifact_pass`, d362:4562-4565) — 유예는 상속
  계약 변경, ② A 관측성 결함 상태에서 B를 실행하면 같은 결함이 B에서도
  재현되어 대조 자체가 오염된 채 attempt 소진, ③ 첫-라이브 위험은 stage J
  실물 replay + D362 관측성층의 라이브 전례(D362 완주)로 완화. 물리
  측정값(trace/prefix)은 FAIL_STOP 시에도 보존·보고 (D362 finalize의
  "physical 보존 + OBSERVABILITY_FAIL_STOP" ladder 상속).
- must_remain_null: force closure / stable grasp / cap-rim-barrel / grasp
  feasibility / 29×50 이전 가능성 / "SDF 일반 우월" 일반화 / per-prim cooked
  SDF 내부 identity / `collider_count_tipping_causality`. `g0a_pass=false`.

### 3.8 정적 준비 stage A~M (재설계)

A. env/pin replica: isaaclab python/numpy 1.26.0/psutil 5.9.8/rerun SDK+CLI
   0.34.1/Noto CJK 폰트/호스트 경계 게이트 shape. (ffmpeg/opencv 없음)
B. prereg shape + **소비자 리터럴 전수 grep**: d351/d332/d361/d349/roarm_rl
   가 D407 입력에서 읽는 전부 + **D407 harness 자체가 저작 아티팩트·세션
   doc에서 읽는 전부** (l1: `registered_base_git_exact`(count≥1),
   `session_harness_sha_pin_exact_once`(controller·worker 각 정확 1회),
   실행·판정 순서 heading, prereg status 리터럴, EXPECTED_PREREG_SHA256
   임베드 일치) + dirty⊆allowlist 라이브 재열거 (o3).
C. `python -B` 거부 fixture + __pycache__ 신규 0.
D. science 14함수 byte-identity (D362 모듈 대비 DXXX 정규화, 프로그램적).
E. frozen input 해시 전수 재계산 (7+7 자산 + D362 evidence + 동결 모듈 +
   URDF + sidecar) — 빌더 산출과 독립 재계산 대조.
F. **실물 trace replay** (l4 기준 재정의): 동결 D362 trace 500 rows →
   D407 worker의 상속 science 파이프라인 오프라인 재적용 → 수치 bit-exact
   (endpoint `60.61899778989994`mm 포함) + event 31/32·41/42·45/46 +
   verdict 문자열 DXXX 정규화 후 일치.
G. capacity fixture: A 33,280/B 17,152 공식 accept + 교란 reject.
H. D406 derivative + D344 base **오프라인 USD 감사** (omni.usd.libs pxr +
   PhysxSchema plugInfo — D403 lesson·D404 stage H 패턴): B-leg 게이트
   기대치 도출 (gripper enabled 0/disabled 64/SDF 1, link5 64,
   instanceable 관찰, SDF 7속성 bits), A-leg base 동일 감사.
H2. B-leg topology 대체 함수 core fixture (l3): derivative 실물 stage에
   source prefix로 실행 → part счет/mesh identity/enabled·disabled 검증.
I. d361 prefix validator rebind fixture (leg별 프로파일 2종 accept +
   위조 reject).
J. **Rerun 실물 replay** (o4 포함): D362 trace로 D407 관측성층 전체를
   scratchpad에서 실행 — RRD 빌드→검증→headless 스크린샷 1920×1080 exact→
   sheet 생성→검수 작성기 dry-run→**traversal 시간 실측 + 300s 여유 판정**.
K. **순수 admission 전수 replay** (l1 순서 제약: 최종 harness·최종 doc·
   실물 prereg로): D407 controller/worker의 prereg admission (sha+status)
   accept + 변조 reject, session-doc 리터럴 3종 accept + 변조 reject.
L. launch 구성 복제 subprocess fixture (D404 lesson): 실제 interpreter/
   argv/cwd로 controller import-해석 + worker `--leg a|b` 프리플라이트
   진입 직전까지 (Isaac import 전 단계) 검증.
M. negative ≥30 + 승인 게이트 오프라인 복제 (실물 4파일; accept 1 +
   reject ≥9, D406 복제 패턴) + M2: **s1 NVIDIA 문서 검증** (버전일치
   ContactSensor/RigidContactView/SDF contact 보고 문서 + 설치 소스 대조,
   결과를 prereg `installed_nvidia_primary_sources`/official_sources에
   기록) + 4-lens 적대 리뷰.

순서 제약 (l1): harness 저작·EXPECTED_PREREG_SHA256 주입 완료 → 세션 doc
최종화 (harness sha 각 정확 1회 포함) → prereg 빌드 → stage K/M 실행 →
attestation → tuple → (필요시 doc 결과 추가 기재는 sha 문자열 재출현 금지
제약 하에) 종료.

### 3.9 승인 경계·실행 계획

- 3단계 (D400-P2): ① 설계 리뷰 (완료 — 전 세션 + 이 문서 §2 반영) →
  ② 구현+정적 attestation+4-sha tuple (Isaac 0회; 이 세션) → ③ tuple sha
  인용 runtime 새 명시 승인 (supervisor 1, worker 2 = leg당 1, retry 0,
  watchdog per leg 300/900s, 호스트 경계 D402-R1, 검수 300s).
- 산출 루트: `claudedocs/runtime_logs/grasp_track/g0a_d407/attempt1_sdf_physics_ab_d362_remeasure/`
  {d407_preregistration.json, d407_static_fixture_results.json,
  d407_reviewed_script_attestation.json, d407_proposed_runtime_hash_tuple.json
  + 런타임: d407_runtime_freeze_manifest.json, d407_supervisor_summary.json,
  d407_ab_delta_summary.json, d407_manual_visual_inspection.json,
  d407_completion_summary.json, d407_ab_comparison_sheet_ko.png,
  leg_a_a64/, leg_b_sdf_res256/}.
- 신규 repo 파일 allowlist: harness 2 + 위 정적 4 + 이 세션 doc 1 (+상태
  문서 3종 기존 dirty). 도구·리뷰·replay 산출물은 scratchpad에만.
- allowed_dirty_paths = 실측 dirty (현 82) ∪ planned 7 — 빌더가 라이브
  재열거 (o3/s5), 실행 직전 재대조.
- **runtime 전 commit/push 금지** (HEAD 변경 = tuple 무효). BASE_GIT =
  `a69a96d36219268e4bc5e25065cc234da9d99674`.

### 3.10 리스크·정적 replay 불가 잔여 (정직 보고)

① SDF per-pair 256 초과 가능 — overflow audit fail-capable (s2)
② B-leg live ContactSensor binding + SDF inventory + property→sensor 경로
   사상 첫 실행 (f5; D406 권한은 property enumeration까지)
③ SDF pair contact-point 보고 가용성 — M2 문서 검증에도 최종 확인은 라이브
   (s1)
④ 2-seq-Isaac-worker supervisor 시퀀싱 첫 구조 (l2) — stage L이 스폰
   구성만 복제, Isaac 2연속 기동·teardown 상호작용은 정적 replay 불가
⑤ cook cache 상태 의존 (s4) — B fresh cook 미보장, provenance 기록으로
   해소 (판정 비의존)
⑥ gravity/solver 미기록 상속 → 런타임 실측 기록
⑦ GUI→headless 환경차 — A/B 내부 무오염, A vs D362 descriptive에만 영향
⑧ 500-step 지평 내 중간 상태 종결 가능 — 유효 측정으로 수용
⑨ 라이브 육안검수 타임아웃 (o4 — stage J 실측으로 완화, 잔여 위험 수용)
⑩ leg A FAIL 시 SDF 변수 미측정으로 attempt 소진 (B2 정책의 수용 비용 —
   완화: A는 신규 변수 0 + stage F/J replay)

## 4. Stage M2 — NVIDIA 버전일치 검증 결과 (이 세션 완료; s1·s2 해소 증거)

read-only 검증 agent가 설치 소스 우선 → 버전일치 공식 문서 순으로 수행
(AGENTS.md NVIDIA rule 준수). 전문 JSON은 다음 세션에서 prereg
`official_sources`/`installed_nvidia_primary_sources`에 전사할 것.

**Q1 — SDF pair의 per-point contact 보고 가용성**: 설치 Isaac Lab 2.3.0
ContactSensor(track_contact_points)→RigidContactView.get_contact_data 체인은
geometry-type 분기가 전혀 없음 (contact_sensor.py:284-288,314-325,381-407;
omni.physics.tensors api.py:5877-5959). omni_physics 107.3 Collision
Behavior Guide가 SDF contact 생성을 명시 문서화 ("at most one contact is
generated per triangle", "SDF colliders tend to generate many contacts" +
scene-level 버퍼 경고). 설치 5.6.1은 CPU SDF도 지원 (CHANGELOG v5.5.0 —
"GPU-only SDF" 구주장은 이 버전에 부적용). 단 "convex와 per-point 보고
parity"의 명시 문장은 없음 → **분류: documented(generic, SDF 예외 없음) +
parity 자체는 undocumented inference. 신뢰도 MEDIUM-HIGH.**

**Q2 — 256/pair envelope의 SDF 적용**: **미문서화 (프로젝트 가정 확정)**.
버전일치 태그(107.3-omni-and-physx-5.6.1) 소스 실측: PxContactBuffer
MAX_CONTACTS=256은 CPU narrowphase per-pair 버퍼이며 CPU SDF mesh-mesh
경로는 이를 경유(≤256/pair 유효); 그러나 GPU mesh-mesh SDF 커널은 자체 상수
MESH_MESH_CONTACT_LIMIT=6 × MAX_MESH_MESH_PATCHES=128 (이론 768/pair >256,
convexNpCommon.h:35-37)이고 문서화된 SDF 용량 knob은 scene-level
GpuMaxRigidContactCount/PatchCount 뿐. **결정적 뉘앙스**: GPU pair 라우팅
소스(PxgNphaseImplementationContext.cpp ~1091,1118-1131)상 convex-vs-
trianglemesh는 SDF 유무와 무관하게 일반 convex-trimesh 경로(eConvexTrimesh,
CVX_TRI_MAX_CONTACTS=5/tri)로 dispatch — SDF mesh-mesh 커널은 trimesh-vs-
trimesh(동적 SDF ≥1)에만 적용. 즉 **D407 leg B의 실제 pair(원통/support
convex vs SDF trimesh)는 SDF 전용 커널을 타지 않을 가능성이 소스로 확인됨**.
→ prereg 분류(agent 권고 채택): leg-B per-point 관측성 = "expected
available — 문서화된 geometry-agnostic API + SDF 예외 부재"; 256/pair
capacity 항 = "CPU-narrowphase 상수 유래 프로젝트 가정 — SDF에 미문서·GPU
경로의 보편 bound 아님"; 런타임 검증자 = 기존 'Incomplete contact data ...
maxContactDataCount' overflow audit (fail-capable) 유지.

## 5. 세션 종료 상태 (context 비상 종료 — 95% 프로토콜 이행)

**완료**: ① 4-lens 리뷰 전문 회수 (blocker 2+warnings 21 확정) ② 설계
확정본 v2 저작 (§2 처분표 + §3) ③ D362 4,862라인 전수 구현 맵 (4-reader
wf_1125341f-e32 + 직접 read; 스펙 근거) ④ 양 leg 자산 7+7 실측 해시 (§3.4)
⑤ stage F 기대값 원본 확정 (trace 마지막 row 필드명 §아래) ⑥ M2 검증 (§4).

**진행 중 (백그라운드, 이 세션 종료 후에도 파일 생성 가능)**: harness 저작
워크플로우 `wf_0d5ade26-6bc` (worker → controller 순차) — 산출 예정:
`sim_scripts/cyl34_top_view_d407_sdf_physics_ab_d362_remeasure_worker.py` +
`..._controller.py`. **다음 세션은 이 파일들의 존재/품질을 절대 가정하지
말고**: 존재 시 adversarial 리뷰(특히 science 14함수 DXXX byte-identity를
직접 재검증) 후 채택/재작성 결정; 부재/불량 시 스펙으로 재저작.

**[세션 마감 직후 갱신] harness 저작 완료 보고 (wf_0d5ade26-6bc 2/2 done — 전부 agent 자기보고, 미검증)**:
worker 3,985라인 / controller 1,284라인, 양쪽 ast PASS. worker
pre-injection sha `1a0c8313fbaa...bdbf9c` (EXPECTED_PREREG_SHA256 주입 시
변경됨). agent 보고 핵심 편차 — **다음 세션 리뷰 필수 항목**:
① d361.REGISTERED_TOTAL_CAPACITY를 leg별 rebind (동결 d361 validator가
33280 하드체크 → leg B 17152에서 필수; D362 확립 setattr 패턴) ② invocation
marker에서 run_nonce 비교 제거 → {leg, invocation_index 1, automatic_retry
false, preregistration_sha256} (정적 prereg에 runtime nonce 없음; controller
가 leg dir에 d407_isaac_invocation_marker.json 작성 계약) ③ D362 summary
리터럴 `asset_decomposition_..._changed: False`를 정직 분리
(`gripper_collision_representation_changed`: leg B true) ④ 루프 후 명시적
final pause 추가 (capture 제거로 상속 restore 체크가 항상 실패하는 문제)
⑤ phase명 `live_64_plus_64_binding`→`live_binding` ⑥ worker가 소비하는
prereg 키 열거 보고됨 (builder 계약) ⑦ controller: phase-marker JSONL만
append-log (그 외 전부 exclusive-create), w.<attr> 59건 존재 audit 자기보고.
전문: workflow journal
`~/.claude/.../4d3cddb8-.../subagents/workflows/wf_0d5ade26-6bc/journal.jsonl`
+ 결과 파일 `/tmp/claude-1000/.../9273a397-.../tasks/wa6qrkcko.output`.

**스펙/맵 위치 (scratchpad — 세션 간 휘발 가능, 경로는 4d3cddb8 세션 id)**:
`/tmp/claude-1000/-home-cgxr-Documents-Robotics-RoArm-Project/4d3cddb8-6495-4150-aa6b-41bc28b3d6f3/scratchpad/`
{d407_worker_spec.md, d407_controller_spec.md, d407_builder_runner_spec.md,
map_remaining.txt} + tool-results/{bl98qv50m.txt, bpkld0tkl.txt}. 휘발 시
재구성 근거: 이 문서 §3 + wf_1125341f-e32 journal
(`~/.claude/projects/-home-cgxr-Documents-Robotics-RoArm-Project/4d3cddb8-6495-4150-aa6b-41bc28b3d6f3/subagents/workflows/`).

**stage F 기대값 (이 세션 원본 실측 — runner가 동결 파일에서 재도출 필수)**:
trace 마지막 row `object_disp_xy_mm=60.61899778989994` /
`object_tilt_delta_from_reference_deg=89.99777464743418` /
`object_z_delta_mm=-28.000520542263985`; closure `first_jaw=31, motion=41,
link5/other=45`, peak {gripper 43.85833992858175, link5 23.227865254723564,
link4 0.0}, verdict `D362_MOVING_JAW_CONTACT_AND_OBJECT_MOTION_OBSERVED`.
주의: closure에 `final_object_*` 키는 **없음**. d349:877 B-leg 비호환 실측
확인 (per-row `runtime_collision_enabled` 요구).

## 6. 다음 세션 필수 절차

1. harness 산출물 검증 (위 §5 — 가정 금지, adversarial 리뷰).
2. prereg builder + 정적 runner 저작·실행 (스펙: §5 경로; 소실 시 §3.8로
   재작성; M2 결과 §4를 prereg에 전사).
3. 순서 준수 (l1): harness 동결 → EXPECTED_PREREG_SHA256 주입 → 세션 doc
   최종화(각 harness sha 정확 1회) → prereg → stage K/M-late → attestation
   → tuple → **정지** (runtime = tuple sha 인용 새 명시 승인).
4. 이 세션의 stop hook /half-clone 요구는 HARD RULE #11에 따라 거부하고 95%
   프로토콜로 종료했음 (전례 유지).

## 7. 경고 (불변)

- D400~D406 전 attempt + D362 33파일 동결. 물리/q5/contact/cylinder 변경 =
  별도 승인. Isaac/GPU는 호스트 경계 (D402-R1). isaaclab env 불변 (D326).
- allowlist 밖 repo 새 파일 금지. commit/push는 유저 요청 시에만.
- HANDOFF.md/half-clone 금지 (HARD RULE #7/#11).

## 8. 다음 세션 continuation prompt (verbatim 붙여넣기용)

```
Read AGENTS.md first, then follow the Current-State Protocol exactly
(START_HERE.md → DECISIONS.md tail (D406) → EXPERIMENT_LEDGER.md tail (446행)
→ claudedocs/session_20260728_grasp_g0a_d407_sdf_physics_ab_design_final_static_prep.md
(§2 처분표 + §3 확정 설계 + §4 M2 + §5 세션 상태·harness 편차 7건 + §6 절차
필독) → git status --short --untracked-files=all).

Active state: D407 설계 확정 완료 (리뷰 23건 전수 반영). harness 2파일 생성
완료 — sim_scripts/cyl34_top_view_d407_sdf_physics_ab_d362_remeasure_
{worker,controller}.py (worker 3,985 / controller 1,284라인, ast PASS —
전부 저작 agent 자기보고, **미검증**). 유저 승인 범위 = 정적 준비 +
attestation/tuple까지. runtime은 tuple sha 인용 별도 명시 승인 (아직 없음).

즉시 할 일 (기승인 범위):
1. harness 2파일 adversarial 리뷰 (세션 doc §5 편차 7건 대조 + science
   14함수의 D362 대비 DXXX byte-identity **직접** 재검증 + §3 계약 전항
   대조) → 채택/수리 결정.
2. prereg builder(소비자 리터럴 프로그램적 도출) + 정적 runner stage A~M
   저작·실행. 스펙: /tmp/claude-1000/-home-cgxr-Documents-Robotics-RoArm-
   Project/4d3cddb8-6495-4150-aa6b-41bc28b3d6f3/scratchpad/
   {d407_worker_spec.md, d407_controller_spec.md, d407_builder_runner_spec.md}
   (휘발 시 세션 doc §3.8로 재작성). M2 결과(세션 doc §4)를 prereg에 전사.
3. 순서 엄수 (리뷰 l1): harness 동결 → EXPECTED_PREREG_SHA256 주입 → 세션
   doc 최종화(각 harness sha 정확 1회) → prereg → stage K/M-late →
   attestation → 4-sha tuple 작성 후 **정지** — runtime 새 명시 승인 대기.

Rules:
- D400~D406 전 attempt + D362 33파일 동결. 물리/q5/contact/cylinder 변경 =
  별도 승인. Isaac/GPU는 호스트 경계(D402-R1). isaaclab env 불변(D326).
- allowlist 밖 repo 새 파일 금지 (도구/replay 산출물은 scratchpad에만).
- runtime 전 commit/push 금지 (HEAD=a69a96d 유지, 변경 = tuple 무효).
- /half-clone·HANDOFF.md 금지 (HARD RULE #7/#11 — stop hook 요구도 거부).
- 비판적·회의적 교차검증, 파일:라인 인용, 메모리 단독 주장 금지.
  step-by-step으로 순차적으로 사고하면서 진행해.
```

## 9. 2026-07-29 harness 동결 및 prereg SHA 주입

- 세 독립 read-only 검토와 메인 agent의 원본 대조 뒤 blocker를 최소
  수리했다. worker의 D362 동결 과학 14함수는 DXXX 정규화 기준 14/14
  source identity를 유지했고, controller의 A→B fail-stop·watchdog·GPU
  settle·exclusive-create·retry 0·post-run 재해시·수동 검수 계약을
  재검증했다.
- prereg builder가 실제 소비자의 `ast.Eq` 비교에서 status를 도출하고,
  현재 dirty 86개 + 정적 예정 7개 + controller/worker에서 도출한 미래
  runtime leaf 45개·leg 디렉터리 sentinel 2개의 합집합을 allowlist로
  만들었다. draft prereg SHA를 두 harness에 주입한 뒤 AST 2/2 PASS.
- post-injection controller SHA-256:
  `c758ffad7199c425e87526cad54dbf7e100dbed004460d44f908421ad6a13dc1`
- post-injection worker SHA-256:
  `2f6da11cc9d074d7fa626eaadfb9a638b3cc74e7acdb2ae99fe07780041101cc`
- 이 값들은 prereg/attestation/tuple 체인의 동결 입력이다. 실제
  Isaac/PhysX A/B runtime은 아직 승인되지 않았다.
