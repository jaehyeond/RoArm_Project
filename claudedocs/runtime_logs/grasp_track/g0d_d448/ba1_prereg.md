# `g0d_d448` / `ba1` preregistration — B601 full-arm side 파지 1클립: 팔 전체(고정 base)가 IK 접근→닫힘→들어올리기로 D29×H50 원통을 잡는가 + 전 과정 RTX mp4

- Date: 2026-08-13 KST (61st session)
- User authority: 61st 채팅 사용자 명시 승인 — 요청 원문 "이거 지금 arm 전체로 해서
  base에 붙여놓고 sim화면으로 해서 물체 잡는걸 mp4영상으로 해서 sim rendering 못해?"
  + AskUserQuestion 답 **"승인 — side 파지 먼저 (권장)"** (신규 case `g0d_d448` 개시).
  이 승인은 **D324 궤적 영상 금지의 명시 해제**를 포함한다 (mp4 1건 + 키프레임 한정).
- 이번 case의 신규 변수 (2개): `[팔 전체 포함 + base 세계 고정 (flying-gripper 제거)]`,
  `[팔 궤적 제어 = 오프라인 수치 IK waypoint + min-jerk 조인트 보간 + PD 드라이브]`.
  mp4 렌더/인코드는 시각화 층위(변수 아님). 그 외 전부 bg1에서 상속(동일 원통, 동일
  분해 충돌 표현 B 계열, 동일 양측 접촉 게이트 상수).
- Scope: 물리 실행 O (실패 가능 — 도달성/추종/파지 각 층위), Isaac Sim 5.1 로컬 4090
  headless (**RTX 렌더 O** — 본 case의 목적), 로봇 하드웨어 0, RunPod 0, lerobot-train 0,
  `g0a_*`/`g0b_*`/`g0c_*` 기존 산출물 편집 0, D427~D447 재판정 0.

## 1. Decision question / branch semantics / non-claims

- 질문: bg1에서 그리퍼 단독(이상 배치)으로 13/13 성립한 B601 파지가, **팔 전체를
  base 고정으로 달고 PD 드라이브 궤적 실행**으로도 성립하는가 — 즉 (i) IK 자세에
  팔이 실제로 도달·정착하고, (ii) 닫힘에서 양측 접촉이 형성되며, (iii) 팔이 들어올릴 때
  물체가 손을 따라오는가. 부가 산출 = 전 과정 RTX mp4 (사용자 요청 시각 증거).
- Branch semantics (1 pose 단판):
  - `BA1_SUCCESS` — G1∧G2∧G3 전부 PASS: 팔 층위까지 sim 필요조건 충족. 영상 = 품의/보고 자료.
  - `BA1_NO_BILATERAL` — 접근·정착은 됐으나 닫힘 양측 접촉 실패 (자세/저작 분석으로).
  - `BA1_SLIP_DURING_LIFT` — G1 성립 후 lift에서 물체 이탈 (동적 유지 실패).
  - `BA1_TCP_TRACK_FAIL` — SETTLE2 종료 TCP 오차 >10 mm: 파지 판정 이전에 제어/드라이브
    층위 실패로 분류 (이후 단계는 기록만, 파지 주장 금지).
  - `IK_UNREACHABLE`/게이트 abort — Isaac 진입 전 종료 (산출물 없음, ba2로 forward-only).
- Non-claims: 실물 B601 파지/제어(모터 펌웨어·마찰·백래시), 벤더 드라이브 게인의 현실성
  (stiffness는 §3-3 하네스 저작), top-down full-arm 도달성(본 case 밖 — side 1 pose만),
  D446 재판정, 다른 방위각/배치 일반화. **mp4는 시각 증거 층위 — 판정 권위는
  `ba1_results.json`/`ba1_trace.npz`** (렌더 그림 단독 제시 금지, hang/lift 수치 캡션 의무).

## 2. Method authority

- 물리 스테핑: bg1과 동일 계열 — default scene, `grasping_utils.simulate_physics_async`
  (설치 `isaacsim.replicator.grasping` 1.0.9) per-step 직접 스테핑, dt 1/60,
  render=False. GraspingManager 미사용 (pose 텔레포트가 없으므로 불필요 — 팔이 이동).
- 조인트 제어: USD DriveAPI targetPosition을 매 step 저작 (min-jerk 보간
  s(τ)=10τ³−15τ⁴+6τ⁵). 초기 상태는 JointStateAPI position 직접 저작 (스폰 자세 =
  standoff IK 해, 관통 없는 자유공간).
- 캡처: 물리 chunk 후 `rep.orchestrator.step()` (D447 ③) — 물리 정지 상태의 스테이지를
  렌더 (p19 실증 경로). 2 step마다 1 프레임 = 30 fps 실시간 배속.
- 라이프사이클: p18/p19 패턴 — BaseException 캡처 + `ba1_failure.json` + rc stdout
  sentinel + fsync → `app.close()` 최종 (D442/D447 ①, exit code로 성공 판정 금지).
- **실행 전 하네스 스모크 (61st, scratchpad `ba1_smoke_probe.py` — 후보 waypoint
  미사용, 중립 q=[10,−90,−45,10,20,30]만, 물체/pedestal은 팔 밖 (0.6,−0.5))**:
  최종 실행 `SMOKE_ALL_PASS=True` — S1 중첩 계층 수리 후 시뮬 성립(아래 ADD-2) /
  S2 정적 유지 max 0.023° / S3 FK-vs-USD 0.31 mm·R 0.00066 (IK 전제 실증) /
  S2b 스텝 추종 15.000° 정확 / S4 손가락 전폐 1e-5 m·재개방 0.0715 m /
  S5 지지 접촉력 median 0.24358264 vs m·g 0.24358230 N (4자리, scene-int 스테핑에서
  contact report 성립) / S6 `rep.orchestrator.step()` 캡처가 물리 0 step·drift 0.
  스모크 시행착오 2건 자진 기록: (a) `omni.replicator.core`와
  `isaacsim.replicator.grasping` 연속 enable(사이 `app.update()` 펌프 없음)은
  **데드락**(16분 futex 대기, 로그 무진행 — kill 후 p19 순서 + 펌프 3회로 해소);
  (b) 드라이브 유지 오차는 τ/k가 아니라 **≈ C·(damping/stiffness)** 스케일
  (스윕: 1e4/500→0.53°, 1e6/500→0.001°, 1e4/0→0.004°, 1e5/2000→0.19°) —
  k·d 동시 비례 변경은 무효과. ADD-1 게인은 이 스윕으로 선정.

## 3. Frozen inputs / pins

### 3-1. 자산 (전부 기존 핀 재사용, 본 case 신규 파일 저작 없음 — 스테이지 오버라이드만)

- 소스: `../g0c_d446/b601_asset/` 9파일 — SHA는 `b601_asset/UPSTREAM.md` verbatim
  (실행 시작/종료 재검증, drift = fatal). **full 로봇 `reBot_B601_DM.usda`를 스테이지에
  참조** (`/World/ba1_robot`, 원점·항등 자세 = base 세계 고정; 공식 root_joint Fixed +
  ArticulationRootAPI(base_link) verbatim 유지 — 텔레포트 없음이므로 D446 ④ⓑ 재앵커
  이슈 비해당).
- 충돌 수리 (D446 ③ 의무): 공식 핑거 1-hull은 hull-fill 쐐기로 파지 구조적 불가 →
  **bg1 변형 B의 split2 충돌을 런타임 오버라이드로 이식**. 절차: `bg1_gripper_split2.usd`
  (SHA = `bg1_split2_audit.json` `out.sha256`, 시작/종료 재검증)에서 핑거당
  `collision_blade_split2`/`collision_mount_split2` 메쉬(points/faces/xform)를 bit-copy
  하여 run 스테이지의 `/World/ba1_robot/.../gripper_left|right` 아래 동명 프림으로 저작
  (approximation=convexHull), 원본 1-hull은 `collisionEnabled=false` (삭제 금지).
- 이식 게이트 (fatal): (a) census 핑거당 enabled 2 + disabled 1, (b) 복사 배열 SHA
  bit-일치, (c) blade 내측면 극값(글로벌 glf 프레임, q=0 기본 자세)이
  `bg1_split2_audit.json` 핀과 dev < 1e-9, (d) 조인트/질량 속성 공식 verbatim
  (오버라이드 0 — p18 attr_snap 방식 비교).

### 3-2. 운동학 모델 핀 (오프라인 FK/IK의 근거 — 실행 시 자산과 대조)

- 조인트 테이블 (joint1~6 + gripper 고정 변환)은 `payloads/Physics/physics.usda`
  (SHA `131e9e66…`) 값을 스크립트에 전사. 실행 시 조합 스테이지의 각 조인트
  `localPos0/localRot0/localPos1/localRot1/axis/limits`와 repr-일치 게이트 (fatal).
- **FK-vs-USD 실측 게이트** (SETTLE 종료): fk_glf(측정 q) vs USD glf world pose —
  위치 dev < 1e-3 m, 자세 dev < 0.5° (내 FK 모델이 PhysX/USD 합성과 일치함을 물리
  상태로 검증). FAIL = fatal (IK 전제 붕괴).
- 오프라인 IK (설계 시 확정, preflight 재계산 게이트): damped least-squares, 수치
  Jacobian, Shepperd 쿼터니언 오차, 수렴 게이트 pos < 1e-6 m ∧ ori < 1e-6 rad ∧
  한계 여유 > 5°. **preflight 재계산 q가 아래 §4 설계값과 max|Δq| < 0.1° 일치해야
  실행 진입** (fatal).

### 3-3. 드라이브/하네스 저작 (ADD-1 계보 — 사전 선언, 이탈 아님)

- 공식 자산은 drive stiffness/damping 미저작 (스키마 기본 0 = 드라이브 무력).
  - 팔 revolute ×6: stiffness **1e5**, damping **500** 저작 (§2 스모크 스윕으로 확정
    — 초안 2000/100은 유지 오차 0.54°로 기각, 1e5/500은 0.023°). **maxForce는 공식
    verbatim 유지** (27/27/27/7/7/7 N·m) → 토크는 실모터 스펙 한계로 포화. stiffness
    절대값의 현실성은 비주장 — 추종 성능은 G-track 게이트가 실측 판정.
  - **ADD-2 (계층 수리 — 스모크 1차 실증 결함의 반응 저작)**: 벤더 full 자산은
    9개 중첩 RigidBodyAPI 링크에 xformstack reset 미저작 →
    `omni.physicsschema` "missing xformstack reset" 에러 9건 + 전 조인트
    "no bodies defined" = **로봇 전체 미시뮬** (bg1 R1과 동일 계열의 벤더 자산 결함,
    D446 ③ 충돌 결함에 이은 2번째). 수리 = 참조 후 run 스테이지에서 각 링크의
    합성 world 행렬을 단일 transform op로 bake + `SetXformOpOrder([op],
    resetXformStack=True)` (파서 에러 메시지가 지시하는 수리이자 D447 ② 패턴).
    게이트: bake 전후 합성 world 변환 편차 < 1e-9 (실측 0.0). 기하·조인트·질량
    수치 무변경 (배치 보존 저작).
  - 그리퍼 prismatic ×2: bg1 ADD-1 verbatim — stiffness 5e3 N/m, damping 2e2,
    maxForce 공식 100 N (파지력 = min(k·err, 100 N) 포화, bg1과 동일 정책).
- articulation: `PhysxArticulationAPI` self-collision **disable** (공식
  `newton:selfCollisionEnabled=0` 의 PhysX 등가 저작) + 좌우 핑거
  `PhysicsFilteredPairsAPI` 1쌍 (R4 verbatim — interlock hull 아티팩트 제외),
  solver iterations 32/1, sleepThreshold 0 (팔·물체 공통, contact report 소실 방지).
- 물체/지지 계약: 원통 D 0.029/H 0.050 m/0.02483 kg, 마찰 0.40/0.30, restitution 0,
  solver 8/1, maxDepenetration 5, vel cap 10/10, sleepThreshold 0,
  PhysxContactReportAPI threshold 0 (bg1 §3-3 verbatim, 위치만 §4). 지지 =
  **pedestal box 0.05×0.05×0.095 m** (REV-1 — 상면 z=0.095, 물체 직하, 마찰 1.0;
  §4 정정 공식으로 팜 무접촉 검증) + 바닥 슬래브 2×2×0.02 m (상면 z=0, 팔 아래
  안전판, 마찰 1.0). 접촉 리포트 대상: 핑거 L/R, 팜, 물체, pedestal, 바닥 (팔
  링크의 의도치 않은 접촉은 별도 채널 기록 — 게이트 아님). 지오메트리 참조가
  instanceable이라 핑거 원본 충돌 비활성 저작 전 해당 instance root 2개를
  `SetInstanceable(False)` (합성 불변 — try-2 instance-proxy 저작 거부의 수정).
- Env pins: `numpy==1.26.0`, `psutil==5.9.8`, rerun 0.34.1, ffmpeg 7.0.2-static
  (`/home/cgxr/.local/bin/ffmpeg`).

## 4. 배치 + waypoint (오프라인 IK 스캔으로 설계 확정 — 61st, 물리 0)

> **REV-1 (본 실행 전 정정 — try-3 SETTLE 스모크 게이트 abort의 반응 수정, 파지
> 판정 데이터 소비 0)**: 초판 §4는 두 기하 오류를 포함했다. (a) **standoff 부호
> 오류** — bg1 §3-2의 접근축은 +x̂(팁 최전방, 팜 후방)인데 초판은 −x̂로 해석해
> standoff를 "물체 너머"에 배치, 팜(z 하단 = glf−0.0385)이 pedestal 위에서 스폰
> 겹침 → depenetration으로 물체 2.86 m 발사(try-3 실측: drift 2863.9 mm, calib
> NaN, 추종 6.1°). (b) **팜 간섭 공식 오류** — 팜 최근접 수평거리는
> 0.073+|X_TCP|가 아니라 **0.073−|X_TCP| = 0.0485 m**. 정정판은 부호 수정 +
> pedestal 축소 + 전 방위 재스캔으로 재설계했다. verdict 관련 게이트(§5)는 무변경.

- 물체 중심 `C = (0.34, 0, 0.12)` m (pedestal 상면 0.095 + H/2).
- 파지 프레임 (bg1 side 규약, 방위각 **φ=225°**): R 열벡터 = [x̂=(−cos φ,−sin φ,0),
  ŷ=(sin φ,−cos φ,0), ẑ=(0,0,1)], glf 목표 위치 t = C − R·[X_TCP,0,0],
  X_TCP = −0.02448 m (블레이드 스팬 중점, bg1 §4 verbatim). 접근축 = **+x̂**.
- waypoint (IK 설계값, 도 단위 [j1..j6] — preflight 재계산 일치 게이트 §3-2):
  - standoff (접근 **−0.05 m**, j5 ±90° 한계로 0.10은 전 방위 도달 불가 —
    블레이드 최전방—물체 표면 간격 11 mm로 무접촉):
    `[−37.525, −119.238, −20.195, −99.046, −82.525, 0.003]` (여유 7.4°)
  - grasp: `[−26.266, −119.595, −22.855, −96.741, −71.266, 0.001]` (여유 10.4°)
  - lift (+0.08 m z): `[−26.266, −92.183, −25.462, −66.723, −71.266, 0.001]` (여유 18.7°)
  - waypoint 간 최대 이동 11.3° (2 s 구간 = 평균 ~6°/s).
- 스폰 시 기하 무접촉 검증(해석적, 정정 공식): 블레이드 z 밴드 [0.100, 0.140] >
  pedestal 상면 0.095 (+5.4 mm), 팜 최근접 수평거리 0.0485 m > pedestal 반폭 대각
  0.0354 m (+13.1 mm).

## 5. Phases + gates

dt=1/60, 총 540 step (sim 9.0 s). 조인트 목표: 팔 = min-jerk 보간, 그리퍼 = 상수.

| Phase | steps | 팔 목표 | 그리퍼 목표 | 비고 |
|---|---|---|---|---|
| SETTLE | 30 | standoff 고정 | OPEN 0.0715 | 스모크 겸용: 추종/FK/스폰 게이트 |
| APPROACH | 120 | standoff→grasp | OPEN | |
| SETTLE2 | 30 | grasp 고정 | OPEN | 종료 시 TCP 오차 실측 (G-track) |
| CLOSE | 120 | grasp 고정 | **0.0 (전폐, 100 N 포화)** | bg1 CLOSE 정책 verbatim |
| LIFT | 120 | grasp→lift | 0.0 | |
| HOLD | 120 | lift 고정 | 0.0 | 종료 시 G2/G3 판정 |

- 게이트 (SUCCESS = G1∧G2∧G3):
  - **G1 close_bilateral**: CLOSE 중 같은 step에서 min(F_L, F_R) > **0.01 N** (bg1 상수).
  - **G2 lift_follow**: obj z 상승량(HOLD 끝 − CLOSE 끝) ≥ glf z 상승량 − **6 mm**,
    그리고 glf z 상승량 ≥ 0.06 m (팔이 실제로 들어올렸는가).
  - **G3 hold_slip**: |Δ(z_glf − z_obj)| (CLOSE 끝 → HOLD 끝) < **6 mm**.
  - **G-track** (분류용, SUCCESS 게이트 아님): SETTLE2 끝 TCP 위치 오차 — ≤10 mm면
    파지 판정 유효, >10 mm면 `BA1_TCP_TRACK_FAIL` 분류.
  - measurement_valid: SETTLE 중 obj drift < 2 mm, base 부동 (<1e-6 m), 전 step 조인트
    상태 유한, step 카운트 정합. SETTLE 중 지지 접촉력 median vs m·g=0.24358 N 4자리
    캘리브레이션 기록 (fg1/bg1 계보 — 기록, 게이트 아님).
- lift/hang 순간 양측력의 파지 해석 금지 (54th·55th 규칙 유지) — G2/G3가 유지 판정.

## 6. Outputs (forward-only, `g0d_d448/`만 쓰기; 실패 시 같은 태그 재실행 금지 → ba2)

`ba1_prereg.md`(본 문서) / `ba1_script.py.txt` / `ba1_argv.txt` / `ba1_results.json` /
`ba1_trace.npz` / `ba1_timeline.rrd` / `ba1_timeline.rbl` / `ba1_rerun_validation.json` /
`ba1_inspection.png` / `ba1_side_grasp.mp4` (1920×1080, 30 fps, h264 yuv420p, ~9 s) /
`ba1_key_*.png` (phase 경계 6장) / `ba1_stdout.log` / `ba1_exit_status.txt` / (실패 시)
`ba1_failure.json`. 중간 프레임 PNG는 scratchpad (세션 임시, 산출물 아님).

## 7. D341 observability + 영상 층위

- verdict가 궤적·접촉·시간에 의존 → **RRD 의무**: 전 540 step의 q(팔 6+그리퍼 2),
  드라이브 목표, F_L/F_R/bilateral-min/팜, obj z & glf z & slip, phase TextLog,
  키프레임 3D(핑거 점군+원통 wire+목표 axes), verdict TextLog. rerun 0.34.1 핀,
  footer verify, exact entity/timeline/component 계약, 고정 blueprint + `.rbl`,
  headless `ba1_inspection.png`, **실제 육안 검수 관찰 기록** (세션 doc).
- mp4/키프레임은 사용자용 시각 증거 — 보고 시 **G1~G3 수치 캡션 의무** (그림 단독
  제시 금지, bg1v 규율 상속). 권위는 results/trace, Rerun은 inspection evidence.
