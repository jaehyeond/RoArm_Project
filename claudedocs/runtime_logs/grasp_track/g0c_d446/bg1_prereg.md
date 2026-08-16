# `g0c_d446` / `bg1` preregistration — reBot B601 평행 그리퍼 flying-gripper 판별: D29×H50 원통을 쥐고 유지할 수 있는가 (충돌 표현 2변형)

- Date: 2026-08-13 KST (59th session)
- User authority: 59th 채팅 사용자 명시 승인 ("1번 진행해. 승인할게") — B601 sim 판별
  case 개시. 교수님 보고 패키지는 사용자가 명시적으로 보류.
- 이번 case의 신규 변수 (2개): `[그리퍼 기종 = reBot B601-DM 평행 그리퍼 (팔 제거 유지,
  fixed-root articulation)]`, `[충돌 표현 = A(공식 USD 1-hull/핑거 verbatim) vs
  B(동일 핀 소스 충돌점의 blade/mount 2-piece 분해)]`.
- Scope: 물리 실행 O (실패 가능), Isaac Sim 5.1 로컬 4090 headless (RTX 렌더 0),
  로봇 하드웨어 0, RunPod 0, lerobot-train 0, `g0a_*`/`g0b_*` 편집 0.

## 1. Decision question / branch semantics / non-claims

- 질문: 팔·IK·궤적 제약이 전혀 없는 이상 배치에서, B601 평행 그리퍼 기하가
  D29×H50 / 24.83 g 강체 원통을 **양측 접촉으로 쥐고 유지**하는가?
  fg1(D445, RoArm attempt3 그리퍼 0/13)과 동일 프로토콜·동일 게이트의 대조 실험.
- 변형 의미: **B(분해 표현) = 사용자 구매 질문의 1차 답** (실효 기하가 실물 블레이드에
  충실). A(공식 USD verbatim) = 벤더 sim 자산 사용성 판정 (§3-2의 hull-fill 쐐기 예측).
- Branch semantics:
  - (i) B ≥1 SUCCESS & A 0 → 실기하는 유지 가능, 공식 USD 충돌 표현이 병목
    (`REAL_GEOM_HOLDS_USD_COLLISION_BLOCKS`) — 구매 논거 지지 + 벤더 자산 결함 기록.
  - (ii) A ≥1 & B ≥1 → 공식 자산 그대로도 유지 (`HOLDS_EVEN_ASWRITTEN`) — 최강 지지.
  - (iii) B 0 → 이 프로토콜·이 sim 계약에서 B601 기하도 유지 실패
    (`B601_FAILS_UNDER_PROTOCOL`) — 구매 논거 불충분, 원인 분석으로.
  - (iv) A ≥1 & B 0 → `ANOMALY` (표현 충실도와 결과가 역전 — 하네스/저작 의심, 판정 보류).
- Non-claims: 실물 B601 파지 성공(구매 후 실검증 필요), RoArm→B601 팔 기구학/IK,
  마찰 현실성, 학습 라벨, D419 재판정, fg1/D445 재판정, 벤더 실기 성능. sim 성공은
  "구매 가치 판단의 필요조건 충족"이지 실기 보장이 아니다.

## 2. Method authority

- fg1 프로토콜 상속 (57th prereg + 58th 실행): `GraspingManager.evaluate_grasp_poses`
  (설치 `isaacsim.replicator.grasping` 1.0.9, extension.toml SHA `5e599aaf…` 재확인),
  default scene 모드(`physics_scene_path=None`, fg1 DEV-2 교훈 선반영), root 텔레포트
  fixed-root articulation, 자체 HANG 게이트, PhysxContactReportAPI threshold 0 명시
  (50th-b N-4), D442 lifecycle (fsync → sentinel → `SimulationApp.close()` 최종).
- prismatic 조인트: `grasping_utils`가 `linear` drive로 분기, target 단위 = m
  (설치 소스 `grasping_utils.py:198-201,268-271,324-327` 직접 확인).
- 신규 하네스 요소(스모크 필수 검증): B601 자산 텔레포트 재앵커(S1), prismatic
  open/close 왕복(S3, 자유공간; **핑거 상호충돌 여부 실측** — articulation
  self-collision 기본값 판별), pedestal 지지(S2/S5), 접촉력 캘리브레이션 m·g(S8).
  스모크는 후보 pose 사용 금지 (fg1 규율).

## 3. Frozen inputs / pins

### 3-1. 자산 (신규 케이스 자산, 본 폴더 안, forward-only)

- 소스: `b601_asset/` — Seeed-Projects/reBot-Isaacsim upstream commit
  `cb824be157fdd5db7d6153b644b9b8ce85775bef` verbatim 9파일, SHA-256은
  `b601_asset/UPSTREAM.md` (실행 시작/종료 재검증, drift = fatal). 라이선스
  CERN-OHL-W-2.0/Apache-2.0 (`LICENSE_upstream`).
- 변형 A: `bg1_gripper_only.usd` — 59th 추출 완료. flatten + CopySpec으로
  `gripper_link` 서브트리(팜+핑거 2) + `gripper_joint1/2` verbatim, root를
  gripper_link 프레임 fixed-root articulation으로 저작. SHA-256
  `bg1_asset_audit.json` `extraction.sha256` (게이트 a census 1+1+1 convexHull /
  b 조인트 13속성 bit-일치 / c 메쉬 SHA 11종 일치 / d 질량 4속성 일치 /
  e 패드 기하 재측정 <1e-9 — **전부 PASS**, `all_gates_pass=true`).
  - **저작 리비전 R1~R4 (본 실행 전, 스모크/진단 증거 기반 반응 수정 — 기하·조인트
    수치 무변경, 구조/스키마 배치만)**:
    - R1: 원본의 중첩 rigid-body 계층(핑거가 팜의 자식)이
      `omni.physicsschema` "missing xformstack reset" 에러로 핑거 미시뮬 →
      **flat 형제 계층**(attempt3 규약)으로 재배치. 핑거 로컬 xform·조인트 프레임
      값 verbatim 유지 (glf=identity라 배치 동일).
    - R2(폐기): ArticulationRootAPI를 root body에 저작 → PhysX가 프레임 미저작
      root_joint를 world identity에 앵커("disjointed body transforms" 경고),
      물리 몸체가 저작 원점으로 스냅되어 텔레포트 pose가 무접촉 (낙하 프로브
      z≈0.014 정지로 실증).
    - R3(폐기): articulation 제거(maximal joints) → JointStateAPI 초기화/판독
      미지원으로 핑거가 닫힌 채 스폰 → 스폰 관통 폭발.
    - **R4(채택)**: ArticulationRootAPI를 **root_joint 프림 위**(Isaac fixed-base
      관례)에 저작 → 재앵커·상태 초기화·판독 전부 정상 (진단: settle drift
      0.001 mm, A=쐐기 배출 42 mm / B=60 N 양측 유지 — §3-2 예측과 정합).
      좌우 핑거 상호 hull 충돌(실물 브래킷은 무접촉 교차 — hull-fill 아티팩트)은
      `PhysicsFilteredPairsAPI`(left→right 1쌍)로 제외; 팜-핑거 쌍은 조인트 기본
      제외.
  - **ADD-1 (저작 파라미터, 이탈 아님 — 사전 선언)**: 공식 자산은 linear drive
    stiffness/damping 미저작(스키마 기본 0 = 드라이브 무력). stiffness 5e3 N/m,
    damping 2e2 N·s/m 저작. maxForce는 공식 값 100 N verbatim 유지 → 파지력은
    min(k·err, 100 N) 포화.
  - root_joint = FixedJoint body0=[] + **프레임 미저작** (D445 ③ⓑ 재사용 패턴).
- 변형 B: `bg1_gripper_split2.usd` — A 파일에서 파생. 각 핑거의 공식 충돌 메쉬
  점군(19,481점)을 x_glf = **-0.050 m** 평면에서 blade(x≥-0.050)/mount(x≤-0.050)
  2조각으로 나누고 각 조각의 convex hull 메쉬를 신규 충돌 프림으로 저작
  (approximation=convexHull), 원본 1-hull 충돌은 `collisionEnabled=false`로 비활성
  (삭제 금지 — attempt3 legacy-disabled 패턴). 팜은 공식 1-hull 유지.
  저작 게이트 (fatal): 조각 정점 ⊆ 원본 점군(bit), blade 조각 내측면(+y_L/−y_R)
  극값 == 원본 blade 극값(bit), census 핑거당 enabled 2 + disabled 1,
  A 대비 그 외 diff 0. SHA는 `bg1_split2_audit.json`에 기록.

### 3-2. 자산 실측 근거 (59th 감사, `bg1_asset_audit.json` + pads npz)

- glf 프레임: 접근축 = **+x̂** (팁 x=0, 블레이드 x∈[-0.04896, 0], 팜 x∈[-0.151,-0.073]),
  개구축 = ŷ (좌 -y / 우 +y 이동, travel 각 [0, 0.0715] m), q=0 = 닫힘(블레이드 내측면
  y≈0 맞닿음), **개구 = qL + qR** (최대 143 mm), 블레이드 z 폭 ±0.0196 m.
- **hull-fill 쐐기 (변형 A 예측, LP 단면 실측)**: 핑거 1-hull이 교차형 마운트
  (+58.6 mm까지 반대편 돌출)와 블레이드 팁을 이어, 블레이드 스테이션에서 실효
  충돌면이 실제 패드면 대비 x=0/-10/-24.5/-39 mm에서 +0.03/+7.1/+17.4/+27.7 mm
  돌출 (쐐기각 ~35°, tan 0.71 > μ_eff~0.45 → 원위 배출 예측). **이 예측은 게이트가
  아니라 기록** — A의 실측 결과가 판정.
- 질량: 팜 0.1818 / 핑거 각 0.0423 kg (공식 verbatim).
- 마찰: 그리퍼 측 물리 재질 미저작 (공식 그대로 유지 → PhysX 기본 재질), 물체/지지대는
  §3-3 계약 값.

### 3-3. 물체/지지 계약 (fg1 §3과 동일, 대조 유지)

- 원통: 해석적 D 0.029 / H 0.050 m / 0.02483 kg, 정립, base-frame 중심
  `[0.4235072423787768, 0.17237803311822986, 0.025]`, 마찰 0.40/0.30, restitution 0,
  solver 8/1, maxDepenetration 5, vel cap 10/10, **sleepThreshold 0 명시 저작**
  (계측 인프라 — 정지 물체 sleep 시 contact report 소실을 스모크로 실증,
  threshold 0 명시와 같은 계열의 하네스 설정).
- 지지: **pedestal box 0.1×0.1×1.0 m** (상면 z=0, 물체 중심 직하) — fg1의 2×2 m 슬래브
  대체. 사유(사전 선언): B601 팜 z 밴드 ±0.0385가 side pose에서 z<0으로 내려가
  광역 슬래브와 겹침; pedestal은 팜 최근접 수평거리 0.073 m > 반폭 0.05 m로 무접촉.
  마찰 1.0/1.0, restitution 0. HANG = pedestal collider 비활성 (fg1 메커니즘 동일).
- Env pins: `numpy==1.26.0`, `psutil==5.9.8`, rerun 0.34.1 (D326/D325).

## 4. Pose set (13, 해석적 구성 — 외부 후보 파일 없음)

fg1의 13 pose는 RoArm 그리퍼 프레임 산물이라 이전 불가. bg1은 실측 §3-2 기하로
닫힌형 구성 (구성 게이트: R 직교성·det=1 ≤1e-12, 배치 항등식 ≤1e-12, fatal):

- 상수: `X_TCP = -0.02448` m (블레이드 스팬 중점), `BITE = 0.012` m (T1 실물 rim 물림
  밴드 0~12 mm 상단, D430 계보), `OPEN = 0.0715` m 양지, `CLOSE = 0.0` m 양지.
- **side 8**: 방위각 φ = k·45°, k=0..7. R 열벡터 = [x̂=(−cosφ,−sinφ,0),
  ŷ=(sinφ,−cosφ,0), ẑ=(0,0,1)], t = C_mid − R·[X_TCP,0,0], C_mid = 물체 중심.
  검증: 블레이드 z 밴드 [0.0054, 0.0446] — 지지 무접촉.
- **top 5**: 기움 θ ∈ {0, 6, 15, 24, 35}° (fg1 rim ladder 각도 계승 + θ=0 수직 추가 —
  D430 수직 축과 직접 대조), 기움 방향 +x 고정. R 열벡터 = [x̂=(sinθ,0,−cosθ),
  ŷ=(0,1,0), ẑ=(cosθ,0,sinθ)], t = (C_x, C_y, z_top − BITE), z_top = 0.05.
  θ=0에서 팁 z=0.038, θ=35 블레이드 최저 코너 z≈0.027 — 지지 무접촉.
- CLOSE 정책 = 전폐 목표 0.0 m + maxForce 100 N 포화 (평행 그리퍼 표준 정책이자
  fg1 관통 목표 14°의 유사체). **폭-정지 정책은 이번에도 시험하지 않는다** (비주장 유지).
- 실행 순서: side 0..7 → top θ 오름차순, 셔플 금지. 변형 A 13 pose 완주 후 스테이지
  재구성, 변형 B 13 pose. 26 실행 전부 완주가 완주 조건.

## 5. Phases + gates (fg1 §5 동일 구조)

- PREGRASP: qL=qR=OPEN, 60 steps @ dt 1/60 (사전 `apply_joint_pregrasp_states`).
- CLOSE: qL=qR=CLOSE, 120 steps.
- HANG: pedestal collider off → 240 steps (30×8 청크, 낙하 곡선) → 복원.
- Gates: `close_bilateral` = 같은 step에서 min(F_left, F_right) > **0.01 N** /
  `HOLD` = hang 낙하 < **6 mm** / **SUCCESS = close_bilateral AND HOLD**.
- Taxonomy (fg1 동일): `NO_JAW_CONTACT` / `ONE_JAW_ONLY_LEFT` / `ONE_JAW_ONLY_RIGHT` /
  `BILATERAL_NO_HOLD` / `PRECLOSE_COLLISION` (+ SUCCESS). lift/hang 순간 양측력 파지
  해석 금지 (54th·55th 규칙). 팜-물체 접촉력은 별도 채널로 기록 (게이트 아님).
- measurement_valid: root 오차 <1e-6 m, spawn drift <2e-3 m, step 카운트 정합.

## 6. Outputs (forward-only, `g0c_d446/`만 쓰기)

기존: `bg1_prereg.md`(본 문서) / `b601_asset/` / `bg1_gripper_only.usd` /
`bg1_asset_audit.json`. 신규: `bg1_gripper_split2.usd` / `bg1_split2_audit.json` /
`bg1_script.py.txt` / `bg1_argv.txt` / `bg1_results.json` / `bg1_trace.npz` /
`bg1_timeline.rrd` / `bg1_timeline.rbl` / `bg1_rerun_validation.json` /
`bg1_inspection.png` / `bg1_stdout.log` / `bg1_exit_status.txt` / (실패 시)
`bg1_failure.json`. 실패 시 같은 태그 재실행 금지 (bg2 forward-only).

## 7. D341 observability

fg1 §7 동일 + 변형 축: RRD에 variant(A/B)×pose 타임라인, 접촉력 시계열
(left/right/bilateral-min/palm), hang 낙하, post-close 기하(핑거 점군·원통·pedestal),
verdict TextLog. rerun 0.34.1 핀, footer verify, exact entity/timeline/component 계약,
고정 blueprint + `.rbl`, headless PNG, **실제 육안 검수 관찰 기록** (생성만으로
"inspected" 보고 금지).
