# Session 2026-08-03 (3rd) — D409 스윕 회수 완료 + 설계 v1 + 4-lens 리뷰 + 정적 준비

이번 case의 신규 변수: [실물 원통 기하 — 사용자 실측 확인 D29×H50mm] (1개)

## 0. 승인 범위

- 사용자 "설계 착수" 승인 지속 (2nd 세션 doc §0; D407 전례 = 설계 + 정적
  준비 + attestation/tuple까지). **attempt1 실행은 tuple SHA 인용 별도
  명시 승인 후에만.**
- 과학 상태 불변: D407 FAIL-STOP, `g0a_pass=false`.

## 1. 스윕 회수 (완료)

`wf_aaa9aa61-d2a` journal result 4건 전문 판독 완료 — 기록 위치 = 2nd 세션
doc (`session_20260803_grasp_g0a_d409_design_kickoff.md`) **§4** (reader별
전체 findings + 차이 보고 A~G 표 + 인계 수치 대체 완료 §4.6). verbatim
보존 = `g0a_d409/design_inputs/evidence_sweep_wf_aaa9aa61-d2a/` (manifest
`d409_design_inputs_manifest.json`). FK 인계값 0.0013mm는 재도출 실측
**0.001216972820130102mm**로 대체 완료 (2회 실행 bit-exact PASS;
`g0a_d409/design_inputs/d409_fk_tcp_scalar_rederivation.json` sha256
`c0b13007d36de91b6aa8f1190d6d14f8e45e39564325292a5b85d51a0655d5aa`).

## 2. D409 설계 v1 — 실물 원통 zero-step 양측 접촉영역 전수 열거

### 2.1 목표와 판정 범위

- 목표: 실물 원통(사용자 실측 D29×H50, HARD RULE #18)에 대해, 순수
  오프라인(FK + hppfcl, Isaac 실행 0)으로 TCP 오프셋 격자를 **전수**
  평가하여 (i) 양측 조(고정=link5, 가동=gripper_link)의 접촉/근접
  영역 지도, (ii) 재정식화된 "fixed-jaw 먼저" 순서 제약 채점, (iii)
  영역 대표(robustness radius) 채점을 산출한다.
- 판정 범위 상한: **접촉 영역 지도 + 순서 제약 채점까지.** stable grasp,
  force closure, grasp feasibility, 파지 성공, PhysX/실물 결과 예측은
  전부 null claim. `g0a_pass` 변경 없음. zero-step 결과는 닫힘 동역학
  (D362형 밀어넘김)을 대체하지 않는다 — 라벨 사다리 1단(기하) 전용,
  기하 단독 학습 승격 금지 (방향 결정 ②).

### 2.2 신규 변수 (1) + 파생 상수

- 신규 변수: 실물 원통 기하 = **r=0.0145m, H=0.050m** (사용자 실측 확인
  2026-08-03, 2nd doc §1; 측정 기기 미보고 — 보고 시 provenance 추가).
- 모델링: **analytic `hppfcl.Cylinder(0.0145, 0.050)`** — D379 지속 규칙
  ①(primitive 우선, 실측 비원통 특징 증거 전 decomposition 금지) 준수.
- 파생 상수 (신규 변수에서 유도 — 별도 변수 아님):
  - 원통 배치: x=0.30000001192092896 (float32(0.3), 동결 계보 동일 규약),
    y=0.0, z_center = table_z + H/2 = **0.01288299962878227m**, 여기서
    table_z = 0.03288299962878227 − 0.045 = −0.01211700037121773m
    (동결 D34×H90 object z에서 역산한 동결 세계 table plane — 동일 테이블).
    quat 항등(축 수직 직립). 단일 배치(placement 일반화는 scope 밖).
  - 격자 도메인 rebase: radial 상한 17,000µm(구 반경 결부) →
    **14,500µm**; anti-retreat "17mm−r" → **"14.5mm−r"** (차이 보고 D/
    격자 전례 rebase 의무).
- 질량 24.83g: **harness 미사용.** 문서 맥락 상수(전도 임계 0.071~0.141N)
  로만 존재. 마찰 미도입.

### 2.3 동결 입력 (전부 sha256 pin — prereg `input_hashes`에 등재)

| 입력 | 파일 | sha256 (본 세션 실측) |
|---|---|---|
| A64 가동 조 기하 (권위) | `g0a_d339/collision_asset/attempt2/d339_gripper_link_cold1_canonical_geometry.json` | `dc258b27cdef5d29e23f1b5ef3041c3afb26f50d8c8ad9b222002532e95f2e5e` |
| A64 고정 조 기하 (권위) | `g0a_d339/collision_asset/attempt2/d339_link5_cold1_canonical_geometry.json` | `c45bd056b3487f92bc724474dbf850ea6da309fea90c4e0a90879ada7ba2b655` |
| part 마스크 (4/17/16) | `g0a_d368/d368_semantic_allocation_evidence.json` | `be2a422b0c74e4781b76a640c5312070b84876b1cb9e661d47e705ccdf789cf5` |
| 동결 자세/anchor | `g0a_d349/d349_frozen_target_distance_measurement.json` | `5de6d14e37d6b74b202d1bb668120a6bb57221eac24ea5c751457ce9823b6300` |
| FK 기하 원천 | `local_assets/roarm_m3/urdf/roarm_m3.urdf` | `64dc8d082cbce9a13a0697a11a0eaeaebbc54bbcd97e7aafaa40da483969dae2` |
| anchor 거리 원천 | `g0a_d371/d371_offline_collider_comparison_evidence.json` | 정적 준비에서 pin |
| FK 스칼라 재도출 | `g0a_d409/design_inputs/d409_fk_tcp_scalar_rederivation.json` | `c0b13007d36de91b6aa8f1190d6d14f8e45e39564325292a5b85d51a0655d5aa` |

- A64 소스 권위 결정 (차이 보고 C): **d339 cold1 canonical geometry 2파일
  = 권위** (per-part vertices_m/triangles + part별 geometry_sha256 내장).
  정적 준비에서 `g0a_d348/attempt2/d348_callback_topology_volume_evidence.json`
  의 part 기하와 **64+64 bit-동일성 확인** — 불일치 시 설계 무효 fail-stop
  (모호성 해소 전 진행 금지).
- 마스크 계보 (차이 보고 F): 단일 소스 = D368 evidence 한 파일.
  link5_fixed 4 [part_027, part_029, part_030, part_031] / gripper_inner
  17 / gripper_outer 16(**음성 대조 전용**). D339 verdict는 FAIL_STOP이나
  asset build+cook witness는 PASS이고 D368이 pin한 관계를 prereg에 명시.
- part 좌표는 body-local (prim→body 항등; 스윕 §4.3) — FK body pose 1개로
  64 part 전부 world 변환.

### 2.4 기구학 계약

- **FK 상수 계열 = URDF 리터럴 단일 계열** (차이 보고 G/3c 처분): FK
  체인은 `roarm_m3.urdf` XML 직접 파싱(d351 `_joint_spec` 패턴)으로 구성.
  `roarm_kinematics._CHAIN`(pi/2 심볼, 0.05196, gripper 조인트 부재)은
  **사용 금지**. 가동 조 확장 = URDF `link5_to_gripper_link` 리터럴
  (origin (0, 0.018821, 0.052035), rpy (−1.5708, −1.5708, 0), axis z)
  — d332:1264 리터럴 계열과 일치, `-math.pi/2` 계열 혼용 금지.
- q5 = `link5_to_gripper_link`, OPEN = 1.5413000583648682 rad (float32),
  CLOSED = 0.0. gripper_link pose(q5) = link5_pose ∘ T_urdf ∘ Rz(q5).
- IK: d323 알고리즘 형태 보존(HOME [0,0,90,0,0,0]° seed, position-only
  5-DOF DLS, max_iter 120, pos_tol 1mm, step clip 4°/step, v6 소프트
  리밋 clip) — 단 FK만 URDF 리터럴 체인으로 교체. **선언된 편차**:
  d335 격자 결과와 bit-호환 아님(비교 anchor는 §2.9 재현 게이트이지
  d335 수치 재현이 아님). 난수 사용 0 (DLS 결정적).
- 관절 리밋 결정: reachable 판정은 **v6 소프트 리밋** (d335~d337 전례
  일치; URDF 하드 리밋은 진단 기록만).
- FK 재현 정확도 서술 규칙: "bit-exact" 주장 금지 — "≤X µm" 형태로
  정적 준비 실측값을 pin (스윕 §4.4: 상수 불일치로 µm대 잔차 구조 존재).

### 2.5 격자 정의 (전수)

- 축 = TCP 오프셋 공간 (d335 계보): target_TCP = center − radial_unit·ρ
  − tangent_unit·τ, z = z_center(실물), tangent_sign = −1.
- 도메인: ρ ∈ {0, 250, …, 14,500}µm (59점) × τ ∈ {9,000, 9,250, …,
  14,000}µm (21점) = **1,239 pose 균일 전수** (µm 정수 키; 조건부 refine
  없음 — 전수 표방이므로 d335의 coarse-실패시-국소정련 구조 폐기).
  τ 도메인은 조(jaw) 측 기하 유래로 판단해 전례 그대로 상속 —
  원통 반경 비결부 (리뷰 확인 항목).
- 모든 pose는 결과 row를 가진다 (admission 실패 pose도 사유와 함께
  기록 — silent cap 0).

### 2.6 q5 닫힘 arc sweep (zero-step)

- 전례 = D351/D354 (직접 상태 기입 격자 + 이분법) — D362 동역학 응답과
  축이 다름을 명시 (시간 step 축 아님, q5 각도 축).
- anchor: float32 linspace(OPEN → 0.0, 33). 인접 anchor 간 누락 접촉
  배제 = 현 길이 상계 `2·Rmax·sin(|Δq|/2)` (Rmax = gripper part 정점의
  조인트 축 최대 반경, 계산·기록).
- 가동 조 첫 교차각 q5*: 이분법, bracket 폭 ≤ 1e-6 rad, 최대 재귀 32
  (수치 제어; 과학 tolerance 아님 — D351 문언 승계). endpoint 유효성
  계약 승계 (clear: distance>0 ∧ witness 분리 / overlap: distance≤0 ∧
  EPA penetration 유한).
- 최적화 (수학적 동치): link5(고정 조)는 q5 불변이므로 pose당 1회만
  질의; arc sweep은 gripper 64 part만.

### 2.7 "fixed-jaw 먼저" 순서 제약 — 재정식화 (차이 보고 E 처분)

zero-step에서 고정 조의 "첫 교차 q5 각도"는 정의 불능 (link5는 q5의
부모 — D354 sweep CSV 전 anchor에서 raw_link5_mm 상수 실증). 따라서:

- **(A) 자세 수준 0차 조건**: q5=OPEN에서 link5 inner-mask(4 parts)
  min signed distance d_fix ∈ [0.1mm, 5.0mm].
  하한 0.1mm = CLEAR_GATE(D339/D349, 무관입 보장), 상한 5.0mm = D330
  fixed_jaw gap 게이트 승계. 근접 밴드 {≤1mm, ≤5mm} 별도 기록.
- **(B) 가동 조 첫 교차 존재**: q5* ∈ (0, OPEN) 존재 ∧ 첫 교차 part ∈
  gripper_inner 17 ∧ q5*에서 비-inner gripper part 전부 clear (>0)
  (competitor exclusion) ∧ 원통 witness가 barrel_interior (strict).
- 순서 제약 채점 = A ∧ B. D362 밀어넘김은 A 위반 자세(fixed 4.27mm
  이격)에서 발생 — 재정식화가 원 의도 보존 (스윕 §4.2). 단 A∧B는
  기하 필요조건이지 밀어넘김 부재의 충분조건이 아님을 명시 (null claim).

### 2.8 채점 계약

- pose 수준 admission (d335 legacy_checks의 FK-등가 재정의): ik_converged
  ∧ commanded TCP 오차 ≤5mm ∧ jaw tangent ≤15° (FK-유도 등가식, harness
  에 수식 pin) ∧ 양측 all-64 무관입(q5=OPEN에서 min distance ≥ 0.1mm —
  단 link5는 (A)와 결합해 [0.1, 5.0]mm) ∧ anti-retreat 14.5mm−ρ ≥ 0.
- 구 "top −15mm" 규칙: **비게이트 진단 기록만** (구 기하(H90) 유래 상수
  이중 적용 회피; cap/rim 배제는 strict 분류기가 담당) — 리뷰 확인 항목.
- barrel/cap 분류기: `_feature_from_cylinder_witness` 이식 + **R/H rebase
  (0.0145/0.050)**. strict 부등호 유지, 사후 tolerance 도입 금지 (D354
  durable rule).
- pinch predicates (d351 `_bind_moving_surface` 수식 이식, 입력은
  FK+hppfcl witness로 대체 주입): 핵심 4개(법선 마주봄, 중심 기준 반대편
  xy, chord 사영 0<t<1, 닫힘 속도 방향 정합)를 **게이트**, 나머지는
  진단 기록. D354 pass=false 전례 — 수식 구조 재사용이지 PASS 결과
  재사용 아님을 prereg에 명시.
- **영역 대표 채점 (차이 보고 D 처분)**: admitted pose들의 4-연결 성분 =
  영역. 영역 대표 = 내부 최심 셀(비허용 경계까지 offset-공간 최소 거리
  최대화; 동률 시 키 사전순). 영역 robustness radius ρ_R(mm) 산출 후
  3중 병기: (i) ρ_R vs **7.881mm** (D330 근접 regime 상한 — 실측 10값
  이봉 구조), (ii) ρ_R vs **36.033mm** — '역사적 실행 오차 proxy
  (D34×H90-era, 단일 목표, 비-replica 평균)' 라벨로만, (iii) stall
  regime (70~81mm) 대비는 도달 불가능 사실 명기. **평균 36.033 단독
  pass/fail 게이트 금지.** offset-공간 거리는 TCP 변위 proxy(z 제외)임을
  한계로 명시.

### 2.9 정합성 anchor 게이트 (정적 준비에서 실측 후 허용오차 pin)

1. **동결 자세 FK 재현**: D349 float32 관절 → URDF-리터럴 FK → link5/
   gripper_link pos+quat vs D349 저장 라이브 pose. 실측 잔차를 pin
   (예상 µm대; pin 전 게이트 발동 금지).
2. **저장-거리 재현**: 동결 자세 FK pose에서 A64 part 질의 → D371 저장
   거리 (link5 4.272736580324082mm / gripper_link 11.340262326338637mm,
   구 원통 (0.017,0.090) 구 배치 사용) 대비 잔차 pin. **명시 구분**: 구
   기하 사용은 질의 파이프라인 calibration 전용이며 D362-era 물리 결과의
   실물 전이가 아님 (D379 위반 아님 — 결과 전이 금지 조항은 접촉순서/
   전도 결과에 대한 것).
3. **결정성**: attempt1은 전 열거 2회 실행, canonical evidence bytes
   bit-exact 요구 (난수 0, NM 정련 없음 — 결정적이어야 정상).
4. hppfcl 질의 파라미터 전례 승계: BVHModelOBBRSS(callback topology
   삼각형), DistanceRequest(True, 1e-9, 1e-9), gjk 1e-9/1000,
   CollisionRequest enable_contact, 서명 거리 = overlap 시 EPA 최대
   깊이 음수 (d349 방식).

### 2.10 Harness 구조

- 파일 2개 (D407 전례): `sim_scripts/cylr29h50_d409_zero_step_dual_jaw_contact_region_enumeration_controller.py`
  + `..._worker.py` (신규 파일 — 동결 스크립트 수정 0; 신규 object 접두
  `cylr29h50_`로 cyl34 계열과 상수 혼입 방지).
- 프로세스: controller(supervisor) 1 + worker 1, `python -B`, worker
  호출 2회(결정성 run1/run2 — D371의 단일 호출에서 확장, prereg에 명시),
  retry 0, timeout 7,200s/run. 원자적 exclusive claim + preclose
  sentinel + stdout/stderr sha256 (D371 골격).
- scope_guards (전부 0 명시): isaac/kit/physx/warp/cuda/gpu import,
  **AppLauncher=0, physx_cook_callbacks=0** (차이 보고 A 처분 — D371과의
  차이 등록; A-family 경로만 준용, 신규 cook 금지), SimulationContext,
  physics step, USD 읽기/쓰기, asset 쓰기, robot HW/serial, lerobot,
  신규 패키지 설치(D326). 허용 = 동결 JSON read + hppfcl/numpy/trimesh
  오프라인 질의 + g0a_d409/ 쓰기.
- env pins (스윕 실측 = 현재 설치와 일치): isaaclab env python 3.11.14
  절대경로, numpy 1.26.0, psutil 5.9.8, hpp-fcl 2.4.4, scipy 1.15.3,
  trimesh 4.5.1, rerun-sdk 0.34.1.
- prereg 스키마 = D371 계보 (artifact/case/new_variables/head pin/
  input_hashes/environment/candidate_contract/registered_metrics/gates/
  negative_controls 2계층/visual_contract/scope_guards/single_run_contract/
  interpretation_boundary/registered_worker_command) + D407 계보
  (allowed_dirty_paths = live dirty + planned static + **소비자 소스에서
  프로그램적으로 도출한 future runtime leaf/디렉토리 sentinel** 합집합;
  원자적 attestation/tuple 게시; crash resume fail-closed 규칙).

### 2.11 대조 (2계층 — D371 전례)

- prepare-시점 음성 대조 (실패 가능해야 PASS):
  1. 원통 반경을 0.017로 변조 → 실물 기하 pin 체크 FAIL 확인.
  2. part vertex stream 1바이트 변조 → geometry_sha256 FAIL 확인.
  3. inner 마스크를 outer 16으로 치환 → 마스크 name-set/sha FAIL 확인.
  4. FK 상수를 pi/2 계열로 치환 → anchor 게이트(§2.9-1/2) 허용오차
     초과 FAIL 확인 (실측으로 판별력 검증 — 판별 불가 시 대조 재설계).
  5. 빈 격자(0 pose) → admission FAIL 확인.
- audit_registered 음성 대조:
  1. barrel/cap에 tolerance 도입 시도 → strict 계약 위반 reject.
  2. bisection bracket 1e-6 초과 → reject.
  3. 36.033 평균 단독 게이트 → 채점 계약 위반 reject.
  4. isaac import → scope guard reject.
- 양성 대조: 동결 (7,11) anchor 재현 (§2.9-1/2) + 격자에 (7000, 11000)
  키 포함 확인 (신규 center 기준 — 의미는 동결 후보와 다름을 명시).

### 2.12 관측성 (D341 계약)

- 기하 verdict → RRD 의무. save-only, file sink를 첫 user log 전 attach,
  종료 시 finalize → footer verify.
- entity 계약 (cook/representation 분리 원칙): `/enum/source/<body>/part_NNN`
  (A64 body-local 원본), `/enum/prototype/cylinder` (실물 primitive),
  `/enum/instance/<대표 pose>` (영역 대표 + anchor 자세 — 전 1,239 pose
  는 evidence JSON이 권위, RRD는 대표 subset을 blueprint에 pin),
  `/enum/candidate/...` (접촉 witness, 영역 경계, q5* 마킹),
  `/metadata/run`. timeline: `q5_arc` (대표 pose들의 arc sweep).
- 수치 권위: Float64 = evidence JSON. RRD Float32 사본은 검수 전용 —
  과학 gate 재해시 금지.
- 실행 세부는 D404~D408 수리 패턴 (절대경로 rerun CLI — 정확 경로 정적
  준비 실측 pin, 1920×1080 정확, ppp 2.0, 오류 배너 0). D371 관측성
  구현은 FAIL 전례 — 참조 금지.
- 수동 검수: pre-armed atomic writer (runtime 전 완성·dry-run, PID/phase
  handshake, deadline, no-replace+fsync; D407-R1/D408 admission
  prerequisites). blueprint + RBL export + headless decision screenshot +
  육안검수 JSON/기록.

### 2.13 산출물 + 예산

- 출력 폴더: `claudedocs/runtime_logs/grasp_track/g0a_d409/attempt1_zero_step_dual_jaw_contact_region_enumeration/`
  — prereg, evidence JSON (권위: 전 pose row + witness + 채점), 영역
  지도 CSV, run1/run2 결정성 증빙, RRD/RBL/screenshot/manual, phase
  markers, worker invocation/supervisor, static fixture results,
  attestation, tuple.
- 질의 예산 추정: pose당 link5 64(1회) + gripper 64×33 anchor + 이분법
  ≈ 2,200 → 전 격자 ≈ 2.7M 질의/run × 2 run. D371 실측(378 질의+분석
  수 분) 외삽 시 run당 수십 분 — timeout 7,200s 이내. silent cap 없음.

### 2.14 Null claims / interpretation boundary

- stable grasp / force closure / grasp feasibility / 파지 성공 / 밀어넘김
  부재 보증 / cap·rim·barrel 접촉 순서의 동역학 / SDF 우월성 / 다른
  원통·배치 전이: 전부 null.
- 기하 라벨 단독 학습 승격 금지 (방향 결정 ②). PhysX 스크린·실물
  캠페인은 별도 case·별도 승인.
- `g0a_pass=false` 불변. D399 사용 금지 (D398-F1 예약).

## 3. 4-lens 적대적 리뷰 (회수 후 기록)

- 발사 이력: run `wf_c46dc45e-62d` (science/frozen/lessons/ops 4 렌즈,
  각 렌즈 blocker/warning/confirmation 스키마 강제). 1차 발사는 세션
  재기동으로 4 agent 전부 조사 도중 중단 (전사 75~86행, StructuredOutput
  미호출 — 회수 불가 확인 후 **동일 run resume으로 verbatim 재발사**.
  설계 파일 §2는 1차 발사 이후 무변경 — 리뷰 기반 동일).
- **미회수 종료 (context 129% stop-hook — /half-clone 거부 HARD RULE #11,
  end-of-session update 전환)**: 재발사 리뷰가 진행 중인 상태로 세션 종료.
- **다음 세션 회수 절차** (전례 = 2nd doc §3과 동일):
  1. journal 판독: `~/.claude/projects/-home-cgxr-Documents-Robotics-RoArm-Project/8849c43e-2288-4c38-af98-a6786a8b137d/subagents/workflows/wf_c46dc45e-62d/journal.jsonl`
     의 `{"type":"result"}` row 4건 (렌즈: science/frozen/lessons/ops).
  2. result 부재/불완전 시: 같은 폴더 agent-*.jsonl 전사에서 StructuredOutput
     `tool_use` input 직접 회수 (프롬프트 원문 내 'StructuredOutput' 문자열
     오탐 주의 — 1차 중단분 전사 4개가 이미 폴더에 있음, mtime 18:23 이전
     것은 1차 중단분).
  3. 그래도 부재 시 재발사: `Workflow({scriptPath:
     "~/.claude/projects/.../8849c43e-*/workflows/scripts/d409-design-v1-4lens-review-wf_c46dc45e-62d.js",
     resumeFromRunId: "wf_c46dc45e-62d"})` — 스크립트 verbatim 재사용.
  4. **회수 전 §4 처분/설계 확정/정적 준비 착수 금지.**

### 3.1 회수 완료 기록 (4th 세션, 2026-08-03)

절차 §3의 1→2→3 순서 그대로 집행, 최종 **4렌즈 전문 회수 완료**.

1. **journal 판독**: 이전 세션 journal에 result **1/4** (OPS 렌즈만,
   agent aea781aa; blocker 1·warning 5·confirmation 8). started 8행
   (1차 중단분 4 + resume 4).
2. **전사 추출**: resume 전사 4개를 파서로 스캔 (`tool_use` 블록의
   name=="StructuredOutput"만 매칭 — 프롬프트 문자열 오탐 배제, mtime
   18:23 이전 1차 중단분 4개 제외). 결과: OPS 1건 (journal과 bit-동일
   교차 확인), science/frozen/lessons 3개 전사는 **StructuredOutput 호출
   자체 부재** (이전 세션 종료 18:34에 조사 도중 중단 확정).
3. **resume 재발사**: `resumeFromRunId: wf_c46dc45e-62d` 재발사 성공.
   단, **세션 경계로 캐시 미적중** — 이전 세션 result가 이월되지 않아
   4렌즈 전원 신규 재실행 (스크립트 verbatim, 설계 §2 무변경 확인 후
   발사). 결과 4/4 완료, agent 오류 0 (565,697 tokens, 682.7s).
   → **OPS는 2-pass 병존**: 원본(1차 회수분)과 재실행분 모두 보존,
   **처분은 두 pass findings의 합집합** 사용. 원본 유니크 = OPS1-W1
   (비결정 산출물/결정성 비교 분리 미명시), OPS1-W2 (scope_guards의
   rerun 허용 누락). 재실행 유니크 = OPS2-B1 (A64 권위 — 타 렌즈와
   수렴), OPS2-W3 (outer⊂inner part 중첩).

- **verbatim 보존**: `g0a_d409/design_inputs/review_4lens_wf_c46dc45e-62d/`
  8파일 (이전 journal 사본, OPS 원본 회수분, 재실행 4렌즈 + 전체 출력,
  recovery manifest). manifest sha256
  `b77fc8e34a14cbf50ddc5310149945069b6ec70ca004977176aa64d0e4c1bbcd`.
- **렌즈별 집계**: SCIENCE b3/w4/c9 · FROZEN b2/w3/c10 · LESSONS b2/w5/c9
  · OPS(재실행) b2/w4/c8 · OPS(원본) b1/w5/c8. blocker 총 9건 →
  실질 이슈 4개로 수렴 (§4.1 P1~P4).
- **리뷰 주장 독립 재검증 (메인 세션, 처분 전 수행)** — 전부 리뷰
  주장대로 확인:
  1. τ 반경 결부: `session_20260712_grasp_g0a_d335_target_family_repair.md:61-63`
     verbatim — "derived from radius 17mm and the frozen fixed-jaw face
     offset 8mm" (설계 v1의 '반경 비결부' 판단이 오류임을 원문 확정).
  2. d348 evidence sha256 재계산 = `83b8c7b16181d0f5c545cfbeaa992c8ebfd6
     9e2310dd33bce2a64234a1deaab6` (리뷰 인용 일치·D368 input pin에도 존재).
  3. ULP: table_z/z_center 십진 리터럴 2건 모두 float64 연산 결과와
     1–2 ULP 불일치 재계산 확인 (§4.2 W-SCI3).
  4. 동결 자세 link5 min 4.272736580324082mm의 argmin part =
     **part_029 ∈ link5_fixed 4-mask** (`d409_anchor_gate_measurement.json`
     m3 stored_pose selected_part) → inner-4-mask min = 전신 min =
     4.2727mm. LESSONS-B2의 '미측정' 지적 해소, SCIENCE-B1 모순 확정.
  5. d351 pinch dict 8-predicate 정확 이름 실재
     (`cyl34_top_view_d351_zero_step_closure_geometry.py:2568-2580`).
  6. D368 마스크: outer 16 ⊂ inner 17, 차집합 = {part_035} 단독 (직접
     파싱; link5_fixed = [part_027, part_029, part_030, part_031]).
  7. D371 dual-run 충돌 증거: worker `:855-864` claim FileExistsError →
     return 73, `:808-823` worker-owned 파일 선존재 거부; controller
     `:1819-1820` worker_invocation_count==1, `:2073-2075` pre-run
     inventory prereg-only.
  8. d335 legacy_checks 원문 (`..._d335_...py:329-341`) —
     fixed_jaw_gap_0_to_5mm·contact_at_least_15mm_below_top 게이트 실재.

## 4. 처분 및 설계 확정 (리뷰 후 기록)

기록 세션 = 4th (2026-08-03). blocker 9건은 실질 이슈 4개(P1~P4)로
수렴 — 전건 **수용·설계 정정**. warning 21건(합집합) 전수 처분 §4.2.
확정본 = §2 본문 + 본 §4 δ (충돌 시 §4가 우선). prereg는 §4 반영본만
저작 가능.

### 4.1 Blocker 통합 처분 (9건 → 4 이슈, 전건 수용)

**P1 — A64 권위 d339 → d348 정정** (SCIENCE-B3 / FROZEN-B1 /
LESSONS-B1 / OPS2-B1; §5.0 M1·M3a 실측이 원인 제공. START_HERE 예고
'필수 반영 ①' 이행):

- §2.3 표 정정: A64 양측 조 기하 권위 =
  `g0a_d348/attempt2/d348_callback_topology_volume_evidence.json`
  (sha256 `83b8c7b16181d0f5c545cfbeaa992c8ebfd69e2310dd33bce2a64234a1deaab6`,
  본 세션 재계산 — D368 input pin과 일치). 질의 기하 =
  `rows[].instance.vertices_m` + `rows[].instance.topology_triangles`
  (callback 원본 — §2.9-4의 'callback topology 삼각형' 요구를 유일하게
  충족; d339 triangles는 Qhull 재구성이라 부적격), per-part 무결성 =
  `payload_sha256`.
- d339 cold1 canonical 2파일 = **역사적 cook witness로 강등** (hash pin은
  계보 기록용 유지, 질의 미사용).
- §2.3의 '64+64 bit-동일성 확인' 정적 체크 대체: (i) d348 내부 무결성
  (payload_sha256 재계산 64+64), (ii) D368 마스크의 d348 결박 재확인
  (특히 상이 part_035/048의 carrier 유효성).
- §2.11-2 음성 대조 재정박: vertex stream 1바이트 변조 → **d348
  payload_sha256 FAIL** 확인으로 변경.
- anchor 게이트(§2.9-1/2) 수치 유효 — §5.0 실측이 이미 d348 기하로 수행.

**P2 — τ 도메인 rebase** (SCIENCE-B2 / FROZEN-B2 / LESSONS-W5):

- §2.5의 'τ 도메인 조 측 기하 유래·원통 반경 비결부' 판단 = **오류로
  철회**. d335 doc 원문(:61-63)이 반경 17mm 결부를 명시 (§3.1 재검증 1).
- τ 도메인 확정: **{6,500, 6,750, …, 11,500}µm (21점, step 250 유지)**,
  유도식 = [R−8mm, R−8mm+5mm], R=14.5mm, jaw 상수 8mm =
  FIXED_JAW_FACE_LOCAL_M 동결 리터럴 (d323:38). 격자 = 59×21 =
  **1,239 pose 불변**. 양성 대조 키 (7000, 11000) 잔존 (11,000 ≤ 11,500).
- edge-touch 규칙 신설: admitted 4-연결 성분이 ρ 또는 τ 도메인 가장자리
  셀을 포함하면 해당 영역 ρ_R = **'domain-censored' 플래그** (게이트
  불사용·보고 의무). '전수' 문언은 '선언된 도메인 내 전수'로 한정.
- prereg에 τ 유래 = 'd335 유도식의 반경 치환(전례 상속)' 정직 라벨.

**P3 — (A) 밴드 vs D362 예시 모순 정정** (SCIENCE-B1 / LESSONS-B2;
SCIENCE-W1 병합):

- 본 세션 실측 보강: 동결 자세 link5 min 4.272736580324082mm의 argmin =
  part_029 ∈ 4-mask → **inner-4-mask min = 전신 min = 4.2727mm ∈
  [0.1, 5.0]** (§3.1 재검증 4).
- 처분 = 리뷰 옵션 (b) 채택: 밴드 [0.1, 5.0]mm 유지(영역 서술자),
  §2.7의 정당화 문장("D362 밀어넘김은 A 위반 자세(fixed 4.27mm 이격)에서
  발생 — 재정식화가 원 의도 보존")은 **오류로 철회·삭제**. 대체 문언
  (prereg interpretation_boundary에도 동일 기재): "A∧B는 D362 밀어넘김
  자세(d_fix 4.2727mm — inner-4-mask min 실측)를 **배제하지 않는다**.
  순서 제약은 순수 기하 서술자이며 밀어넘김 스크리닝 능력은 미검증(null)."
- 옵션 (a)(상한을 4.27mm 미만으로 재-pin) 기각 사유: 알려진 실패 예시를
  배제하도록 상한을 사후 선택하는 것은 역방향 gate-shopping — 독립 근거
  없는 스크리닝 능력의 제조. 근접 밴드 {≤1mm, ≤5mm} 진단 기록 유지.
- SCIENCE-W1 병합: 5.0mm 상한 = 구 D330 planar proxy 게이트 상수의
  **재사용**임을 prereg에 공개 (D330 metric = tangent-사영 planar gap
  0~5mm vs 신규 metric = hppfcl 3D min over 4-mask; (7,11)급 자세에서
  ~1.7mm 상이). planar-gap 등가치를 진단 컬럼으로 전 pose 기록.

**P4 — dual-run 계약 신설 (D371 골격과의 충돌 해소)** (OPS1-B1 /
OPS2-B2; 메인 세션 독립 검증 §3.1-7):

1. attempt 폴더 하위 **run1/, run2/** — run별 claim·preclose sentinel·
   summary·stdout/stderr sha256·supervisor·invocation 기록.
   registered_worker_command는 out-dir 인자만 상이.
2. run2는 **run1 preclose sentinel PASS 후에만** 발사. run당 retry 0.
   run2 실패/불일치 = attempt 전체 fail-closed 소모 (D408-R1 문화).
3. prereg 필드: single_run_contract → **determinism_run_contract**
   {worker_invocations_total: 2, per_run_out_dirs: [run1, run2],
   automatic_retries: 0, run2_precondition: run1_preclose_pass,
   byte_compare_manifest: [열거]}.
4. **determinism_manifest**: byte 비교 대상 = canonical evidence JSON +
   영역 지도 CSV의 canonical 직렬화 bytes만 열거. RRD/RBL/screenshot/
   manual/phase/stdout/supervisor = **명시 제외** (OPS1-W1 병합 —
   RecordingId·타임스탬프·PID 비결정).
5. supervisor가 run1/run2 비교 후 **canonical 승격 기록** (bit-exact
   PASS 시 run1 산출을 canonical로 승격).
6. D371 대비 의미 변경 감사 체크 전수 열거: worker_invocation_count
   계열(run별 ==1 ∧ total==2), pre-run inventory 계열(run 폴더 구조
   반영).
7. 관측성 산출물 저작 시점 = 양 run 종료·bit-exact 판정 후 canonical
   evidence로부터 **별도 phase에서 저작** (D408 read-only replay 전례;
   RRD file sink attach-before-first-log D341 준수). 수동 검수 =
   attempt당 1회.

### 4.2 Warning 전수 처분 (합집합 21건 — 병합 후 16항)

| # | 출처 | 처분 |
|---|---|---|
| W-SCI1 | SCIENCE-W1 | P3에 병합 (5.0mm = 상수 재사용 공개 + planar-gap 진단 컬럼). |
| W-SCI2 | SCIENCE-W2 | 7.881 라벨 확정: "n=5 근접 클러스터 표본 최대·3D z-포함·단일 목표·D34×H90-era" + 36.033과 동일 historical-proxy 한정. 근접 클러스터 xy-only 사영(1.7~7.0mm)을 차원 정합 비교치로 병기(진단). |
| W-SCI3 | SCIENCE-W3 | 파생 상수 = 연산 시퀀스로 pin: table_z := float64(0.03288299962878227) − 0.045 = **−0.012117000371217726** (hex −0x1.8d0cc428f5c28p-7); z_center := table_z + 0.025 = **0.012882999628782275** (hex 0x1.a6266f0a3d70cp-7). §2.2의 십진 리터럴 2건 = 1–2 ULP 오기 → 본 값으로 대체 (본 세션 재계산). 혼합 정밀도 명시: x = float32 리터럴, z_center = float64 연산 결과(캐스트 없음, hppfcl Transform float64). |
| W-SCI4 | SCIENCE-W4 | pinch 4-core exact 이름 pin (d351:2568-2580 대조 완료): `moving_and_fixed_inward_normals_opposed` / `jaw_surface_points_on_opposite_xy_sides_of_center` / `cylinder_center_projection_inside_contact_chord` / `q5_decrease_moves_contact_toward_fixed_surface`. 나머지 4 predicate = 진단 기록. 선정 근거(대향 1 + 기하 배치 2 + 닫힘 방향 1) prereg 명기. |
| W-FRZ1 | FROZEN-W1 = OPS2-W3 | interpretation boundary 추가: "part 수준 마스크는 face 수준 inner/outer를 구분하지 못함 (outer 16 ⊂ inner 17, 공유 carrier 16; 차집합 = part_035 단독). inner-17 소속 = inner-face 접촉의 필요조건 판정." §2.11-3 판별 근거 = part_035 단일 차이 문서화 (name-set/sha 비교 결정적 FAIL 유지). 첫 교차 witness의 face 소속 진단 병기. |
| W-FRZ2 | FROZEN-W2 | §5.0 라벨 재지정: 저장 자세 재현 = **M3a**, FK-유도 질의 재현 = **M3b** — 이후 전 문서 M1/M2/M3a/M3b로 참조. START_HERE '선행 실측 3건' → '4건(M1·M2·M3a·M3b) + 처리량' (세션 종료 update 시 정정). §5.0 원문은 append-only 원칙상 무수정. |
| W-FRZ3 | FROZEN-W3 | d371 evidence sha256 본 세션 실측 pin 완료: `e300063d37de44d895da3b96ea6ac95c0d108d217f6f74c458bab218d7bccdf5` — §2.3 표의 '정적 준비에서 pin' 행 완결. |
| W-LES1 | LESSONS-W1 | top−15mm 강등은 유지, 근거만 정정: strict 분류기는 cap 평면 경계 배제만 수행 — 근접-rim 마진 기능 대체 아님. 신규 진단: witness top-마진(z_top − witness_z) 전 pose 기록 + 영역 채점 시 rim-근접 셀 비율 보고 (참고 밴드 <7.5mm, H50 비례; 게이트 아님). |
| W-LES2 | LESSONS-W2 | RRD에 **전 1,239 셀** admission/영역 채색 레이어 log (`/enum/grid`; Float64 권위 = evidence JSON 불변, 대표 subset은 카메라/상세 뷰 전용으로 강등). visual_contract에 rerun CLI exact 버전(rerun-cli 0.34.1 실측) + per-entity required-component 목록 + RBL verify PASS 명시 등재. |
| W-LES3 | LESSONS-W3 | phase 계약 명문화: canonical evidence JSON + verdict sha256 게시 → **이후** RRD/RBL/screenshot/manual (D371 지속 규칙 'measurement-before-presentation'). phase marker로 강제. |
| W-LES4 | LESSONS-W4 | 정적 준비에 **등가성 fixture** 추가: d335 저장 rows((7000,11000) 포함 수 개)에 재정의 admission 수식 replay → 원 legacy_checks와 check-단위 일치(허용오차 pin). pinch 4-core 선정 근거 명기. D342(comparator 기계 결박)/D405(소비자 도출) 정합. |
| W-OPS1 | OPS1-W1 | P4-4/7에 병합 (determinism_manifest + 관측성 별도 phase + 수동 검수 attempt당 1회). |
| W-OPS2 | OPS1-W2 | scope_guards 허용 목록에 명시 추가: rerun-sdk import(save-only RRD 저작) + 절대경로 rerun CLI subprocess(verify/RBL/screenshot 전용; `/home/cgxr/miniconda3/envs/isaaclab/bin/rerun` = rerun-cli 0.34.1 실측). Isaac/kit/physx/warp/cuda/gpu/AppLauncher/cook/USD/HW = 여전히 전부 0. |
| W-OPS3 | OPS1-W3 = OPS2-W2 | manual writer 요건 **전 목록 필드 단위 열거** (D408 실측 승계): controller/writer identity, nonce/HMAC envelope, tuple SHA binding, source/screenshot manifest, 공통 monotonic deadline, 11-field false-publishable 스키마, exclusive-create+no-replace+fsync, 최악 traversal 예산, 런 중 writer 생성/수리 금지. 각 항목 고의 변조 reject fixture. harness 파일 수 정정: 2 → **3** (controller / worker / manual_writer — D408 전례). |
| W-OPS4 | OPS1-W4 = OPS2-W1 | 처리량 벤치마크를 정적 준비 산출물로 영속화 (도구+JSON, 2회 bit-exact). prereg registered budget: **pose당 ≤3,600 질의, run당 ≤4.5M** (이분법 반복×질의 폭 포함). timeout 7,200s 유지 (실측 ≥70× 여유). 세션 doc 서술 수치(8.2µs 등)는 참고로 강등 — prereg는 영속 파일만 인용. |
| W-OPS5 | OPS1-W5 = OPS2-W4 | 접두 개명: cylr29h50_ → **cyld29h50_** (지름 인코딩 규약 유지, 'r29=반경 29mm' 오독 제거). harness 3파일명: `sim_scripts/cyld29h50_d409_zero_step_dual_jaw_contact_region_enumeration_{controller,worker,manual_writer}.py`. |

**anchor 게이트 확정 pin** (START_HERE 예고 '필수 반영 ②' + OPS2
confirmation의 ANY-reject 의미론): 4채널 {link5 FK pos err, gripper FK
pos err, link5 dist delta, gripper dist delta} 중 **어느 하나라도
> 0.0005mm → reject** (ANY-reject). 실측 여유: 리터럴 통과측 최대
0.000137mm (3.6×), pi/2 거부측 link5 FK 0.001261mm·link5 dist
0.001191mm (2.4×). **pi/2의 gripper dist delta 0.0001777mm는 임계
이하 — 거리 채널의 판별은 link5 단독**임을 prereg에 명기. rot err는
진단 기록.

### 4.3 설계 확정본 v2 (δ-목록; §2 + 본 목록 = 확정 계약)

1. §2.2: 파생 상수 연산 시퀀스 pin (W-SCI3 값). τ 도메인 rebase 추가
   서술 (P2).
2. §2.3: A64 권위 = d348 (P1); d371 evidence sha pin (W-FRZ3);
   outer⊂inner 명기 (W-FRZ1).
3. §2.5: τ ∈ {6,500..11,500}µm (P2); domain-censored/edge-touch 규칙.
4. §2.7: 정당화 문장 삭제·대체 (P3).
5. §2.8: pinch 4-core exact 이름 (W-SCI4); top-마진/rim-근접 진단
   (W-LES1); 7.881/36.033 라벨 (W-SCI2); planar-gap 진단 컬럼 (P3).
6. §2.9: anchor 게이트 ANY-reject 4채널 0.0005mm pin; 결정성 =
   determinism_manifest 열거 파일만 (P4).
7. §2.10: harness 3파일 (W-OPS3); dual-run 계약 (P4); scope_guards
   rerun 허용 추가 (W-OPS2); phase 순서 measurement-before-presentation
   (W-LES3); registered budget (W-OPS4); 접두 cyld29h50_ (W-OPS5).
8. §2.11: 음성 대조 2 재정박(d348 payload_sha256), 3 판별근거 문서화;
   등가성 fixture 신설 (W-LES4).
9. §2.12: 전 셀 RRD 레이어 + CLI 버전/컴포넌트/RBL verify 명시
   (W-LES2); manual writer 전 요건 열거 (W-OPS3).
10. §2.13: run1/run2 폴더 구조 (P4); 벤치마크 영속화 (W-OPS4).
11. §2.14: A∧B의 D362 비배제 + part/face 한계 문언 추가 (P3, W-FRZ1).

### 4.4 확정 선언

- **설계 확정 (v2)** — 2026-08-03 4th 세션. 신규 변수 수 불변: 1
  (실물 원통 기하). τ rebase는 d335 유도식의 반경 치환(파생)이며 신규
  변수 아님 (원문 근거 §3.1-1).
- 다음 = §5 정적 준비 (음성 대조 2계층, D371 골격 + 본 §4 δ). runtime
  attempt1은 tuple SHA 인용 **사용자 별도 승인** 후에만.

## 5. 정적 준비 (설계 확정 후 기록)

### 5.0 선행 실측 (리뷰와 병행 수행 — 설계 본문 §2 무변경 유지)

리뷰 결과와 무관하게 필요한 §2.9/§2.11/§2.13 실측을 선행 완료. 산출 =
`g0a_d409/design_inputs/d409_anchor_gate_measurement.json` (sha256
`8cc6116616d61615ee144db9265ca6fbbc2183ca07342594339162d4dab96724`, 도구
동봉, **독립 2회 실행 bit-exact PASS**; Isaac 0 / physics step 0).

- **M1 — A64 소스 동일성 (설계 §2.3 fail-stop 조항 발동)**: d339 cold1
  canonical vs d348 attempt2 instance, per-part vertex **집합** 비교
  (순서 무관). 결과: gripper 59/64, link5 56/64만 집합 동일 — **13개
  part 상이**. 상이 패턴 전부 동일 방향: d348 ⊂ d339 (d348 정점 전부가
  d339에 존재, d339에만 여분 정점 1~2개, d348 집합에서 최대 5.53mm
  이탈). 상이 part 중 gripper part_035·part_048은 **D368 inner mask
  carrier**. 또한 배열 bit-동일은 0/64 (정점 순서 permutation + 삼각분할
  상이: d339 triangles=Qhull 재구성, d348 topology_triangles=callback
  원본).
- **M3 — 저장 자세 재현이 권위를 판정**: d348 instance 기하 + D349 저장
  라이브 자세 → live_topology 기준값 재현 **delta 0.000000000mm** (link5
  4.272736580mm / gripper 11.340262326mm; D371과 동일 결과 독립 재현).
  즉 **라이브 A64 = d348 계보 실증** → 설계 v1 §2.3의 "d339 canonical
  권위" 결정은 §4 처분에서 **d348 권위로 정정 필요** (d339는 역사적
  cook witness로 강등). 처분 확정은 리뷰 회수 후.
- **M2 — FK 자세 재현 (상수 계열 판별)**: URDF-리터럴 FK가 D349 동결
  float32 관절로 라이브 자세 재현: link5 pos 0.000136mm / rot
  0.000056°, gripper pos 0.000137mm / rot 0.000078°. pi/2-심볼 변형
  (roarm_kinematics 계열): link5 0.001261mm — **9배 이탈, 판별 가능**.
  부수 발견: pi/2 계열 TCP vs D349 `commanded_tcp` = 0.000017mm ≈ 0 —
  **D349의 commanded_tcp가 pi/2 체인 산출임을 실증** (인계값
  0.0013mm ≈ 재계산 0.0012170mm의 기원 해명: pi/2-체인 FK vs 라이브의
  차이였음). 설계 §2.4의 URDF-리터럴 단일 계열 결정을 실측이 지지.
- **M3 — FK-유도 자세 질의 재현**: URDF-리터럴 FK 자세로 d348 기하 질의
  → live_topology 기준 대비 delta link5 0.000118mm / gripper
  0.000120mm. pi/2 변형은 link5 0.001191mm. → anchor 게이트 허용오차
  pin 후보: **FK pose gate pos ≤0.0005mm ∧ 거리 delta ≤0.0005mm**
  (리터럴 통과 / pi/2 거부 — 음성 대조 §2.11-4 판별력 실측 확보).
- **처리량 벤치마크 (§2.13 예산 검증)**: BVH 128개 빌드 3.1ms; distance
  질의 10,240회 실측 **8.2µs/질의** → 2.7M 질의 ≈ 0.4분, 5.4M ≈ 0.7분.
  timeout 7,200s는 압도적 여유 (유지).

### 5.1 정적 준비 착수 (4th 세션, 설계 확정 v2 이후)

- **S1/S2/S3 실측 산출물 완성** —
  `g0a_d409/design_inputs/d409_static_prep_s1s2s3.json` (artifact sha256
  `f2aaadd13e6822ceebd6a5d565010c0f45c201d08d35b830d582e5ac36dfd63d`,
  도구 sha256 `e7b541ef5d25039314ff66568f4e114172a19348ed45204fd35a9192148a0f90`,
  독립 2회 실행 payload **bit-exact PASS**
  `43a46e0552e7c23de936b6da59eeb9771b675651c0edbb301d0ed8e5d575b124`;
  Isaac 0 / physics 0):
  - **S1 (P1 의무)**: d348 128 part(64+64) D409-canonical per-part
    geometry hash 전수 pin (runtime 재검증 기준), 해시 유일성 PASS,
    stored payload/witness sha verbatim 계보 기록.
  - **S2 (P1 의무)**: D368 마스크 ↔ d348 결박 전항 PASS — link5_fixed
    4 일치, inner 17/outer 16 전원 d348 row 실재, outer = inner −
    {part_035} 재확정, part_035/048 gripper row 실재.
  - **S3 (W-OPS4 의무)**: 12,800 질의 실측 run1 8.71µs / run2 7.54µs
    → 4.5M ≈ 34~39s ≪ 7,200s (예산 검사 양 run TRUE). 거리 집합은
    결정성 payload에 포함(sha 비교), 타이밍은 설계상 결정성 제외 명시.
    prereg 예산은 이 파일 인용 (세션 doc 서술 수치 참고 강등 — W-OPS4).
- **harness 3파일 저작 백그라운드 위임** (D407 전례 wf_0d5ade26-6bc와
  동일 패턴): run `wf_d6a61f26-880` — worker/manual_writer 병렬 저작 →
  controller(실물 인터페이스 정합) → read-only 일관성 검토. 스펙 = 본
  doc §2+§4 (충돌 시 §4 우선), 파일명 접두 **cyld29h50_** (W-OPS5).
  **산출물 가정 금지** — 회수 후 적대 리뷰·수리 전에는 채택 아님.

## 6. Tuple (정지 지점 — 미도달)

## 7. 세션 종료 상태 / 다음 단계 (2026-08-03 3rd 종료 시)

- Session progress rule 충족: 이 세션은 실패 가능한 오프라인 실측 3건을
  실행했고(§5.0 M1~M3 + 처리량), **M1이 실제로 설계 v1의 fail-stop
  조항을 발동시켰다** (d339↔d348 13 part 상이 발견 → 권위 정정 필요).
- 완료: 스윕 회수(2nd doc §4), FK 스칼라 재도출·파일화, 설계 v1 저작,
  선행 실측 3건, 리뷰 재발사.
- 미완: 4-lens 리뷰 회수(§3 절차) → §4 처분(핵심: A64 권위 d339→d348
  정정, ε/게이트 pin 실측 반영) → 설계 확정 → 정적 준비(음성 대조) →
  attestation/tuple 작성 후 정지. **attempt1은 tuple SHA 인용 사용자
  별도 승인 필수.**
- 과학 상태 불변: D407 FAIL-STOP, `g0a_pass=false`. D399 사용 금지.
