# Session 2026-08-03 (2nd) — D409 설계 착수: 실물 원통 zero-step 양측 접촉영역 전수 열거

이번 case의 신규 변수: [실물 원통 기하 — 사용자 실측 확인 D29×H50mm] (1개)

## 0. 승인 범위와 Session progress rule

- 사용자 지시(2026-08-03): **"설계 착수"** + P0 실측 보고. D407 전례
  (EXPERIMENT_LEDGER 445행) 준용: 설계 착수 승인 = 설계 + 정적 준비 +
  attestation/tuple 작성까지. **실제 attempt1 실행은 tuple SHA 인용 별도
  명시 승인 필요.**
- Session progress rule: 이 세션은 설계 세션. 실패 가능 실험은 설계 확정
  후 정적 단계(음성 대조 포함)와 attempt1에서 수행한다. 설계 없이 실행
  불가한 표준 3단계 구조(AGENTS.md D400-P2 계보)가 사유다.
- 과학 상태 불변: D407 FAIL-STOP, `g0a_pass=false`.

## 1. P0 실측 provenance (사용자 보고, 2026-08-03 — HARD RULE #18 권위)

- 질량: **24.83g 재확인** — 사용자 "직접 확인했고" (2026-08-03, 이 세션).
  2026-07-31 최초 보고와 동일값. 측정 기기 모델/단위 분해능은 미보고 —
  후속 보고 시 추가 기록.
- 치수: **지름·높이 = 명목과 일치** — 사용자 "나머지 지름 높이 같아"
  (2026-08-03). 즉 실측 **D=29mm, H=50mm** 확정, 반지름 r=14.5mm.
  측정 도구(캘리퍼 여부) 미보고 — 후속 보고 시 추가 기록.
- 효력: D409 신규 변수(실물 원통 기하)의 값이 이로써 확정된다.
  D29×H50은 이제 "명목"이 아니라 "사용자 실측 확인"으로 승격.
  질량 24.83g는 D409에서 분석 상수(전도 임계)로만 사용, 마찰 미도입.
- 파생 상수(계산): W = 0.02483×9.81 = 0.2436N.
  전도 임계 F_tip = W·r/h: h=25mm→0.141N(≈14.4g중),
  h=40mm→0.088N(≈9.0g중), h=50mm→0.071N(≈7.2g중).
  균일 밀도 가정 전도각 θ = atan(r/(H/2)) = atan(14.5/25) ≈ **30.1°**.

## 2. 기울임/전도힘 손측정 프로토콜 (사용자 질문 답변 — 기록만)

목적: 라벨 사다리 2단(PhysX)이 "상대 스크린 전용"인 이유는 (a) 실물-테이블
마찰 미실측, (b) 전도 임계(0.07~0.14N)가 sim 접촉등록 문턱(0.1N)과 같은
자릿수라서다. 이 손측정은 로봇·저울 정밀장비 없이 실물의 "넘어짐 특성"을
관찰해 ① 균일 밀도(CoM 높이 25mm) 가정 검증, ② 마찰계수 μ의 범위 확보를
하는 파일럿이다. **D409(기하 전용)는 이 결과를 쓰지 않는다** — 2단 PhysX
스크린 case의 준비물이며, Variable Ladder상 sim 반영은 별도 case 변수다.

### 방법 A — 기울임(전도각) 시험 (준비물: 평평한 판 + 스마트폰 각도계 앱)

1. 판(책/클립보드) 위에 원통을 세운다 (축 수직).
2. 판 한쪽을 아주 천천히 들어올린다.
3. 원통이 **넘어지는 순간**의 판 기울기 각도를 각도계 앱으로 읽는다.
4. 같은 자리에서 5회 반복, 각도 5개 기록.
- 예측: 균일 밀도면 **약 30°**에서 넘어진다. 실측이 크게 다르면(예: 25°
  이하/35° 이상) CoM이 중앙이 아니라는 뜻 → sim 원통 질량분포 rebase 근거.
- 넘어지기 전에 **미끄러지면** 그 각도도 기록 — 미끄러짐 각 φ에서
  μ = tan(φ)가 바로 나온다 (이 경우가 오히려 정보가 더 많음).

### 방법 B — 수평 밀기 slide-vs-tip 시험 (준비물: 손가락/얇은 카드 + 자)

1. 실제 작업 테이블 위에 원통을 세운다.
2. 정해진 높이에서 옆면을 수평으로 아주 천천히 민다.
3. 관찰만 한다: **기울며 넘어짐(tip)** vs **선 채로 미끄러짐(slide)**.
4. 높이 3곳 × 3회: 꼭대기 근처 h≈50mm / 중간 h≈40mm / 아래 h≈25mm.
- 원리: 넘어짐 필요 힘 = W·r/h (높을수록 쉬움), 미끄러짐 필요 힘 = μ·W
  (높이 무관). 낮은 문턱이 먼저 일어난다. 따라서 **힘을 잴 필요 없이**
  관찰만으로 μ 범위가 나온다:
  - h에서 tip → μ > r/h (예: h=50 tip → μ > 0.29)
  - h에서 slide → μ < r/h (예: h=25 slide → μ < 0.58)
  - 예: "h=50 tip + h=25 slide"면 **0.29 < μ < 0.58**로 좁혀진다.
- (선택) 힘 수치까지 원하면 0.1g 분해능 저울/소형 스프링 저울로 밀거나
  당기며 순간값(g중)을 읽는다. 7~14g중 수준의 미세 힘이라 분해능이
  안 되면 생략 — slide/tip 관찰만으로 충분하다.

기록 양식(사용자 보고용): 방법 A = 회차별 넘어짐 각(°, 미끄러짐 시 표기).
방법 B = (높이 mm, 회차, tip/slide) 9행.

## 3. 증거 스윕 상태 (미회수 — 다음 세션 필수 회수)

D409 설계 입력 4-reader 증거 스윕(FK/D349 동결 자세, A64 witness·마스크
실측, D330/D362/q5 제약·채점 원천, D379 rebase·D371 오프라인 전례·D341
관측성)을 발사했으나 **context 89% stop-hook으로 결과 미회수 종료**
(HARD RULE #11에 따라 /half-clone 거부, end-of-session update로 전환).

- run `wf_aaa9aa61-d2a`, journal:
  `~/.claude/projects/-home-cgxr-Documents-Robotics-RoArm-Project/d2632654-c933-459c-a2ef-41631607f21d/subagents/workflows/wf_aaa9aa61-d2a/journal.jsonl`
- 워크플로는 세션 종료 후에도 완주하며 journal에 `{"type":"result",...}`
  row로 4-reader 결과가 남는다. **다음 세션: journal의 result row를 먼저
  읽고, 부재/불완전 시에만 agent-*.jsonl 전사 파싱 또는 재실행**
  (오늘 wf_c2a1c870-60e 회수 전례와 동일 절차).
- 스윕 프롬프트 원문(재실행 시 verbatim 사용):
  `~/.claude/projects/.../d2632654-*/workflows/scripts/d409-design-evidence-sweep-wf_aaa9aa61-d2a.js`
- 회수 전 설계 v1 저작 금지 — 특히 인계-전용 수치(FK 0.0013mm, inner
  mask 4/17/16)는 스윕 실측으로 대체하기 전 인용 금지 (1st 세션 doc
  §6.5 warning ④).
- **후속(같은 날 세션 말미): 스윕 완주 확정** — 4/4 agent 정상 완료
  (agents_error 0), journal에 result row 4건 잔존. 부분 수신한 핵심:
  ① FK "0.0013mm" 인계값의 repo 산출물 **부재 확정**; 저장 스칼라
  재계산치는 **0.0012170mm**(commanded_tcp vs actual_tcp, ~7% 차) —
  설계에서는 인계값 대신 재도출 실측을 인용할 것. ② D349 동결 자세
  권위 파일 = `g0a_d349/d349_frozen_target_distance_measurement.json`
  (`target_state_guard.commanded_joint_rad_float32` 등; **float64 관절
  벡터는 미저장** — 재도출 결정성 검증 필요). ③ 오프라인 FK 체인
  (`roarm_kinematics.py` `_CHAIN`)에는 gripper 조인트가 없어 가동 조
  frame은 link5_to_gripper_link 변환 확장 필요(d332:1264 전례).
  전체 4-reader 결과는 journal 전문 판독으로 회수(발췌 인용 금지).

## 4. 증거 스윕 회수 결과 (후속 세션 2026-08-03, journal 전문 판독)

### 4.0 회수 메타데이터

- journal 실측: §3 경로에 8 row (`started` 4 + `result` 4), agent 4/4 정상
  결과 잔존, error 0. **result row 4건 전문 판독 완료** (발췌 아님 — 4건
  각각의 전체 findings를 그대로 읽음).
- verbatim 보존 (journal은 repo 밖 세션 폴더라 소실 위험 → repo 내 복사):
  `claudedocs/runtime_logs/grasp_track/g0a_d409/design_inputs/evidence_sweep_wf_aaa9aa61-d2a/`
  - `journal_verbatim_copy.jsonl` sha256 `6429d7f9…703e`
  - `sweep_result_a4a26e2e7924c4350.json` `ab943586…03e4` (reader: D379/D371/D341/env)
  - `sweep_result_a22577108e9341d23.json` `6da5f43b…4438` (reader: D330/D362/q5 기구학)
  - `sweep_result_a2f6db548ca5b291c.json` `95536ef1…ab1c0` (reader: A64
    witness/마스크/분류기)
  - `sweep_result_abd53643b8d69a55b.json` `3c3af0c3…aed5` (reader: D349 자세/FK)
  - 전체 목록·해시 = 같은 폴더 `d409_design_inputs_manifest.json`.

### 4.1 Reader 1 — D379 rebase / D371 오프라인 전례 / D341 관측성 / env

- **D379 원문 확인** (DECISIONS.md:21987, 22000-22004, 22036-22043):
  ① primitive cylinder 우선 — 실측 비원통 특징의 접촉 중요성 증거 전
  convex decomposition 금지. ② D362(D34×H90/0.72kg) 물리증거를 실물
  D29×H50에 전이 금지. ③ rebase 선행감사(17 failed P34 parts)는 D380이
  완료. → D409 원통은 analytic `hppfcl.Cylinder`로 모델링.
- **D34×H90 하드코드 위치 실측**: d362 worker(:95-97, 0.017/0.090/0.72),
  d371(:1136), d372(:1410), d332/d360/d407 worker — 전부 동결 계약,
  수정 금지. D409는 신규 스크립트 + `g0a_d409/` 신규 폴더로만 자체 상수
  도입 (Variable Ladder forward-only).
- **D371 실행 구조 실측**: isaaclab env python 고정, hpp-fcl 2.4.4 /
  numpy 1.26.0 / psutil 5.9.8 / scipy 1.15.3 / trimesh 4.5.1 / rerun-sdk
  0.34.1. 378 질의 = 5 family part 합(128+128+64+23+35) × distance+collision
  각 1회, part당 밀리초급 — D409 규모(격자×128 part×arc)도 CPU 오프라인
  감당 가능. harness 골격(prereg-first, 입력 sha pin, scope_guards 전 금지
  항목 0 명시, 단일 worker retry 0, 측정 verdict의 presentation-이전 커밋,
  2계층 음성 대조: prepare 5 + audit_registered 4) 승계.
- **D341 Rerun 계약**: 기하 verdict → replayable RRD 의무. entity 분리
  (source/instance/prototype/candidate), sink 선-attach, footer verify,
  blueprint+RBL, headless screenshot, 육안검수. 단 **D371의 관측성 구현
  자체는 FAIL 전례**(rerun CLI 부재 run_exception; root validation/manual
  JSON 미생성) — 실행 세부는 D404~D408 수리 패턴(절대경로 rerun CLI,
  1920×1080 정확, ppp 2.0 등)을 따라야 함.
- **env 실측**: D326 pin 정확 준수(numpy 1.26.0, psutil 5.9.8) + 위 버전
  전부 dist-info 실재. D409 추가 설치 불필요.

### 4.2 Reader 2 — D330 36.033mm / D362 시퀀스 / q5 닫힘 기구학

- **36.033mm의 정확한 정의**: '실행 후 실제 TCP vs 목표 TCP' 유클리드
  거리의 10-env 평균 (D330; commanded FK 오차는 0.404mm로 별개). 단일
  고정 목표(D34×H90 sim)이며 reachable family 격자 평균이 아님. jitter
  ±0.02rad + null space로 env들은 동일-조건 replica가 아님(D331).
- **분포 실측 (매우 중요)**: 10개 값 = {1.884, 3.044, 6.004, 7.686,
  7.881, 27.233, 70.303, 75.775, 79.989, 80.530}mm — **이봉(bimodal)**.
  근접 군집 5개(1.9~7.9mm) + 중간 1(27.2) + stall 군집 4(70~81mm).
  평균 36.033 근방 ±9mm에 관측치 0 → **'빈 영역의 평균'이라 스칼라
  단독 채점은 통계적으로 방어 불가**. 선택지: 근접군집 상한(~8mm) /
  empirical 분위수 / 참조 스케일로만 사용.
- **D362 밀어넘김 시퀀스**: moving jaw 접촉 확정 step 32(2.869N) →
  원통 XY 이동 41/42 → fixed jaw 접촉 45/46은 '물체가 밀려와서' 생긴
  것. zero-step 기하에서는 같은 시퀀스가 재현될 수 없음을 설계에 명시.
- **D362 q5 300 step은 명령 램프가 아니라 PD 동역학 응답** — D409의
  zero-step sweep 전례는 D351/D354의 직접 state-write 격자+이분법.
  '시간 step 축'과 'q5 각도 축' 혼용 금지.
- **q5 기구학 확정**: q5 = URDF `link5_to_gripper_link`(parent=link5,
  child=gripper_link, origin (0, 0.018821, 0.052035), rpy (-1.5708,
  -1.5708, 0), axis z, limits [0, 1.571]). 월드 축·원점 합성식은 D351
  검증본(런타임 대비 ≤1µm). 격자 인접각 사이 누락 접촉은 현 길이 상계
  `2*Rmax*sin(|Δq|/2)`로 배제 인증(D354 실사용). D354 첫 교차각 인증
  전례: q_clear=1.0269782543182373 / q_overlap=1.0269775390625,
  bracket 폭 1e-6rad.
- **[차이 보고 — 설계 핵심 수정] 'fixed-jaw 먼저'는 문자 그대로 구현
  불가**: link5(고정 조)는 q5의 부모라 sweep 내내 불동 — D354 sweep CSV
  전 33 anchor에서 `raw_link5_mm` = 4.2726455336106985mm 상수 실증.
  zero-step에서 고정 조의 '첫 교차 q5 각도'는 정의 불능. **재정식화 =
  (A) 자세 수준 0차 조건: q5=OPEN에서 link5–원통 signed distance ≤ ε
  AND (B) moving-jaw 첫 교차각 q5* ∈ (0, OPEN) 존재 + 그 지점 유효성**.
  D362 밀어넘김은 (A) 위반 자세(fixed jaw 4.27mm 이격)에서 닫았기 때문 —
  재정식화가 원 의도를 보존. ε 후보: D330 게이트 0~5mm / D332 BORDERLINE
  0.1mm / D339·D349 CLEAR_GATE 0.1mm → 설계 파라미터로 명시 pin 필요.
- **접촉 판정 전례**: hppfcl 권위 — BVHModelOBBRSS + analytic Cylinder,
  DistanceRequest(True,1e-9,1e-9), gjk 1e-9/1000, 경계 0.1mm →
  OVERLAP/CLEAR/BORDERLINE(D332). D351 endpoint 유효성 계약 + interval
  bound 승계. **기하 0.1mm(거리)와 D362 0.1N(힘)은 별개 임계값** —
  혼동 금지.

### 4.3 Reader 3 — A64 witness / D368 마스크 / 분류기 / pinch predicate

- **A64 vertex 배열 실재 위치**: 1.14MB summary가 아니라
  `g0a_d339/collision_asset/attempt2/d339_{gripper_link,link5}_cold1_canonical_geometry.json`
  (`parts` 64개: vertices_m/triangles/geometry_sha256 등; cold1==cold2).
  callback witness 원본(`…_cold1_callback_witness.json`)에는 polygon별
  **반평면 방정식(plane [nx,ny,nz,d])** 이 있어 half-space margin 계산에
  즉시 재사용 가능. 규모: gripper 549v/842t, link5 566v/876t.
  **좌표는 body-local 확정** (prim→body 항등 1.11e-16 + bounds 대조).
  D339 자체 verdict는 FAIL_STOP이지만 asset build+cook witness는 PASS이고
  D368이 pin — prereg에 이 관계 명시 필요.
- **[차이 보고 — 마스크 매핑 정정] 4/17/16의 단일 소스는 D368 evidence
  한 파일**: `g0a_d368/d368_semantic_allocation_evidence.json`의
  `patch_allocation` — link5_fixed = 4 parts [part_027, part_029,
  part_030, part_031] (live face 12), gripper_inner = 17 parts (live
  face 40), gripper_outer = **16 parts = 음성 대조군**(inner 마스크
  아님). 'D350→4, D354→17, D368→16' 해석은 오류 — D350/D354는 raw 면
  patch 정의만 제공, part 마스크 계산·저장은 D368 1회. 채점용 inner
  계열은 4+17, 16은 negative control 전용.
- **barrel/cap 분류기**: `_feature_from_cylinder_witness` (d351:910-939)
  — strict z 부등호만, tolerance 없음(D354 durable rule: 사후 tolerance
  도입 금지). 순수 numpy로 이식 가능하나 **R/H 상수 rebase 필수**
  (0.017/0.090 → 실물 0.0145/0.050) — d332 상수 그대로 import 시 경계
  ±45mm→±25mm 오류.
- **pinch_facing_geometry (antipodal 지표)**: 본체는 d351
  `_bind_moving_surface`(:2418-2918) — 부호/순서 predicate 전용
  (fixed·moving normal 마주봄, chord 사영 0<t<1, 중심 기준 반대편 등
  13 check). 입력이 전부 명시적 world 점/벡터라 FK+hppfcl witness 대체
  주입 가능. 단 **D354는 pinch pass=false FAIL_STOP 전례** — 수식·계약
  구조의 재사용이지 PASS 결과 재사용 아님을 prereg에 명시.

### 4.4 Reader 4 — D349 동결 자세 / 순수 FK / 인계 수치 처분 / 격자 전례

- **D349 동결 자세 권위 확정**:
  `g0a_d349/d349_frozen_target_distance_measurement.json` —
  `target_state_guard.commanded_joint_rad_float32` == actual =
  [0.03750238195061684, 0.542945146560669, 1.9687392711639404,
  0.18299327790737152, 0.0, 1.5413000583648682]. target_contract =
  (radial 7.0mm, tangent 11.0mm, q5 1.5413, seed 33201). 라이브 body
  자세(link5/gripper_link pos+quat)와 TCP(commanded/actual)도 동일
  파일에 저장, env origin=0. **float64 관절 벡터는 어느 JSON에도 미저장**
  (float32만) — float64 필요 시 d335 재실행이 유일 경로, 결정성 1회
  검증 의무.
- **순수 FK 원천 3층**: ① URDF `local_assets/roarm_m3/urdf/roarm_m3.urdf`
  :172-238 (조인트 체인 리터럴), ② `sim_scripts/roarm_kinematics.py`
  `_CHAIN`(:18-26 — **gripper 조인트 없음, link5/TCP 종단**; 소프트
  리밋은 v6-유래로 URDF와 다름), ③ d323 `_fk_link5_runtime`/
  `_fk_runtime_tcp`(TCP_LOCAL_OFFSET [0,0,0.115428]). 가동 조 확장
  전례 = d332:1264 `Tmat((0,0.018821,0.052035),(-1.5708,-1.5708,0)) @
  Trot_z(q5)` vs cube3cm:119 `-math.pi/2` — **상수 계열 혼용 금지,
  하나로 pin**.
- **[차이 보고] FK '0.0013mm' repo 산출물 부재 확정** (전수 grep — 세션
  doc 인계 파일화본과 START_HERE 금지 조항에만 등장). 저장 스칼라
  재계산치 = **0.0012170mm** (commanded_tcp vs actual_tcp; deltas
  [1.749e-7, 1.199e-6, 1.181e-7]m). 인계값과 ~7% 차 — float32 입력 FK,
  link5-frame 비교, 반올림 등 다른 산출 경로였을 가능성.
- **[차이 보고] 오프라인 FK 상수 ↔ URDF 리터럴 µm급 불일치**: _CHAIN
  0.05196 vs URDF 0.051959 (1µm), pi/2 심볼 vs 1.5708 리터럴
  (3.67µrad ≈ 0.9µm@0.24m). 이 합이 0.0012mm 잔차 규모와 일치.
  → **'bit-exact 재현' 주장 불가, '≤X µm 재현'으로 서술**; 상수 집합
  하나로 통일 pin.
- **격자 전례 (d335/d336/d337)**: 축은 관절이 아니라 TCP 오프셋 공간
  (radial 0~17,000µm × tangent 9,000~14,000µm, µm/nm 정수 키, 유니크
  2,629→2,632, d337 passing 2,560, selected=(7.0,11.0)). 게이트 집합
  (ik_converged, commanded TCP ≤5mm, jaw tangent ≤15°, fixed_jaw gap
  0~5mm, 무관입, top −15mm, anti-retreat, raw clearance ≥0.1mm) 재사용
  가능. **단 domain 상한 radial 17,000µm과 anti-retreat '17mm−r'은 구
  반경(17mm) 결부 상수 — 실물 14.5mm로 rebase 필수, 복사 금지**.
- **[차이 보고] d335~d337 격자는 Isaac-의존** (물리 0 step이지만 라이브
  state write+read). 완전 오프라인 전례는 D371뿐 — 저장 자세를 hppfcl
  Transform3f로 재구성, A 후보 오프라인 재계산이 D349 라이브 거리와
  delta 0.0mm bit-exact (link5 4.272736580324082mm / gripper_link
  11.340262326338637mm). → D409 신뢰성 anchor: 동결 (7,11,1.5413)
  자세에서 FK-유도 자세 질의가 이 저장 거리를 재현하는 재현 게이트를
  첫 체크로. 단 FK 상수 불일치 때문에 0.0mm bit-exact 기대 불가 —
  허용오차를 실측으로 pin.

### 4.5 스윕 발 차이 보고 종합 (설계 반영 의무 목록)

| # | 인계/기존 표현 | 실측 정정 | 설계 반영 |
|---|---|---|---|
| A | "D371 = Isaac 실행 0 전례" | D371 cook worker는 headless Isaac 1회(cook 전용). Isaac-0였던 건 A-family 경로(동결 witness read만) | D409 prereg에 `app_launcher=0`, `physx_cook_callbacks=0` 명시, A-family 경로만 준용 |
| B | "사용자 실측 D29×H50" (스윕 시점엔 명목) | 스윕 시점 repo엔 명목뿐이었으나 **본 세션 §1의 사용자 P0 실측 보고로 해소** — 실측 D29×H50 확정 (HARD RULE #18) | prereg에 실측값+provenance(§1) pin |
| C | "A64 = g0a_d339 witness" | D371이 실제 로드한 A64 part 기하 = `g0a_d348/attempt2/d348_callback_topology_volume_evidence.json` 계열. d339 witness와 역할·파일 상이(모순은 아님) | 권위 소스 1개 pin + d339↔d348 64+64 bit-동일성 정적 확인 항목 추가 |
| D | "margin vs 36.033mm" | 실행 오차(기하 margin 아님)·단일 목표·비-replica 평균·이봉 분포·D34×H90-era | '역사적 실행 오차 proxy'로 라벨, 이봉 구조 명시 채점 |
| E | "fixed-jaw 먼저 q5 비교" | 고정 조는 q5 불변 — 문자 그대로 정의 불능 | (A) 자세 수준 ε-근접 AND (B) moving-jaw q5* 존재로 재정식화 |
| F | "D350/D354/D368 = 4/17/16" | 세 마스크 전부 D368 한 파일 계산·저장; 16은 outer 음성 대조군 | 마스크 출처 단일 파일로 기재, 16은 negative control 전용 |
| G | "FK 0.0013mm" | repo 부재 확정; 저장 스칼라 재계산 0.0012170mm (~7% 차) | 재도출 파일 인용 (§4.6), 0.0013 인용 금지 유지 |

### 4.6 인계-전용 수치 대체 완료 (1st doc §6.5 warning ④ 처분)

- **FK 0.0013mm → 대체 완료**: 재도출 실측
  **0.001216972820130102mm** — 산출·게시 =
  `g0a_d409/design_inputs/d409_fk_tcp_scalar_rederivation.json`
  (sha256 `c0b13007d36de91b6aa8f1190d6d14f8e45e39564325292a5b85d51a0655d5aa`),
  도구 = 같은 폴더 `d409_fk_tcp_scalar_rederivation_tool.py`
  (sha256 `f0251ea61268b506…`), isaaclab env python 3.11.14, **독립 2회
  실행 canonical payload bit-exact PASS** (결정성 검증 의무 이행).
  성격: 저장 리터럴 재계산이며 FK 재실행 아님 — FK 재실행 재현 오차는
  D409 harness가 자체 결정성 체크와 함께 별도 산출.
- **inner mask 4/17/16 → 대체 완료**: 스윕 실측 (§4.3 — D368 evidence
  단일 파일, 16은 outer 음성 대조).
- **문헌 41건 / 17/17 MATCH → 미대체 유지**: D409 설계 입력이 아니므로
  재도출 불요 — 계속 인용 금지 (인용 필요 시점에 재검증).

## 5. 다음 단계 / 승인 경계 (구 §4 — 번호만 이동, 내용 불변)

1. (다음 세션) 스윕 회수 → §4로 기록 → 설계 v1 저작 → 4-lens 적대적
   리뷰 → 처분 반영 설계 확정 → 정적 준비/harness → attestation/tuple
   작성 후 정지. **attempt1 실행은 tuple SHA 인용 사용자 별도 승인.**
2. (사용자, 병행) §2 손측정 방법 A/B 수행 후 결과 보고 — 기록만.
3. 과학 상태 불변: D407 FAIL-STOP, `g0a_pass=false`. D399 사용 금지
   (D398-F1 예약). 산출 폴더는
   `claudedocs/runtime_logs/grasp_track/g0a_d409/`만 사용.
