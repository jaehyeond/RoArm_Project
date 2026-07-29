# D407 설계 착수 — 증거 스윕 + 설계 초안 (실행 없음, 리뷰 회수 대기)

Date: 2026-07-28 밤 KST (D406 full PASS 직후 세션). 이번 case의 신규 변수
(후보, prereg 미저작): **1** —
`gripper_link_collision_representation_a64_to_sdf_res256_v1`의 물리 적용.

**승인 해석 (세션 초두 고정, 유저 무정정)**: 유저 "D407 설계 착수 승인" =
설계 + 정적 준비 + attestation/tuple 작성까지. **실제 물리 실행은 tuple sha
인용 별도 명시 승인 필요** (D400-P2 3단계, DECISIONS.md:23653-23663).

**Session progress rule 정당화**: 이 세션은 설계 세션 — 실패 가능 실험 없음.
사유: D407 물리는 미승인(물리 별도 승인 규칙), 설계 자체가 유저 지시 대상.
정적 준비 단계(다음 세션)부터 실패 가능 fixture 다수 포함 예정.

## 1. 이 세션이 한 일 (감사 가능 step-by-step)

1. 부트: START_HERE/DECISIONS(D402-R1~D406)/LEDGER 444행/D406 세션문서 2건
   read + **D406 runtime 산출물 원본 재검증** (completion pass=true·verdict
   문자열·cook 136/136·property 65/66·tuple sha bc54e7c5·prereg status
   `PREREGISTERED_NOT_EXECUTED` 전부 bit-일치. 유일 뉘앙스: "육안검수 8체크
   전부 true"는 실측 JSON상 true 7 + `text_overlap_or_clipping_observed=false`
   (false=통과 상태) — "8개 통과 조건 충족"이 정확한 표현, 결론 동일).
2. 4-reader 병렬 증거 스윕 (workflow `wf_4819f3b0-545`, 4/4 완료, 126
   findings): D362 계약 / D362 스크립트 해부 / D400-D406 인프라 / D362→D400
   아크·규칙. journal:
   `~/.claude/projects/-home-cgxr-Documents-Robotics-RoArm-Project/6bc39c14-4ebc-4e93-a954-d8e407bc8d37/subagents/workflows/wf_4819f3b0-545/journal.jsonl`
3. load-bearing 사실 원본 스팟 재검증 (아래 §2).
4. 설계 초안 저작 (§3에 전문 보존).
5. 4-lens 적대 설계 리뷰 발사 (workflow `wf_0375d167-bc7`) — **결과 미회수**
   (context 한계로 세션 종료). journal:
   `~/.claude/projects/-home-cgxr-Documents-Robotics-RoArm-Project/6bc39c14-4ebc-4e93-a954-d8e407bc8d37/subagents/workflows/wf_0375d167-bc7/journal.jsonl`
   렌즈 4종: science(숨은 leg 차이/leg 간 오염/metric 상속) · frozen(D406
   derivative in-place 소비 시 쓰기 위험/변수 ladder/allowlist) ·
   lessons(D403~D406 소비자 리터럴 표면/정적 replay 불가 잔여) · ops(2-worker
   승인 구조/watchdog/cook 캐시 프로세스 간 지속성/D341·D324 충족).

## 2. 이 세션 원본 재검증 사실 (설계의 근거)

- D362 계약: 원통 r0.017×h0.090m/0.72kg/friction 1.5/1.2/rest 0.0, spawn
  [0.30000001192092896, 0, 0.03288299962878227], Q_FROZEN_OPEN_F32(q5 OPEN
  1.5413000583648682), OPEN 200 + close 300 = 500 steps, dt 0.005, actuator
  80/4/2.5/3.14, event 임계 0.1N/0.5mm/1.0°/연속2, seed 33201
  (d362 harness:48-124 직접 read).
- D362 결과: endpoint XY `60.61899778989994mm` / tilt `89.99777464743418°` /
  z `-28.000520542263985mm`; event 31/32→41/42→45/46; link4 0N; peak
  43.86/23.23/0.0N (DECISIONS.md:20757-20802).
- **A-leg 자산 무결**: D344 attempt3 base USD 현재 디스크 sha ==
  d362_preregistration.json input_hashes pin bit-일치 (roarm_m3.usd
  `a4be58e8...`, physics `043a5d35...`) — ledger 426행 기록과도 일치.
- **D406 derivative 6파일 sha256 실측**: .asset_hash `ae762fcc...`(32B),
  config.yaml `5745bbb8...`(663B), root roarm_m3.usd `c02808ab...`(3177B, SDF
  opinions), configuration/{base `ea0ee8f2...`(2769018B), physics
  `043a5d35...`(base와 bit-동일), robot `2227536f...`, sensor `3f44081f...`}.
  전부 .gitignore(*.usd) 대상 → sha pin이 유일한 freeze 수단.
- ledger 426행 verbatim: "Contact capacity 33,280 must be re-derived for SDF
  inventory"; "SDF+custom-geometry+29x50 동시 = 3 변수 ladder 위반".
- ledger 427행: 선행 제안 ladder = A64 baseline leg + gripper-only SDF leg의
  **2-live-leg 구조** (29×50용 superseded — 구조만 계승).
- BACKLOG.md:149-160: 목표 = "D362 전도가 실제로 개선되는지 물리로 첫 재측정";
  SDF 무비판 채택 금지.
- 스크립트 해부 (reader 검증 + 스팟 확인): D362는 단일 파일 4-stage
  (prepare/run/_worker/finalize), IK/waypoint 없음(정적 자세 실험), 14 science
  함수 byte-identity 계약(:889-939), A64-결합 게이트 5곳(capacity
  33,280=:1185,1324-1364 / live_binding_64_plus_64=:3905-3907 / d349 topology
  audit / shape inventory / rerun 64+64 표시), worker는 GUI(headless=False
  강제 :4835-4858), MP4 replay는 D362 신규 변수였음.
- 인프라: 동결 D400 worker는 fresh OUT_DIR copytree로 derivative 재생성 구조
  (worker.py:1194-1273); D406 derivative는 per-attempt 산출물. D400 prereg에
  물리 A/B rung 설계 문구 없음(reader 확인) — D407 설계는 신규 저작이 맞음.

## 3. D407 설계 초안 전문 (리뷰 반영 전 v1 — scratchpad 원본 보존용)

> 아래는 `wf_0375d167-bc7` 리뷰 대상 초안 verbatim. 리뷰 blocker 반영 후
> 확정본은 다음 세션 문서에서 저작한다.

### 3.1 무엇을 왜

- 과학 질문: D362 전도(XY 60.619mm, ~90° 전도)가 gripper_link 충돌 표현을
  A64→SDF res256으로 바꾸면 실제로 달라지는가를 물리로 첫 측정.
- 가설 증거 기반(인과 주장 아님 — D368 `collider_count_tipping_causality`
  =null 유지): D334 cooked hull 팽창 ~3.5-9.4mm + cook parity 1.46% FAIL.
- 개선 임계값(improvement gate) 사전등록 안 함 — 측정 case. `g0a_pass=false`
  유지.

### 3.2 실험 구조 — 2-live-leg A/B (supervisor 1회, worker 2회)

- **Leg A (control)**: 동결 D362 계약 verbatim + D344 attempt3 base asset
  (A64 64+64). 신규 변수 0 — 현재 스택 위 D362 재현.
- **Leg B (treatment)**: 동일 계약 + D406 derivative (link5 A64 64 유지,
  gripper A64 64 비활성 + SDF res256 1 mesh). 신규 변수 1.
- 2-leg 근거: ① 7/25 driver 교체(580.159.03→580.173.02, D402-R1)로 D362
  당시 스택과 다름 — 같은 스택 A/B만이 표현 변수를 고립; ② ledger 427행
  선행 ladder도 같은 구조; ③ A-leg vs 동결 D362 trace는 descriptive 재현성
  판독(게이트 없음)으로 격하, **A vs B 같은 세션 대조가 primary contrast**.
- 순서 고정 A→B, A operational FAIL_STOP 시 B 미발사. leg별 retry 0, seed
  동일 33201.

### 3.3 동결 계약 상속

§2의 D362 계약 전 항목 그대로 (원통/자세/q5/500-step/dt/actuator/임계/seed
불변; IK·경로 부재도 계약). 14 science 함수는 D362 원본과 byte-identical
(label 정규화만 — D362가 D360에 쓴 `_frozen_d360_science_source_contract`
메커니즘을 D362→D407로 재사용). 미기록 물리 설정(gravity/solver/offset)은
같은 env 구성 경로(roarm_rl RoArmCubeTap10cmEnv + d332
_configure_runtime_env)로 두 leg 동일 적용 + 런타임 실측 기록 의무(게이트
아님).

### 3.4 자산 (전부 sha pin)

- Leg A: D344 attempt3 base (D362와 bit-동일 실측, §2).
- Leg B: D406 derivative를 read-only in-place 소비 (D400 게이트 전체 PASS를
  통과한 바로 그 산출물; 재생성 변동 배제). 6파일 sha pin (§2). A↔B authored
  차이 = D400 게이트 검증 opinion 집합뿐임을 prereg 명문화 (instanceable=
  false 2 scope 포함 — 등록 변수 정의의 일부).

### 3.5 A64-결합 게이트 재설계 (5곳)

| D362 게이트 | A-leg | B-leg |
|---|---|---|
| capacity 33,280=1×(1+1+64+64)×256 | 동일 | **17,152**=1×(1+1+64+**1**)×256 재산정 |
| live_binding_64_plus_64 | 동일 | link5 64 + gripper enabled 0/disabled 64/SDF 1 — 기대치는 오프라인 USD 감사에서 도출 |
| shape inventory {…64,64} | 동일 | {…link5:64, gripper:**1**} |
| d349 topology audit (A64 결합) | 동일 | gripper 측을 SDF 입력 mesh identity(41,094 verts/13,698 tris, STL `7946a374...`)로 대체; link5 측 유지 |
| rerun 64+64 표시 | 동일 | gripper=SDF mesh 1 표시 (표시층) |

SDF per-pair>256 시 "Incomplete contact data" 경고 → D362 동일 overflow
audit이 잡음(fail-capable). + 원통 runtime geometry type probe(read-only,
양 leg 동일 코드) — ledger 427행 `..._RUNTIME_GEOMETRY_TYPE_PENDING` 해소.

### 3.6 관측성 (D341 물리 요건)

- leg별: durable JSONL prefix(d361 validator rebind), trace CSV/JSON, RRD
  전체 500-row timeline + decision scalars + contact points/force arrows,
  blueprint+RBL, headless 스크린샷, validation JSON(0.34.1 pin).
- worker **headless=True** (D362는 GUI): stale viewport 결함 경로(D365) 원천
  제거, actual-viewport PNG 없음. 양 leg 공통 → 대조 무오염. A vs D362
  비교에 환경 차이로 명기.
- **MP4 없음** (D324 trajectory video 금지 유지; D362 1920×1088 실패 클래스
  제거). beginner sheet는 canonical trace에서 matplotlib offline 생성.
- 라이브 육안검수 1회(양 leg 스크린샷 포괄): D406 원자적 작성기 + argv-청정
  감시 + 정직 보고 + 300s.

### 3.7 판정 설계

- Primary 판독값: leg별 최종 row(500) disp_xy/tilt/z + event onset/
  confirmation steps + peak forces 3종 + **B−A delta**(정의식 사전 고정).
- descriptive 분류: `toppled := 최종 tilt delta > 45°` (게이트 아님).
- verdict: leg별 physical sub-verdict + 전체
  `D407_SDF_PHYSICS_AB_TIPPING_REMEASURE_{MEASURED|FAIL_STOP}`.
- must_remain_null: force closure/stable grasp/cap-rim-barrel/grasp
  feasibility/29×50 이전 가능성/"SDF 일반 우월" 일반화/per-prim cooked SDF
  내부 identity. `g0a_pass=false` 유지.

### 3.8 정적 준비 stage 계획 (A~M)

A env/pin replica · B prereg shape + **소비자 리터럴 전수 grep**(d351/d332/
d361/d349/roarm_rl가 D407 입력에서 읽는 전부) + dirty⊆allowlist · C -B 거부
+ __pycache__ 0 · D science 14함수 byte-identity(프로그램적) · E frozen
input 해시 전수 재계산 · F **실물 trace replay**(동결 D362 trace 500 rows →
상속 파이프라인 오프라인 재적용 → endpoint 60.61899778989994mm·event
31/32·41/42·45/46·sub-verdict bit-exact 재현) · G capacity fixture
accept/reject · H D406 derivative 오프라인 USD 감사(omni.usd.libs pxr +
PhysxSchema plugInfo — D403 lesson; B-leg 게이트 기대치 도출원) · I d361
prefix validator rebind fixture · J Rerun 실물 replay(D362 trace로 관측성 층
전체 scratchpad 실행 + 검수 작성기 dry-run) · K 순수 admission 전수
replay(accept+변조 reject+이전 실패 재현) · L launch 구성 복제 subprocess
fixture · M negative ≥30 + 승인 게이트 복제 + 4-lens 리뷰.

### 3.9 승인 경계·실행 계획

- 3단계: ① 설계 리뷰 → ② 구현+정적 attestation+4-sha tuple(Isaac 0회) →
  ③ tuple sha 인용 runtime 승인(supervisor 1, worker 2=leg당 1, retry 0,
  watchdog 300s/900s per leg, 호스트 경계).
- 산출: `claudedocs/runtime_logs/grasp_track/g0a_d407/
  attempt1_sdf_physics_ab_d362_remeasure/{leg_a_a64,leg_b_sdf_res256}/`.
- 신규 repo 파일(allowlist 등재 대상): harness 1(`sim_scripts/
  cyl34_top_view_d407_sdf_physics_ab_d362_remeasure.py`), prereg/attestation/
  tuple/정적결과 JSON, 세션·상태 문서. 도구는 scratchpad에만.
- **runtime 전 commit/push 금지** (HEAD 변경=tuple 무효).

### 3.10 리스크 (정직 보고)

① SDF per-pair 256 초과 가능(overflow audit fail-capable) ② gravity/solver
미기록 상속→런타임 기록 ③ GUI→headless 환경차(A/B 내부 무오염, A vs D362
descriptive만 영향) ④ B-leg=체인 첫 SDF 물리 — 첫 라이브 실행 자체가
실험(D403 lesson) ⑤ 500-step 지평에서 중간 상태 종결 가능(유효 측정으로
수용).

## 4. 다음 세션 필수 절차

1. **리뷰 회수**: `wf_0375d167-bc7` journal.jsonl에서 4-lens 결과 read →
   blocker 전부 해소 반영한 설계 확정본 저작 (반영 없는 확정 금지).
2. 확정 설계로 prereg builder(리터럴 프로그램적 도출) + harness 구현 + 정적
   runner A~M — 전부 유저 기승인 범위 (runtime 제외).
3. tuple 작성 후 **정지** — runtime은 tuple sha 인용 새 명시 승인.
4. 주의: evidence sweep 전체 출력(/tmp)은 휘발 — 핵심은 이 문서 §2에 보존,
   전문은 위 journal 경로에 영속.

## 5. 경고 (불변)

- D400~D406 전 attempt 동결 (D406 PASS attempt 포함). D362 33파일 불변.
- allowlist 밖 repo 새 파일 금지 (이 세션 신규 = 이 문서 + 상태 문서 3종만).
- isaaclab env 불변 (D326). Isaac/GPU는 호스트 경계 (D402-R1).
- commit/push는 유저 요청 시에만 — 이 세션 종료 시점 dirty 82 (기존 81 + 이
  문서).

## 6. 부록 — 4-lens 리뷰 결과 (세션 마감 직후 완료, headline 회수)

4/4 완료, **전 렌즈 refutation_failed=true (설계 core 생존)**. **blocker 2 +
warnings 21** — 확정 설계는 아래 전부 반영 필수. 전문: wf_0375d167-bc7
journal (§1 경로).

**BLOCKER 1 [frozen] — Leg A 자산 pin 구멍**: 설계 v1은 Leg A가 실제 소비하는
gitignored USD 레이어 중 2개만 pin (root+physics = D362 prereg 열거 그대로).
configuration/roarm_m3_base.usd(2,769,018B 지오메트리, physics.usd가 참조)·
robot·sensor.usd는 소비되지만 미pin → "A↔B 차이 = D400 opinion 집합뿐" 주장이
게이트로 검증 불가, retry-0 attempt가 무검출 오염 위험. 수리: **Leg A 전
레이어 pin + 미저작 4개 레이어의 A↔B bit-동일성 게이트 추가** (리뷰어 실측:
base/robot/sensor 현재 D406 사본과 hash-일치 ea0ee8f2/2227536f/3f44081f).

**BLOCKER 2 [ops] — Leg A 관측성 실패 분기 미정의**: "A operational FAIL_STOP
→ B 미발사"만 정의, A의 Rerun/관측성 실패가 B 발사를 막는지 미정의. 수리:
prereg에 분기 명문화 (설계 방침: 기술 PASS + 관측성 FAIL이면 B 미발사 —
attempt 전체 FAIL_STOP; D404 전례의 기술/관측성 분리 판정 준용).

**주요 warnings (전문 journal)**: ① SDF pair의 contact-point 보고 가용성
미검증 — `all_qualifying_robot_force_points_finite_for_observability`
(d362:4198-4200)가 B-leg에서 fail 가능; NVIDIA 버전일치 문서 검증 + prereg
분류 필요 ② 256/pair envelope의 SDF 유효성 = convex 기반 가정 — 엔진 한계로
표기 금지 (NVIDIA 규칙) ③ instanceable=false가 link5 scope에도 적용 —
link5측 delta 귀속 caveat 명문화 ④ cook cache가 프로세스 간 지속(D406
hit 136/miss 0) — leg별 cook/cache provenance 카운터 기록 추가 ⑤ dirty는
draft의 81이 아니라 실행 시점 재열거 (현재 82) ⑥ 6번째 A64 게이트
`corrected_d348_128_of_128`(d362:3903-3904) 존재 — B-leg에서 historical-
evidence audit으로 재라벨 ⑦ leg 간 GPU settle 대기 부재 (MIN_GPU_FREE
8192MiB fail-closed) ⑧ [D405-class] draft "6파일 pin"이 실제 7파일 열거 —
카운트 리터럴 전수 프로그램적 도출 ⑨ draft의 "physics 043a5d35 = base와
bit-동일" 표현 모호 (roarm_m3_base.usd 아니라 D344 base asset의 physics.usd와
동일이라는 뜻 — 확정본에서 재서술) ⑩ D406 attempt1 동결 폴더 내 derivative의
in-place 라이브 소비 적법성 — mtime/사이드카 오염 방지 계약 필요 ⑪ 2-seq-
Isaac-worker는 repo 첫 구조 — 정적 replay 불가 잔여 목록에 추가 ⑫ session-
doc-literal 자기 검사(D362 :1004-1005류)의 replay 계획 명시 ⑬ harness 1파일
vs D400-P2 4-sha tuple(controller/worker 분리) 정합 재설계 ⑭ Korean font
게이트 유지 여부 명시 ⑮ D324 target-vs-actual frame marker를 leg별 산출물에
명시. (전체 21건 — journal 참조)

## 7. 다음 세션 continuation prompt (verbatim 붙여넣기용)

```
Read AGENTS.md first, then follow the Current-State Protocol exactly
(START_HERE.md → DECISIONS.md tail (D406) → EXPERIMENT_LEDGER.md tail (445행
= D407 설계 세션) →
claudedocs/session_20260728_grasp_g0a_d407_design_kickoff_evidence_sweep.md
(§3 설계 v1 전문 + §6 리뷰 결과 필독) → git status --short
--untracked-files=all).

Active state: D407 [sdf_physics_ab_d362_remeasure] 설계 초안 v1 + 4-lens 적대
리뷰 완료 (blocker 2 + warnings 21, 전 렌즈 core 생존 — 세션 문서 §6).
유저 승인 범위 = 설계+정적 준비+attestation/tuple까지. runtime은 tuple sha
인용 별도 명시 승인 (아직 없음).

즉시 할 일 (기승인 범위):
1. §6의 blocker 2 + warnings 21 전부 반영한 설계 확정본 저작:
   - BLOCKER 1: Leg A gitignored USD 전 레이어(base/robot/sensor 포함) sha
     pin + 미저작 4개 레이어의 A↔B bit-동일성 게이트 추가.
   - BLOCKER 2: Leg A 관측성 실패 시 B 미발사 분기 명문화.
   - 리뷰 전문 journal:
     ~/.claude/projects/-home-cgxr-Documents-Robotics-RoArm-Project/6bc39c14-
     4ebc-4e93-a954-d8e407bc8d37/subagents/workflows/wf_0375d167-bc7/
     journal.jsonl (증거 스윕 전문은 wf_4819f3b0-545/journal.jsonl)
2. prereg builder(소비자 리터럴 프로그램적 도출 — D405/D406 lesson) +
   harness(sim_scripts/cyl34_top_view_d407_sdf_physics_ab_d362_remeasure.py;
   controller/worker 분리 여부는 §6 warning ⑬ 반영해 재설계) + 정적 runner
   stage A~M (§3.8).
3. attestation + 4-sha tuple 작성 후 정지 — runtime 새 명시 승인 대기.

설계 v1 core (§3 전문): 2-live-leg A/B (A=D362 계약 verbatim + D344 base A64
control / B=동일 계약 + D406 derivative SDF; supervisor 1, worker 2, leg당
retry 0, A→B 고정, A FAIL_STOP 시 B 미발사, seed 33201); capacity A 33,280 /
B 17,152; A64-결합 게이트 재설계(§6 ⑥: 6번째 게이트 포함해 6곳); science
14함수 byte-identity(d362:889-939 메커니즘); headless 양 leg 공통; MP4 없음;
라이브 육안검수 1회(D406 원자적 작성기 계약); 개선 임계값 없음;
g0a_pass=false 유지.

Rules:
- D400~D406 전 attempt + D362 33파일 동결. 물리/q5/contact/cylinder 변경 =
  별도 승인. Isaac/GPU는 호스트 경계(D402-R1). isaaclab env 불변(D326).
- allowlist 밖 repo 새 파일 금지 (도구/리뷰 산출물은 scratchpad에만).
- runtime 전 commit/push 금지 (HEAD 변경 = tuple 무효). dirty 82+ 미커밋.
- /half-clone·HANDOFF.md 금지 (HARD RULE #7/#11).
- 비판적·회의적 교차검증, 파일:라인 인용, 메모리 단독 주장 금지.
  step-by-step으로 순차적으로 사고하면서 진행해.
```
