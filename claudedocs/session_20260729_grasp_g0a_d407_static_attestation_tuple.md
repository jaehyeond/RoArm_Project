# D407 SDF physics A/B — static attestation and tuple preparation

Date: 2026-07-29 KST  
Case: `g0a_d407/attempt1_sdf_physics_ab_d362_remeasure`  
이번 case의 신규 변수:
[`gripper_link_collision_representation_a64_to_sdf_res256_v1`]

## 1. 범위와 승인 경계

이번 세션은 D407 harness 검토·필요한 최소 수리, preregistration,
static stage A~M, reviewed static attestation, 4-SHA tuple 작성까지만
수행한다. Isaac/Kit/PhysX/GPU runtime, physics step, q5 표본, contact query,
cylinder 생성은 수행하지 않는다. tuple 작성 후 정지하며 실제 A/B
runtime은 tuple-file SHA를 인용한 사용자 별도 명시 승인이 필요하다.

## 2. 부팅 및 독립 검토

- `AGENTS.md` → `START_HERE.md` → `DECISIONS.md` D402-R1~D406 →
  `EXPERIMENT_LEDGER.md` 444~446행 → 지정된 두 세션 문서 순서로
  재독했다.
- 시작 시 `git status --short --untracked-files=all`은 예상과 같은
  86개였고, 예상 밖 파일은 없었다.
- `HEAD == origin/master ==
  a69a96d36219268e4bc5e25065cc234da9d99674`를 직접 확인했다.
- 세 read-only 검토를 병렬 실행했다. worker 과학 계약, controller 운영
  계약, prereg/builder 소비 계약을 분리했으며, 각 finding은 메인 agent가
  원본 file:line으로 다시 판독했다. Claude의 사용량 한도 실패 0/6
  workflow는 판정 증거로 사용하지 않았다.

## 3. Harness 수리와 동결

- 최종 행 수: worker **4,334행**, controller **1,862행**.
- worker: D362 동결 과학 14함수 DXXX-normalized source identity 14/14,
  baseline + 500-row horizon, event/verdict/delta, A/B representation,
  topology·관측 계약을 재검증했다.
- controller: A→B 순서, A FAIL 시 B 미발사, process-group watchdog,
  inter-leg GPU settle, exclusive-create, retry 0, post-run 전체 재해시,
  단일 수동 검수 계약을 재검증했다.
- 최소 수리는 import 전 `-B` guard, frozen-input 실재 집합, SDF topology
  변환/해시, baseline/500-row gate, process-group 정리, final tuple·root
  artifact 재해시, first-failure 분류 보존을 포함한다.
- 등록 설계 문서에 prereg SHA 주입 후 controller/worker SHA를 각각
  정확히 한 번 기록했고, prereg builder가 이를 확인했다.

## 4. Preregistration 확정

- artifact: `D407_PREREGISTRATION_V1`
- prereg SHA-256:
  `6deb6779a18619f547952de9119eee599ea5dd40ac466d57d6a813988afb1269`
- status는 실제 worker/controller 소비자의 `ast.Eq` 비교에서 도출한
  `PREREGISTERED_NOT_EXECUTED`이다.
- frozen inputs 36개, A/B asset 7+7개, D334 sidecar 3개를 exact-set으로
  확인했다.
- dirty allowlist는 live dirty 86개 + planned static 7개 + 실제
  controller/worker 소비자에서 도출한 future leaf 45개와 leg directory
  sentinel 2개의 합집합이다. 최종 고유 allowlist는 138개다.
- 설치 pin은 `numpy==1.26.0`, `psutil==5.9.8`,
  `rerun-sdk==0.34.1`, `isaaclab==2.3.0`을 fail-closed로 확인했다.
- M2는 설계 확정 문서 §4에서 전사해 설치 NVIDIA primary source 8개와
  공식 source 7개를 pin했다. SDF point 보고 API는 geometry-agnostic으로
  문서화됐지만 convex/SDF parity는 문서에 직접 쓰이지 않은 추론이고,
  `256 contacts/pair`는 engine hard limit나 GPU 공통 상수가 아니라
  **미문서화 프로젝트 가정**이다. 따라서 첫 runtime의 fail-capable
  overflow warning audit가 최종 판정자다.

## 5. 정적 stage A~M

### 5.1 첫 fail-closed 시도와 최소 수리

- `/tmp/d407_static_prep_20260729/static_run_final_v1/`의 첫 시도는
  **repo 결과를 쓰기 전에** fail-closed로 끝났다. 실패 기록 SHA-256은
  `f7aa510ebae5f42537765d96d6c04739e1ac03eed768d7c4edb8595e915676f9`이다.
- 실패 원인은 과학/물리 결과가 아니었다.
  1. 도구 sandbox가 `/dev/nvidia*`와 host PID를 숨겨 D402-R1 host-boundary
     정적 체크가 실패했다.
  2. USD float32 `0.01`의 실제 값 `0.009999999776482582`
     (`0x3c23d70a`)을 Python float64 `0.01`과 직접 비교한 runner 오기였다.
  3. 같은 sandbox 선행 실패 때문에 controller zero-tuple probe가 tuple
     gate가 아니라 host-boundary에서 거부되어 Stage L의 좁은 분류가
     실패했다.
- 실패 scratch는 보존했고 삭제/덮어쓰기하지 않았다. float 속성은 값의
  float64 표면표현이 아니라 등록된 float32 bit pattern으로 비교하도록
  고쳤고, 새 forward-only 경로 `static_run_final_v2`를 사용했다. 호스트
  경계 재시도도 Isaac/Kit/PhysX, CUDA job 또는 물리 worker를 발사하지 않는
  동일 정적 runner였다.

### 5.2 최종 static runner 결과

- stages A~M: **13/13 PASS**
- checks: **58/58 PASS**
- positive fixtures: **10/10 PASS**
- negative fixtures: **59/59 PASS**
- 결과:
  `claudedocs/runtime_logs/grasp_track/g0a_d407/attempt1_sdf_physics_ab_d362_remeasure/d407_static_fixture_results.json`
- 결과 SHA-256:
  `568e7df1fdcb5bdd5117fc418bdeb55c284131e21f6e77dd782ac583f22ee1ea`
- post-publish dirty exact receipt SHA-256:
  `cbc949772dd83e045d66e403afade9f9c553664e00e00c7831f0382eac1701c6`
- stage K 시점 dirty path와 결과 발행 직전 dirty path가 정확히 같았고,
  결과 파일 1개를 더한 발행 후 실제 dirty 집합도 예측 집합과 정확히
  같았다.
- 정적 counter 10종은 전부 0:
  script/runtime import past refusal, Isaac/Kit/PhysX launch,
  SimulationApp, physics worker, USD create/write, hardware GPU job,
  physics step, q5 sample, contact query, cylinder create/write가 모두
  **0**이다.

### 5.3 정적 과학·형상 수치

- D362 science source identity: **14/14**
- 동결 trace: **500 rows**, 200-step OPEN + 300-step close/observe
- 최종 XY 이동:
  `60.61899778989994 mm`
- 최종 기울기 변화:
  `89.99777464743418 deg`
- 최종 z 변화:
  `-28.000520542263985 mm`
- event onset/confirmation:
  gripper `31/32`, object motion `41/42`, link5 `45/46`
- peak force:
  gripper `43.85833992858175 N`, link5 `23.227865254723564 N`,
  link4 `0.0 N`
- source-derived capacity:
  source envelope `256 contacts/pair`, leg A `33,280`, leg B `17,152`
- read-only USD:
  A=`link5 64 + gripper_link 64 enabled`;
  B=`link5 64 + A64 gripper 64 disabled + SDF mesh 1 enabled`
- B SDF mesh:
  vertices `41,094`, triangles `13,698`,
  source-stream SHA-256
  `31aead25f7aa879a358a046bc01291ef2e260a2b367a990dacc255c17a2a5a31`,
  body-local float64 points SHA-256
  `522a4f0fe91a04bf54c5c8be6492748c7490fc557fa8c0867200d97332dfa9db`
- body-local bounds:
  min `[-0.010767397438303794, -0.009999632356670897, -0.0386173457368133]`,
  max `[0.06708260664084253, 0.015240367659608567, 0.0007502218245168529]`
- SDF 7개 authored 속성은 type/value/float32 bit/uniform/no-time-sample/
  no-connection 계약을 전부 통과했다. `sdfNarrowBandThickness`와
  `sdfMargin`의 bit pattern은 각각 `0x3c23d70a`이다.

### 5.4 Rerun·sheet 직접 육안검사

원본 해상도로 다음 다섯 PNG를 직접 열어 검사했다.

- Stage J replay/렌더 traversal 시간은
  `6.191016671997204 s`였고 Mesa lavapipe software Vulkan을 사용했다.
  hardware GPU runtime 사용은 false다.

1. leg A Rerun `1920x1080`, SHA
   `c8f4f7dea01aaae4f97d85e55caae7805434b4917c0763d428d480f5e3e93d30`
2. leg B Rerun `1920x1080`, SHA
   `baca0fd801b926ce57ef6b3454c6b3e46de3c543f0f8af0a6f83a712c9ecba09`
3. leg A beginner sheet `3840x1720`, SHA
   `eb1ea7d887ef3f94149465029b68956b8298de4fdba9c7dca4ffa52e217ae611`
4. leg B beginner sheet `3840x1720`, SHA
   `a85a84978920043a0fb50dd73b80aab822fd3e7b8552b8f9088ef0343af9dd53`
5. A/B comparison sheet `3840x1080`, SHA
   `118c7ae2e4dff4c0d459db830149a1239870d6e9a1bdfaaaddc356bb7cf68ac5`

관찰:

- 두 Rerun 화면에서 gripper/jaw, cylinder, force witness, q5/force/object
  motion 3개 시계열 패널이 식별 가능했다. A 제목은 `actual 64+64
  colliders`, B 제목은 `link5 64 + gripper SDF mesh`로 서로 구분된다.
- Rerun 기본 notification/hover legend가 화면 일부를 덮고, 같은 위치의
  support-table force label 두 개가 서로 닿는 UI 수준의 overlay는 보였다.
  그러나 decision subject, 곡선, event 변화 및 패널 제목은 가려지지 않아
  판독 가능했다. 이를 presentation warning으로 기록하되 정적 replay
  소비계약의 blocker로 분류하지 않았다.
- 두 beginner sheet는 시작(row 0), 첫 robot-body 접촉 확인(row 232),
  최종(row 499)의 primary/opposite view와 q5·force·motion plot을 모두
  보여 주며, 최종 원통 전도가 명확하다. 한글/영문 glyph 누락, 패널 잘림,
  의미를 바꾸는 text overlap은 보이지 않았다.
- A/B sheet는 좌우 leg, 세 시점, 요약 수치와 `Δ(B-A)`를 함께 보여 주며
  clipped panel이 없었다. A와 B 수치가 같은 이유는 이 단계가 같은 동결
  D362 trace의 **표시 소비 경로**를 검증한 것이기 때문이다. PNG에도
  `PHYSICS NOT RECOMPUTED`가 명시되어 있으며, 이는 실제 B-leg physics
  결과나 과학 A/B verdict가 아니다.
- RRD/RBL:
  A RRD
  `80e746ceda7fd8d65a77db1e78088246c6eefc79503d0302f155523d73d39b90`,
  A RBL
  `b0f9aa6b2859c5129cb1c4677680aeaf3048704c0260fd7b106a194614cd3eed`;
  B RRD
  `5247eb1f88ec44cbec2193608effbd2c61310256e4095020ecf8e7e8632f6e15`,
  B RBL
  `6987c5e701bd0d058402ef2291efa92c7f9f2d318cd79d8c980d24ff0a283606`.
  footer verify, exact entity/timeline/component, blueprint와 screenshot
  계약은 runner에서 모두 PASS했다.

이 정적 세션의 fail-capable 작업은 source/asset/prefix/tuple/manual-writer
mutation 및 59개 reject fixture였다. 실제 물리 perturbation을 하지 않은
이유는 사용자가 이 턴의 권한을 static attestation/tuple까지로 명시하고
tuple 작성 뒤 정지하도록 요구했기 때문이다.

## 6. M-late 실게이트와 원자적 attestation/tuple 게시

### 6.1 정적 도구 동결값

- static runner: 3,046행, SHA-256
  `25134381aeecf85e02302130a92735aa9d00c5b78e64040f6ae00d8098793917`
- manual writer dry-run: 209행, SHA-256
  `760c3447a6e83fe80242d00979c6d1c609f9a483cfcb5fdcf6621701b3343514`
- M-late/atomic publisher: 874행, 최종 SHA-256
  `1c20072dddad8777fd7ed2a0043f496f4366983b936578f7d2911a11b83b4284`
- 위 도구와 모든 replay 중간 산출물은
  `/tmp/d407_static_prep_20260729/`에만 두었다.

### 6.2 실제 controller tuple validator 정적 replay

- 실제 controller의 `_validate_approval_tuple()`을 runtime 함수 호출 없이
  scratch 후보에 적용했다.
- 정상 후보: **1/1 accept**
- 등록된 변조 후보: **15/15 reject**
- M-late report SHA-256:
  `6af2db8e8b9a5168fc37d23d370bbf9fec65e19ce3932b97f205ca70e560daa7`
- `runtime_executed=false`; `run_runtime`은 호출하지 않았다.

### 6.3 게시와 최종 파일 상태

- sandbox 안의 첫 게시 시도는 scratch와 repo가 서로 다른 mount라
  hardlink가 `EXDEV`로 거부되었고, 최종 파일·candidate를 만들기 전에
  fail-closed했다. 동일 파일시스템인 D402-R1 호스트 경계에서 **정적 게시
  동작만** 재시도했다.
- attestation과 tuple은 hardlink no-clobber → scratch source unlink →
  양 directory `fsync` 순서로 게시했다. 최종 두 파일은 regular,
  non-symlink, `st_nlink=1`이고 candidate는 남지 않았다.
- 게시 후 실제 controller validator를 최종 tuple-file SHA로 다시 호출해
  PASS했다. 이때도 runtime 함수는 호출하지 않았고, D407 runtime 출력은
  사전에 존재하지 않았다.
- 현재 dirty는 91개, prereg allowlist는 138개이며
  `dirty ⊆ allowlist`, allowlist 밖 경로는 0개다.
- final static outputs는 preregistration, static results, reviewed
  attestation, proposed runtime tuple의 정확히 4개다.

## 7. 최종 분류와 4-SHA tuple

### 7.1 검토 finding 분류

- **BLOCKER — 수리 완료**
  1. leg A 실행 뒤 생길 파일 때문에 B 시작 시 dirty gate가 자기 정상
     산출물을 오염으로 오판할 수 있었다. builder가 실제
     `controller._runtime_output_paths()`와 worker 소비자에서 future leaf
     45개 및 leg directory sentinel 2개를 도출하고, live dirty + planned
     static + future runtime 합집합을 작성하도록 고쳤다.
  2. prereg status를 `!=` 정규식으로 추측하지 않고 실제 코드의
     `ast.Eq` 리터럴에서 도출하도록 고쳤다.
  3. harness에는 import 전 `-B` 거부, frozen-input exact set,
     baseline/500-row, process-group cleanup, final tuple/root artifact
     post-run rehash 및 first-failure 보존을 보강했다.
  4. 정적 게시기는 overwrite/race, symlink·비정규 파일, dangling
     symlink, crash-window resume, cross-device hardlink를 fail-closed하고
     최종 SHA·candidate 부재·`nlink=1`을 재확인하도록 고쳤다.
- **WARNING — 채택 비차단**
  1. Stage G의 두 fixture 이름은 leg A/B별 envelope처럼 보이지만 실제
     perturbation 대상은 공유 `source_envelope`이다. 거부 의미는 유효하나
     명칭은 다소 부정확하다.
  2. Rerun 기본 notification/hover와 같은 위치의 support force label
     접촉은 presentation warning이다. decision subject와 곡선은 판독
     가능했다.
  3. bounded single-writer 계약 밖에서 동시에 외부 파일을 바꾸는 극소
     TOCTOU(time-of-check/time-of-use, 검사와 사용 사이 변경) 위험은
     일반적으로 남는다.
- **FALSE-POSITIVE**
  1. A/B가 같은 D362 trace 수치를 보이는 것은 표시 소비 경로의 정적
     replay이며 B physics 재측정이 아니다.
  2. Mesa lavapipe Rerun 렌더는 software Vulkan 표시 작업이며
     Isaac/PhysX 또는 hardware-GPU runtime이 아니다.
  3. 첫 runner의 sandbox GPU-node/PID 가시성 실패는 D402-R1 경계
     불일치이며 GPU 고장이나 과학 FAIL이 아니다.

최종 독립 read-only 재검토와 메인 agent 재대조 후 blocker는 0개다.

### 7.2 승인 대상 4-SHA tuple

순서가 고정된 네 값은 다음과 같다.

1. `preregistration_sha256`:
   `6deb6779a18619f547952de9119eee599ea5dd40ac466d57d6a813988afb1269`
2. `reviewed_script_attestation_sha256`:
   `86d587e687b4d139083137913bd15b57cf8f394e2fd5114bafa38567415bae91`
3. `controller_script_sha256`:
   `c758ffad7199c425e87526cad54dbf7e100dbed004460d44f908421ad6a13dc1`
4. `worker_script_sha256`:
   `2f6da11cc9d074d7fa626eaadfb9a638b3cc74e7acdb2ae99fe07780041101cc`

tuple file:
`claudedocs/runtime_logs/grasp_track/g0a_d407/attempt1_sdf_physics_ab_d362_remeasure/d407_proposed_runtime_hash_tuple.json`

tuple-file SHA-256:
`c7001b76fa0a6c3393d9df744bb1fc0fb419400d46d84fc69e531730400d4b99`

## 8. 남은 runtime 위험과 다음 승인 경계

정적 PASS가 아직 답하지 못한 live 질문은 다음과 같다.

1. SDF contact pair가 프로젝트 가정인 256 points/pair envelope를 넘는지
   실제 overflow audit가 필요하다.
2. leg B에서 ContactSensor가 실제 SDF part에 binding되는지,
   property query 결과가 센서 관측으로 이어지는지는 첫 live 항목이다.
3. SDF pair의 contact point가 실제 runtime에서 제공되는지 확인해야 한다.
4. A와 B 두 연속 Isaac worker 사이 teardown·GPU settle이 실제로
   상호간섭을 막는지는 아직 미실행이다.
5. cook cache 상태 의존성이 있어 B의 fresh cook은 보장되지 않으며
   provenance만 기록한다.
6. gravity와 solver 설정은 runtime에서 기록·대조해야 한다.
7. D362 GUI와 D407 headless 차이는 기술적 설명 변수로 남는다.
8. 500-step horizon은 물체 상태가 완전히 수렴하기 전에 끝날 수 있으나
   등록된 동일 측정창으로 수용했다.
9. live 수동검수 timeout은 운영 위험이다.
10. leg A가 기술 또는 관측성 FAIL이면 attempt는 소진되고 B는 미측정이다.

또한 leg B derivative는 gripper representation 변경과 함께 link5에
`instanceable=false`가 적용되어 있다. 그러므로 이후 관측되는 B−A 차이를
무조건 gripper SDF 하나의 인과효과라고 과대해석하면 안 된다.

현재 status는 `STATIC_REVIEWED_RUNTIME_NOT_EXECUTED`,
`scientific_or_physics_verdict=null`, `g0a_pass=false`다. 다음 문장은
**사용자가 새 메시지로 명시해야만** runtime 승인으로 성립한다.

> D407 tuple-file SHA-256 c7001b76fa0a6c3393d9df744bb1fc0fb419400d46d84fc69e531730400d4b99를 사용한 Isaac/PhysX A/B runtime 1회를 승인합니다 (controller 1회, leg A/B worker 각 1회, retry 0).

이 문서 작성 시점까지 Isaac/Kit/PhysX/GPU runtime, physics step, q5,
contact query, cylinder 실행은 없었고 commit/push도 하지 않았다.
