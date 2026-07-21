# D373 P34 live asset identity preflight — fail-stop

Date: 2026-07-22 KST  
Case: `D373 [p34_live_asset_identity_preflight]`  
Output: `claudedocs/runtime_logs/grasp_track/g0a_d373/attempt1_p34_live_asset_identity_preflight/`  
Final operational verdict: `D373_P34_LIVE_ASSET_IDENTITY_FAIL_STOP`  
`g0a_pass=false`

## 1. 무엇을 왜 확인했는가

D372는 교수님 안인 `link5 몸통 + 고정 턱 + 움직이는 턱`을 수동 의미 기반 복합
충돌체 P34로 설계했다. 그러나 D372는 오프라인 기하 판정이었고 실제 USD나 PhysX에
올리지 않았다. D373은 그 P34를 실제 derivative USD에 한 번 기록한 뒤 다음 질문만
확인하도록 승인됐다.

1. D372 Float64 원자료가 등록된 Float32 authored stream으로 정확히 기록·readback되는가.
2. 활성 owner/path/count가 정확히 link5 16 + gripper_link 18인가.
3. PhysX가 property binding과 instance/prototype callback polygon을 유효하게 읽는가.
4. authored↔callback surface/bounds/original-polygon topology-volume과
   mass/COM/inertia가 동결 계약을 만족하는가.

이번 case의 신규 변수:

`[p34_live_asset_materialization_and_binding_v1]`

이 단계는 원통 파지 물리시험이 아니다. physics/q5/contact/cylinder/target·IK·path는
모두 0으로 동결했다.

## 2. 부팅 및 Git 교차검사

- 프로젝트 `AGENTS.md`와 Current-State Protocol에 따라 `START_HERE.md`, 전체
  `claudedocs/DECISIONS.md`, `claudedocs/EXPERIMENT_LEDGER.md`, D372 current session과
  원 evidence를 다시 읽었다.
- 작업 시작 전 Git은 다음과 같이 일치했고 worktree는 clean이었다.
  - `HEAD = 5214721e91bd23b224998cba2b13a1f76294edad`
  - `origin/master = 5214721e91bd23b224998cba2b13a1f76294edad`
  - subject `D371-372변경`
- commit/push는 실행하지 않았다.
- 사용자 소유 D334 collision-table sidecar 세 파일은 사전등록과 종료 감사에서 모두
  같은 SHA-256을 유지했다. inventory SHA-256은
  `86c3a8f58b0866458910d2cab13da69f04c2dba5ddfc430a8b648367d759fef2`다.

## 3. 동결 입력

주요 입력은 다음 SHA-256으로 다시 고정했다.

- D372 P34 Float64 geometry:
  `12fd1f32c35dfb9ae36cbbb412f6a51536aa1cc07c2dc17d05a5d189f3ee83e4`
- D372 evidence:
  `d68f658089aaf838ff454e9d0b301ec3f602785a3a730b3c329aa7785010e984`
- D372 completion:
  `57f3ed8fe6f057d059980a78bb51be8e881d8300297a4f41def6ddf94ad0cf43`
- D344 base root USD:
  `a4be58e87b1f9790f2a2ed600f0620c79d3cfb95c608b3c598308d52f5e46fff`
- D344 base physics layer:
  `043a5d35aa425c2589c77a34fcf415401ce9f9e7155e84ed75f6737df79fb503`
- D373 preregistration:
  `f6cc93647c16ad441776846308601d797fcdcdf081ba2e57c0cec4b571b21e2d`

Derivative asset은 D344 asset 디렉터리의 non-physics 파일을 복사하고 physics layer의
A64 subtree만 P34 subtree로 교체했다. non-physics 파일은 bit-exact였고 바뀐 파일은
physics layer뿐이다. 새 physics layer SHA-256은
`1284fe48686baf1746d3a1537cb4774f3f32292f87fafb5eacf1e69772c8a9e8`이다.

## 4. NVIDIA 버전 및 1차 자료

설치 환경:

- Isaac Sim `5.1.0.0`
- `omni.physx 107.3.26`
- NVIDIA driver `580.159.03`
- RTX 4090 Laptop GPU, `16376 MiB`
- Ubuntu `22.04.5`, kernel `6.8.0-124-generic`

사용한 NVIDIA 1차 자료:

1. **Omni Physics 107.3 — Rigid Bodies**  
   https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/rigid_bodies.html  
   하나의 rigid body 아래 여러 collider를 둘 수 있으나, articulation link는 scenegraph
   instancing이나 point instancing을 할 수 없다고 명시한다. 이 버전은 설치
   `omni.physx 107.3.26`과 맞는다.
2. **Omni Physics 107.3 — Query The Mass and Volume**  
   https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/mass_inertia_queries.html  
   `IPhysxPropertyQuery.query_prim`은 유효한 rigid-body prim과 descendant collider를
   stage/prim ID로 비동기 열거한다.
3. **Omni Physics 107.3 — Colliders**  
   https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/collision.html  
   convex Mesh collider와 approximation 의미를 확인했다.
4. **Isaac Sim 5.1.0 — Physics Simulation Fundamentals**  
   https://docs.isaacsim.omniverse.nvidia.com/5.1.0/physics/simulation_fundamentals.html
5. **Isaac Sim 5.1.0 — Performance Optimization Handbook**  
   https://docs.isaacsim.omniverse.nvidia.com/5.1.0/reference_material/sim_performance_optimization_handbook.html

설치된 local schema에서도 `physxConvexHull:minThickness` 형식은 `float`, 즉
Float32다. installed runtime warning과 asset validator rule은 동적 RigidBodyAPI가
instance proxy 내부에 들어간 D373 stage 구성을 지원하지 않는다고 일치해서
property-query 원인 판정에 사용했다.

## 5. 사전등록

Prepare 단계는 `13/13 PASS`였다.

- 신규 변수 정확히 1개.
- P34 manifest: link5 `16`, gripper_link `18`, total `34`.
- failure-capable prepare controls `4/4 PASS`:
  - wrong frozen input hash 거부
  - one missing part 거부
  - fixed/moving owner swap 거부
  - one Float32 byte flip 거부
- actual worker `1`, automatic retry `0`, watchdog `900s`.
- callback `68`, property query `2`, stage attach/detach `1/1`.
- physics/q5/contact/cylinder/target·IK·path/automatic-decomposition/settings change는 0.

등록 임계값:

- surface/bounds: `0.0001m = 0.1mm`
- authored↔callback original-polygon topology-volume relative: `0.005`
- callback↔property volume relative: `0.05`
- authored MassAPI state atol: `1e-12`
- live property mass/COM/inertia/axes atol: `1e-9`

## 6. 실제 실행 순서

1. GPU가 사용 가능한 host 환경에서 headless Isaac Sim을 한 번 시작했다.
2. frozen D372 geometry와 preregistration hash를 재확인했다.
3. timeline이 stopped이고 time `0.0s`인지 확인했다.
4. D344 base asset을 한 번 복제하고 P34 link5 16 + gripper_link 18 Mesh collider를
   physics layer에 기록했다.
5. direct physics-layer Float32 readback, MassAPI, canonical comparator를 실행했다.
6. in-memory inspection stage를 만들고 `/World/Robot` reference에
   `SetInstanceable(True)`를 적용했다.
7. live active collider inventory와 live instance/prototype authored stream을 읽었다.
8. PhysX stage를 한 번 attach했다.
9. 각 34 part에 대해 prototype→instance 순서로 callback을 68회 실행했다.
10. link5와 gripper_link에 property query를 각각 한 번 실행했다.
11. timeline과 scope counter를 다시 기록하고 PhysX stage를 detach한 뒤 앱을 닫았다.

작업자 프로세스는 `7.671346287010238s`, return `0`, timeout/signal 없이 끝났다. 하지만
return 0은 정상 cleanup만 뜻하며 worker 내부 계약은 `false`였다.

## 7. 실제로 성공한 원시 획득

### 7.1 USD와 live authored stream

- derivative materialization `1`.
- 활성 P34는 정확히 link5 `16` + gripper_link `18` = `34`.
- 활성 A64 `0`.
- 비활성 collision path는 알려진 legacy mesh 두 개뿐이다.
- direct USD Float32 points/counts/indices/aggregate payload는 `34/34 exact`.
- live instance points/counts/indices는 `34/34 exact`.
- live prototype points/counts/indices는 `34/34 exact`.
- owner/path/count bijection은 `34/34 exact`.

이 수치들은 `d373_worker_raw_summary.json`의 authored-readback 집계와 live inventory에
기록됐다. 따라서 “P34가 USD나 live stage에 아예 없다”는 해석은 반증된다.

### 7.2 PhysX callback protocol

- callback request `68/68` 완료.
- 각 channel은 callback exactly once, inline, result valid, one convex,
  serialization error 없음.
- part protocol은 `34/34 PASS`.
- worker exception은 `null`.

각 callback의 원 polygon/vertex/index 배열은 `callback_witnesses/`의 68개 JSON으로
남았다. 다만 후속 공식 surface/bounds/topology-volume classification은 fail-stop 때문에
실행하지 않았으므로 그 최종 gate는 `null`이다.

### 7.3 authored MassAPI와 timeline

- link5 mass/COM/inertia/principal axes base↔derivative 최대 절대차 `0.0`.
- gripper_link도 최대 절대차 `0.0`.
- timeline before/after raw tuple은 bit-exact이고 계속 stopped였다.

이는 authored API 보존만 증명하며, property query가 실패했으므로 cooked/live property
mass/COM/inertia gate를 대신하지 않는다.

## 8. 전체 identity가 중단된 이유

### 8.1 Float32 `minThickness` 거짓 FAIL — 확정

34개 collider에 decimal `0.0001m`를 기록했지만 schema type이 Float32라 실제
readback은 모두 다음 값이었다.

`0.00009999999747378752m`

decimal 기준과의 절대차는
`2.526212488436659e-12m = 2.526212488436659e-9mm`다. 형상 의미로는 무시할 만큼
작지만, D373 코드는 이 Float32 값을 Python decimal `0.0001`과 `1e-12m` tolerance로
비교했다. 차이가 tolerance의 약 `2.5262배`라 direct 34개와 live 34개가 모두
`min_thickness_frozen=false`가 됐다.

다른 per-row points/counts/indices/owner/schema check는 모두 true였다. 따라서 이것은
P34 geometry나 설정 손상의 증거가 아니라, 비교 전에 기준값을 schema Float32로
양자화하지 않은 typed-scalar gate의 거짓 FAIL이다.

더 중요한 계보상 문제는 이것이 새 실패가 아니라는 점이다. D342가 동일한 authored
`minThickness`에 미등록 `1e-12m` comparator를 적용해 정확히 같은
`2.526212488436659e-12m` delta로 실패했고, D343은 expected typed bits
`0x38d1b717`과 typed value `9.999999747378752e-05m`를 권위로 삼아 이미 수리했다.
D373 preregistration/worker가 그 D343 typed contract를 상속하지 않고 decimal 비교를 다시
넣은 것은 do-not-repeat 계보 회귀다.

### 8.2 whole-robot instance proxy property-query 실패 — 확정

D373 inspection stage는 `/World/Robot` 전체를 instanceable로 만들었다. 그 아래의
`link5`와 `gripper_link`는 동적 articulation rigid-body instance proxy가 됐다.

실제 설치 PhysX 로그는 두 body에 대해 다음 의미의 경고를 기록했다.

`RigidBodyAPI on an instance proxy not supported, unless set to kinematic or not enabled.`

이어 관절 body가 없다는 오류가 stderr에 기록됐고, 두 property query 모두
`PhysxPropertyQueryResult.ERROR_PARSING(5)`를 반환했다.

- link5 expected collider rows including disabled legacy: `17`; observed error row `1`.
- gripper_link expected `19`; observed error row `1`.
- 두 rigid-body result의 path ID/path/mass는 `0/empty/0` sentinel이었다.
- 응답은 약 `4.14ms/2.63ms`에 끝났으므로 timeout이 아니다.

Omni Physics 107.3의 “articulation links may not be instanced”와 설치 validator의
동일 규칙, runtime warning, joint-body failure, 직후 ERROR_PARSING이 모두 일치한다.
D339에서는 rigid-body owner가 non-instance이고 개별 collider geometry만 proxy여서
같은 property query가 VALID였다. D339도 app update pump `0`이었고 D373 callback
`68/68`이 먼저 끝났으므로 pump 부족이나 미완료 cook은 원인에서 제외된다.

### 8.3 canonical comparator population 사각지대 — 확정

D373은 root asset을 D345 `_stage_rows()`에 넘겼다. D345는
`Usd.PrimRange.Stage(stage)` 기본 순회를 사용해 instance proxy를 포함하지 않는다.
반면 D373 live inventory는 `Usd.TraverseInstanceProxies()`를 명시해서 정확한 34 part를
읽었다.

그 결과 canonical comparator는 base/variant에서 각각 35개 상위 row만 보고:

- base A64 path `0`
- variant P34 path `0`
- variant P34 Mesh `0`

을 잘못 보고했다. 보인 35개 row끼리의 outside-subtree hash가 같았다는 제한된 사실은
유효하지만, instance proxy 아래 전체 stage를 포함한 증명이 아니다. 실제 34-part live
inventory와 68 callback이 있으므로 이 zero count는 asset absence가 아니라 traversal
scope 실패다.

### 8.4 supervisor 권위 단절 — 확정

- supervisor: process return `0`, `pass=true`.
- worker raw summary/preclose sentinel: `worker_protocol_pass=false`.
- worker stdout도 `pass=false`를 출력했다.

supervisor가 worker summary verdict를 읽지 않고 return code만 성공으로 사용했기 때문에
정상 cleanup을 계약 PASS로 잘못 기록했다. D373 authority는 raw worker `false`다.

## 9. 왜 정상 분석·1920x1080·RRD를 실행하지 않았는가

사전등록은 numeric/binding failure를
`D373_P34_LIVE_ASSET_IDENTITY_FAIL_STOP`으로 매핑했다. 동결 controller의 `analyze()`는
`supervisor.pass`와 `worker_protocol_pass`가 모두 true일 때만 callback 수치 분류로
진입한다. 현재 worker가 false이므로 다음 항목보다 먼저 중단한다.

- authored↔callback surface/bounds/topology-volume classification
- property-volume matching
- exact 1920x1080 board
- save-only RRD/RBL/strict validation
- manual original-resolution inspection

현재 `--stage analyze`를 호출해도 board/RRD가 생기는 것이 아니라 exception만 추가된다.
`finalize()` 역시 존재하지 않는 evidence/visual/manual files와 identity PASS를 요구하므로
정직하게 실행할 수 없다. 따라서 둘 다 호출하지 않았고 attempt1을 동결했다.

시각화 상태는 다음과 같다.

- exact 1920x1080 board: `not_run/null_due_upstream_fail_stop`
- save-only RRD: `not_run/null_due_upstream_fail_stop`
- RBL: `not_run/null_due_upstream_fail_stop`
- manual inspection: `not_run/null_due_upstream_fail_stop`

실패를 성공처럼 보이게 하는 우회 그림이나 미등록 post-hoc gate는 만들지 않았다.

## 10. 범위 보존

실제 worker counter:

- `simulation_app_launches=1`
- `derivative_asset_materializations=1`
- `approved_collision_mesh_and_schema_authors=34`
- `physx_stage_attaches/detaches=1/1`
- `physx_callback_requests=68`
- `physx_property_queries=2`
- `worker_invocations=1`, `automatic_retries=0`

다음은 모두 `0`이다.

- SimulationContext construction, reset
- timeline play/commit
- controlled physics step, public forward
- q5 command/sample
- contact query
- cylinder create/write
- target/IK/path/pose change
- automatic convex-decomposition sweep
- inherited material/mass/actuator/physics-setting change
- Isaac Hydra render
- SimulationApp update pump

따라서 D373은 원통에 접근하거나 접촉하거나 잡는 시험을 하지 않았다. D362의 “현재 A64
경로에서 원통이 밀려 쓰러짐”만 물리 authority로 남는다.

## 11. 판정

Final operational verdict:

`D373_P34_LIVE_ASSET_IDENTITY_FAIL_STOP`

초보자용으로 풀면 다음과 같다.

- 새 P34 충돌체 34개를 USD와 PhysX callback까지 올리는 데는 성공했다.
- 하지만 “그 형상이 끝까지 같은지 인증하는 검사” 중 Float32 비교, whole-robot
  instancing property query, instance-proxy population audit, supervisor 판정 연결이 잘못됐다.
- 그래서 P34가 좋다/나쁘다, A64와 물리가 같다/다르다, 원통을 잡는다/못 잡는다는 결론은
  아직 내릴 수 없다.

다음 항목은 모두 `null`이다.

- full P34 live identity
- authored↔callback final surface/bounds/original-polygon topology-volume
- live property mass/COM/inertia/axes
- physics equivalence and runtime speed
- tipping causality
- grasp feasibility

`g0a_pass=false`다.

## 12. 다음 승인 경계

권장 최소 다음 case는 아직 미승인인 forward-only offline-only
`D374 [d373_fail_stop_provenance_and_failure_visualization]`다.

D374 범위 후보:

1. immutable D373 raw/USD만 읽는다; Isaac/PhysX worker는 0.
2. schema-typed Float32 scalar authority를 formalize한다.
3. default traversal과 instance-proxy traversal 범위를 분리한다.
4. whole-robot instancing/property-query incompatibility와 supervisor authority 단절을
   machine-readable evidence로 확정한다.
5. full identity/physics를 `null`로 표시한 실패 전용 exact 1920x1080 board와 save-only
   RRD/RBL을 만든다.

그 뒤에도 repaired live Isaac worker는 별도 case·별도 승인이다. repaired live identity가
PASS하기 전에는 A64/link5-only P34/gripper-only P34/both-P34 physical comparison을 실행하면
안 된다.

## 13. 주요 산출물과 해시

- preregistration: `f6cc93647c16ad441776846308601d797fcdcdf081ba2e57c0cec4b571b21e2d`
- supervisor: `3891bb51fbab02731edbea43e516048b6b4fac4b005e6bec5f27c5cabcb39643`
- raw summary: `dd57da307acf6134487bcd1dfa4a847fd41f24832177421f6291c45b06091373`
- preclose sentinel: `a32f6f423d0b7620a940d534e9f70bdc873fcdea90852d3731fdf6cc19bfa06a`
- stdout: `b5766ff871b552118f86049a5b6a38dec609be4e555d3748bc5794bea48ad43a`
- fail-stop attestation:
  `a47ea8600ddc74600644c2d747dd5f95861a2ecbcb2e0667ba0641e17f717206`
- D373 derivative physics layer:
  `1284fe48686baf1746d3a1537cb4774f3f32292f87fafb5eacf1e69772c8a9e8`

The fail-stop attestation is an append-only derived summary over immutable D373 originals. It is
not the absent normal controller completion summary and does not promote any null gate to PASS.
