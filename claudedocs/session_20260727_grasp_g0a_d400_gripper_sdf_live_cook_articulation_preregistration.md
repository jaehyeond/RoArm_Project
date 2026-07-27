# D400 gripper_link SDF 구성·load admission·owner enumeration preflight 사전등록

Date: 2026-07-27 KST  
Case: `g0a_d400`  
Attempt reserved:
`attempt1_gripper_link_sdf_res256_live_cook_articulation_preflight`  
Status: `PREREGISTERED_NOT_EXECUTED`

위 attempt 폴더명은 이미 예약한 forward-only 경로라 유지한다. 이름 속
`live_cook_articulation`이 V2의 증명 범위를 넓히지는 않는다.

## 1. 무엇을 왜 하는가

D400은 RoArm의 두 턱을 한꺼번에 바꾸는 실험이 아니다. 다음 한 변수만
검사하도록 범위를 줄였다.

`이번 case의 신규 변수:
[gripper_link_collision_representation_a64_to_sdf_res256_v1]`

- 고정 턱이 속한 `link5`는 기존 A64 충돌체 64개를 그대로 유지한다.
- 움직이는 턱이 속한 `gripper_link`만 A64 64개에서 SDF 입력 메시 1개로
  바꾼다.
- SDF 해상도는 설치 schema 기본값과 같은 `256` 한 값만 사용한다.

SDF(signed distance field, 공간의 각 점에서 표면까지 거리를 격자로
저장하는 충돌 표현)가 현재 관절 링크의 충돌 입력으로 정확히 작성되고
PhysX stage에 수용·열거되는지 확인하지 않은 채 원통 접촉으로 넘어가면,
접촉 실패가 SDF 구성 실패인지 파지 자세 실패인지 분리할 수 없다. 따라서
D400은 접촉 전에 **입력 구성, stage load, 전역 cook queue 종료, rigid-link
owner 열거**까지만 확인하는 사전검사다. 내부 SDF shape의 실제 접촉 참여는
D400의 증명 범위가 아니다.

## 2. 이번 승인으로 실제로 한 일

사용자의 “정정된 연구 순서로 진행” 승인은 승인 범위를 넓히지 않고
**D400 사전등록 작성·검토**로만 소비했다. 검토 결과, 재현 가능한 실행
파일의 해시를 보지 않은 채 곧바로 worker를 승인받는 것도 위험하다고
판단해 이후 경계를 세 단계로 더 명확히 분리했다.

1. 이번 턴: 사전등록 작성·검토
2. 다음 별도 승인: 두 실행 파일, 정적 검토 attestation, proposed runtime
   hash tuple만 작성
3. 그 결과를 보고 다시 별도 승인: Isaac/PhysX worker 정확히 1회 실행

1. `AGENTS.md`의 Current-State Protocol에 따라 상태 문서와 Git을
   재확인했다.
2. `HEAD == origin/master ==
   4c88865bdd4ac82f034253320cb3e46f9770a46d`를 확인했다.
3. 기존 dirty worktree의 사용자 소유 문서 변경을 보존했다.
4. D344 A64 자산, D368 원본 메시 계보, D373-D378의 live owner·supervisor
   실패 교훈을 다시 읽었다.
5. 설치된 PhysX schema와 NVIDIA 107.3 SDF 작성 계약을 교차검사했다.
6. 아래 사전등록 JSON을 새 forward-only D400 경로에 작성했다.
7. 세 독립 검토가 API 적용 위치, property-query 권위, cook 통계의 한계,
   worker 수명주기와 Rerun 분기를 적대적으로 검토했다.
8. 발견된 거짓 PASS 가능성을 사전등록 V2에서 수리했다.
9. exact Rerun entity/component/timeline 계약과 script-hash 선행 경계를
   추가한 뒤 JSON 구문·중복키·동결 해시를 다시 검사했다.

이번 턴에는 코드, 파생 USD, Isaac/Kit/PhysX 프로세스, GPU 계산, q5,
원통, 접촉, 물리 step을 실행하지 않았다.

사전등록:
`claudedocs/runtime_logs/grasp_track/g0a_d400/attempt1_gripper_link_sdf_res256_live_cook_articulation_preflight/d400_preregistration.json`

## 3. NVIDIA 공식 계약과 설치본 교차검사

설치 기준:

- Isaac Sim `5.1.0.0`
- Isaac Lab `2.3.0`
- PhysX schema / omni.physx `107.3.26`
- PhysX engine `5.6.1`은 설치 계보에서의 추론이며, runtime에서 다시
  기록해야 한다.

NVIDIA의 SDF 예제는 실제 `UsdGeom.Mesh` prim에 다음 세 항목을 함께
적용한다.

1. `UsdPhysics.CollisionAPI`
2. `UsdPhysics.MeshCollisionAPI`와 `approximation="sdf"`
3. `PhysxSchema.PhysxSDFMeshCollisionAPI`

공식 자료:

- [Omni Physics 107.3 — Colliders](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/collision.html)
- [Omni Physics 107.3 — Collision Behavior Guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/collision_guide.html)
- [Omni Physics 107.3 — Rigid Bodies](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/rigid_bodies.html)
- [Isaac Sim 5.1.0 — Physics Simulation Fundamentals](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/physics/simulation_fundamentals.html)

설치 schema 근거:

- `PhysxSchema/resources/schema.usda:1043-1141`
- schema SHA-256:
  `fe075bce4bde5ba7db69c6ccef0c4c26909336ab34c619129fc276f7cb4d7abc`
- `omni.usd.schema.physx` extension SHA-256:
  `595c2af276510d9584fafe952fa0ad09ec26869c5441a8c8a48ab1df514d7924`
- core `omni.physx` extension SHA-256:
  `6c9d9ed33d927e302334b7cae8ed0c81c4fba37bfbfac07053b72ccc16b7398f`
- private query binding `_physx.pyi` SHA-256:
  `ff13abb83480dcc707ac2ad60062306aef7a33f885d32ed4c8ee6dfea2008e79`
- `omni.physx.cooking` extension SHA-256:
  `89e9f38426bcf632b6cdfe2f89e29e4ed5d70810c55baef6b1b66860fa274b58`
- 설치 NVIDIA triangle-mesh collision test SHA-256:
  `46dcafcf790db0b2f4f3333236c8db7e98a8510326370dc49e3fae0bb822f4d3`

명시적으로 작성하고 typed readback할 일곱 값:

| 속성 | D400 값 | 뜻 |
|---|---:|---|
| `sdfResolution` | `256` | 격자 간격 = 메시 AABB의 가장 긴 길이 ÷ 256 |
| `sdfSubgridResolution` | `6` | 희소 SDF의 세부 격자 분해도 |
| `sdfBitsPerSubgridPixel` | `BitsPerPixel16` | 희소 세부 격자의 거리값 저장 정밀도 |
| `sdfNarrowBandThickness` | `0.01f` | bounding-box 대각선에 대한 무차원 비율; 0.01m가 아님 |
| `sdfMargin` | `0.01f` | bounding-box 대각선의 1%만큼 계산 영역 확장 |
| `sdfEnableRemeshing` | `false` | 자동 재메시 금지 |
| `sdfTriangleCountReductionFactor` | `1.0f` | 삼각형 축소 금지 |

`sdfMargin=0.01`은 SDF 영역 경계를 메시 bounding-box 대각선의 1%만큼
각 방향으로 넓힌다는 뜻이다. 0-등위면 또는 실제 충돌 표면 자체를 1%
부풀렸다는 뜻은 아니다.

설치된 NVIDIA test에서 가져온 **process-wide 전역 cook queue 진단식**은
다음과 같다.

`running_tasks = total_scheduled_tasks - total_finished_tasks`

그러나 이 값은 특정 prim이나 SDF를 식별하는 공개 API가 아니다. 단순히
처음과 끝이 모두 `0`이면 아무 cook도 일어나지 않은 상태도 거짓 PASS할 수
있다. 그래서 V2는 attach 전에 scheduled/finished/cache hit/cache miss를
기록하고 baseline running task가 `0`인지 확인한다. attach 뒤에는
`scheduled_delta>0`, `finished_delta==scheduled_delta`, 최종 running task
`0`을 모두 요구한다. 그래도 증명 가능한 것은 “이 stage에서 전역 cook
작업이 실제 발생해 제한 시간 안에 끝났다”까지다. 특정 SDF 내부 객체와 실제
접촉 shape 생성은 여전히 `null`이다.

또한 설치 NVIDIA test는 시작할 때 process-global local mesh cache를
비운다. D400은 다른 세션/프로세스 상태를 바꾸지 않기 위해 그 cache를
지우지 않는다. 따라서 공식 test 전체를 그대로 상속한 것이 아니라 **카운터
계산식만 참고**하며, cache 때문에 `scheduled_delta=0`이면 PASS가 아니라
`INCONCLUSIVE_FAIL_STOP`으로 처리한다.

Kit update를 진행하면서 이 조건을 기다리되, D400에서는 전체 300초·무진행
60초 watchdog 안에서만 허용한다. update pump는 별도 계수하며 physics
step으로 세지 않고 최대 `300,000`회로 막는다. timeline은 전 구간
STOP/time 0이고 physics step은 0이다.

## 4. 사용할 메시와 최소 변경

D400은 P34나 D397의 미완성 분할 조각을 사용하지 않는다. D344 자산에
이미 들어 있는 원본 전체 `gripper_link` 메시를 그대로 사용한다.

- STL:
  `local_assets/roarm_m3/urdf/meshes/gripper_link.stl`
- STL SHA-256:
  `7946a374e24a2f467a0581b4946e0ec41b1b86a92f070bc00aa9bced1bf65a56`
- USD source Mesh:
  `/roarm_m3/gripper_link/collisions/gripper_link/node_STL_BINARY_/mesh`
- live Mesh:
  `/World/Robot/gripper_link/collisions/gripper_link/node_STL_BINARY_/mesh`
- 꼭짓점 stream: `41,094`
- 삼각형: `13,698`
- native USD type: points `point3f[]`, counts/indices `int[]`
- native points/counts/indices 결합 SHA-256:
  `31aead25f7aa879a358a046bc01291ef2e260a2b367a990dacc255c17a2a5a31`
- body-local bounds:
  `[-0.0107673974,-0.0099996324,-0.0386173457]`부터
  `[0.0670826066,0.0152403677,0.0007502218]m`

Kit를 띄우지 않는 installed standalone USD Python binding으로 D344
prototype을 직접 읽은 결과도 다음과 일치했다.

- type `Mesh`
- `orientation=rightHanded`이며 authored opinion은 없음(USD fallback)
- `subdivisionScheme=none`이며 authored opinion이 존재
- `holeIndices` authored opinion 없음, effective `[]`
- USD `material:binding` relationship target 1개:
  `.../node_STL_BINARY_/Looks/DefaultMaterial`

현재 D344 자산을 standalone USD로 읽으면 `collisions` scope가
instanceable이고 이 Mesh는 instance proxy다. D373의 실패를 반복하지
않기 위해 파생 자산에서 정확히 `/roarm_m3/link5/collisions`와
`/roarm_m3/gripper_link/collisions` 두 scope에만 `instanceable=false`를
쓴다. `SetActive(false)`나 prim deactivate는 쓰지 않는다. 이 조치는 메시
좌표나 물성의 과학 변수가 아니라, 등록된 prim을 직접 검사하기 위한
상속된 stage-structure 수리다.

허용하는 변경:

1. `gripper_link/d338_convex_parts/part_000..063`의 정확히 64개에
   `physics:collisionEnabled=false` 작성; deactivate 금지
2. 기존 parent Xform collider는 비활성 상태 유지
3. 원본 Mesh leaf에 `physics:collisionEnabled=true`, 세 SDF API,
   `approximation="sdf"`, 일곱 속성 적용
4. 위 두 collision scope의 instanceability만 해제

이 exact allowlist 밖의 base→derivative composed semantic diff는 `0`이어야
한다. 즉 link5 메시·충돌 enable 상태, 조상 transform, 물성, joint/drive,
physics scene은 그대로여야 한다.

금지하는 변경:

- `link5` A64
- points/counts/indices와 조상 transform
- mesh repair, vertex deduplication, face reorder, remesh, triangle reduction
- material, mass, COM, inertia, actuator, joint, drive, physics scene
- P34/D397 geometry

native USD Float32/Int32 stream과 기존 mm→m transform을 함께 보존하고,
`orientation=rightHanded`, `subdivisionScheme=none`, unauthored
`holeIndices=[]`, 단일 USD material binding도 exact readback한다(이
relationship 자체가 PhysX per-face material 배정을 증명하는 것은 아니다).
별도로 D368
Float64 meter body-local canonical stream도 재계산한다. 따라서 서로 다른
dtype 계보를 한 hash로 섞거나 단위를 두 번 변환하는 오류를 금지했다.

## 5. D400의 판정 권위

### PASS에 직접 쓰는 근거

- 입력 SHA-256
- runtime 직전 HEAD·dirty 경로·사전등록·두 실행 스크립트 해시 manifest
- source→파생 authored→live points/counts/indices/mesh semantics/transform
  typed exact와 allowlist 밖 semantic diff `0`
- active inventory:
  `link5 A64=64`, `gripper A64=0`, `gripper SDF=1`
- 새 collider가 실제 `UsdGeom.Mesh`이고 올바른 세 API·`sdf` token·일곱
  typed 값과 `physics:collisionEnabled=true` 보유
- Robot, 두 rigid owner, SDF Mesh가 non-instance/non-proxy
- `attach_stage=True`라는 stage load-admission
- 전역 cook queue의 baseline running `0`, scheduled delta 양수,
  finished delta 일치, 최종 running `0`
- 다음 property-query 예상 계약을 실제 worker가 completed callback과
  exact row/path set으로 검증:
  link5 `65`행
  (`54dd23cb24d9c85d505fd9c44708248e8715b904f4ea4d275d7992ab21ef7a5a`),
  gripper `66`행
  (`7b23094e24a7f574e3d0cdc7057f6510817ba6cd47ccc3ba319b27dce4fe2821`)
- gripper query에 exact SDF Mesh path가 enabled 상태로 존재
- attach 시작부터 pre-close까지 ERROR/FATAL 0, SDF/cook/collision/fallback/
  rigid-body/articulation 관련 WARN 0인 Kit log
- authored USD 물성 exact, live 물성은 D375 callback 기준의 field-specific
  tolerance 내 불변
- raw summary SHA를 담은 pre-close sentinel과 외부 supervisor가 일치
- worker 기술 PASS 뒤에만 RRD/RBL/검증/정적 1920×1080 decision board/
  실제 Rerun Viewer 1920×1080 screenshot/receipt/육안검사 완료

live tolerance는 mass `1e-12kg`, COM `1e-12m`, diagonal inertia
`5e-13kg·m²`, quaternion `1e-12`의 absolute tolerance와 공통 relative
`1e-7`로 분리했다. PhysX callback의 principal axes는 `xyzw`이므로
비교 전에 `[q3,q0,q1,q2]`로 `wxyz`에 변환한다.
같은 회전을 나타내는 `q`와 `-q`가 거짓 FAIL을 만들지 않도록, 첫
비영(절댓값 `>1e-15`) 성분이 양수가 되게 부호를 canonicalize한 뒤
성분을 비교한다.

여기서 `65/66`은 아직 D400 실측 결과가 아니다. D375는 비활성 legacy
collider도 질의에 나타날 수 있음을 보였지만, 새로 비활성화할 A64 64개
전부와 새 SDF Mesh가 정확히 이 행수로 반환되는지는 D400 worker가 처음
검증한다. 따라서 불일치하면 기대값을 실행 뒤 고쳐 PASS시키지 않고
사전등록대로 FAIL_STOP한다.

### 진단으로만 남기는 값

- source/live USD bounds
- 원본 mesh의 topology·void
- exact SDF path property query의 유한 AABB·양수 volume
- 예상 격자 크기
- cook cache hit/miss
- Rerun의 Float32 공간 사본

### 반드시 `null`로 남길 값

- 실제 cooked SDF 0-등위면의 mm 오차
- cooked SDF가 jaw의 빈 공간을 보존했는지
- 고정 턱과 움직이는 턱 사이 OPEN clearance
- 실제 접촉 참여·접촉력·마찰·미끄러짐·전도·파지·들기
- 29×50mm 제품 적합성
- CPU/GPU 실제 충돌 경로

`scientific/physics verdict=null`과 별개로 프로젝트 전역
`g0a_pass=false`는 유지한다. `g0a_pass`를 null과 false 양쪽에 놓는 모순은
V2에서 제거했다.

특히 convex collider에서 사용했던 PhysX 원 polygon callback과
topology-volume gate는 SDF cooked-grid 권위가 아니므로 D400에
재사용하지 않는다. `VALID` property query도 내부 SDF grid나 실제 접촉
참여를 증명하지 않는다. 또한 D400은 q5와 공통 world pose를 읽지 않으므로
두 턱 사이 OPEN clearance는 D402 zero-step 자세 case로 이연한다.

## 6. 구현 선행 경계와 이후 실행 계약

실제 실행 전에 별도 승인을 받아 다음 네 파일만 작성·정적 검토해야 한다.

- `sim_scripts/cyl34_top_view_d400_gripper_link_sdf_res256_live_cook_articulation_preflight.py`
- `sim_scripts/cyl34_top_view_d400_gripper_link_sdf_res256_live_cook_articulation_worker.py`
- `d400_reviewed_script_attestation.json`
- `d400_proposed_runtime_hash_tuple.json`

이 선행 단계에서는 Isaac/Kit/PhysX를 import·launch하지 않고, 파생 USD도
만들지 않는다. 두 script의 검토된 SHA-256을 attestation에 고정한 뒤,
사전등록·attestation·controller·worker 네 SHA-256을 proposed tuple에
고정하고 그 tuple 파일 SHA를 사용자에게 보고한다. 이후 runtime 승인은 그
tuple SHA를 명시해야 하며 case 이름만 적은 일반 승인은 부족하다. runtime
manifest는 controller의 `--approved-tuple-sha256`, tuple 파일 SHA, tuple
속 네 값, 현재 네 파일 hash가 모두 정확히 같아야 worker를 spawn한다.

그 이후 실제 실행까지 별도 승인되면 다음 순서로 한 번만 진행한다.

1. controller가 runtime 직전 HEAD/origin, exact dirty path+hash,
   사전등록과 두 future script hash를 `d400_runtime_freeze_manifest.json`에
   고정한다. 예기치 않은 dirty path가 하나라도 있으면 worker 전에 멈춘다.
2. 패키지 pin, GPU, 기존 프로세스와 오프라인 음성 대조군을 검사한다.
3. headless Isaac worker를 정확히 1회 spawn하고 retry는 0으로 둔다.
4. worker가 `SimulationApp` launch 시작/종료를 표식한다.
5. worker 안에서만 D344 자산을 D400 경로로 1회 복사하고 등록된 SDF
   opinion을 쓴 뒤 authored typed readback을 완료한다.
6. non-instance live owner와 active collider inventory를 검사한다.
7. raw timeline이 STOP/time 0임을 기록하고 PhysX stage를 1회 attach한다.
8. attach 전/후 전역 cook counters를 기록하며 bounded update pump로
   nonzero 작업 발생과 drain을 확인한다.
9. link5/gripper property query를 각 1회 실행하고 모든 callback row,
   exact path set, completed callback을 보존한다.
10. authored/live 물성을 판정한다.
11. cleanup을 시작해 detach 1회 → StageCache erase 1회 → non-membership을
    먼저 확인한다.
12. 그 최종 counter와 cleanup 결과를 포함한 raw summary를 쓰고, 그 SHA와
    `safe_to_close_app`를 담은 pre-close sentinel을 쓴다.
13. `SimulationApp.close()`를 호출한다. D375/D377의 terminal-close
    계보를 상속해 `close_returned` 표식은 선택 진단일 뿐 PASS 필수값이
    아니다.
14. 외부 supervisor가 attach-start부터 pre-close까지의 authoritative
    Kit-log window를 감사하고, return code뿐 아니라 내부
    PASS·SHA·counter·erase·잔류 프로세스를 함께 판정한다.
15. worker 기술 PASS일 때만 save-only RRD finalize/verify/RBL,
    정적 decision board와 실제 Rerun Viewer screenshot을 각각 정확한
    1920×1080으로 만들고 원해상도 육안검사를 한다.

전체 watchdog는 300초, 무진행 watchdog는 60초다. timeout이면 이 case가
만든 process group만 종료 대상으로 삼고, 기존 Isaac GUI나 다른 GPU
프로세스는 건드리지 않는다. 기존 process command가 D400 future script명
또는 exact output root를 포함하는 **충돌 프로세스**일 때만 fail-closed한다.
그 밖의 Isaac/Kit/GPU 프로세스는 기록하되 존재만으로 실패시키지 않는다.
free VRAM은 최소 `8192MiB`를 요구한다.

Rerun 계약도 범위 문자열로 뭉뚱그리지 않았다. `preflight_phase` timeline의
`0/1/2`는 각각 source baseline/live configuration/post-query decision이며,
정확한 non-system entity는 73개다. link5 A64는
`part_000`부터 `part_063`까지 64개 exact path를 요구하고, 각 Mesh3D·
Transform3D component의 실제 component name까지 검증한다. source는
phase 0, live Mesh·link5 64개·API/owner 상태는 phase 1, cook·물성·최종
counter는 phase 2에 정확히 한 row씩만 허용하고 timeless/다른 phase row는
0으로 둔다. 정적
decision board와 실제 Viewer screenshot은 서로 다른 1920×1080 파일이다.

worker가 기술적으로 FAIL하면 Rerun을 시작하지 않는다. 이때 Rerun 파일이
없는 것은 observability FAIL이 아니라 의도된 fail-stop이다. worker 기술
PASS 뒤 Rerun 계약만 실패한 경우에만
`D400_OBSERVABILITY_OR_COMPLETION_INTEGRITY_FAIL_STOP`으로 분리한다.

## 7. 실패할 수 있는 대조군

worker를 늘리지 않고 실행 전 오프라인 계약 검사로 다음 고장을
의도적으로 거부한다.

- source hash 또는 point/index/count 한 값 변경
- `sdf` 대신 `convexHull`
- 필수 API 하나 누락
- resolution `255`, `257`, `512`
- remeshing `true`, reduction `!=1.0`
- SDF API를 Mesh가 아닌 Xform에 적용
- instance proxy owner
- gripper A64와 SDF 동시 활성
- SDF Mesh의 `collisionEnabled=false`
- link5 A64가 64가 아님
- query row count/path-set/completed callback/SDF exact path 불일치
- 전역 cook counter의 `0→0` 거짓 PASS
- mass/COM/inertia 변경
- 내부 FAIL인데 process return code만 0
- RRD footer 잘림

이 대조군은 “검증 코드가 실제 잘못된 상태를 거부할 수 있는가”를 확인하기
위한 것이며, SDF 접촉 실험은 아니다.

## 8. 적대적 검토에서 발견하고 고친 핵심 오류

초안은 그대로 실행하면 다음 거짓 결론을 낼 수 있었다.

1. cook counter가 `0→0`인데도 “cook 완료”라고 판정할 수 있었다.
2. property query가 단순 `VALID`인 것만으로 exact SDF path가 열거됐다고
   오판할 수 있었다.
3. A64 제거 방법이 deactivate 또는 collision-disable 두 방식으로 열려
   재현성이 없었다.
4. worker cleanup 전에 raw/pre-close를 써서 detach/cache erase 결과가
   sentinel 권위 밖에 남았다.
5. worker 기술 FAIL인데도 Rerun 파일을 무조건 요구해 실패 원인을 섞었다.
6. `g0a_pass`가 `null`과 `false`에 동시에 들어 있었다.
7. “articulation attachment”라는 표현이 zero-step evidence보다 강했다.
8. 일반 USD 시각 재질 binding을 PhysX 물리 재질로 부를 위험이 있었다.
9. `orientation`/`subdivisionScheme`의 값만 보고 authored 여부를 빼먹었다.
10. `65/66`을 아직 측정하지 않은 기대값이 아니라 완료 결과처럼 읽을 수
    있었다.
11. 실행 파일을 작성·검토하기 전에 곧바로 worker 승인을 받을 수 있었다.
12. Rerun의 `part_000..063`, component, phase 표현이 exact 계약이 아니었다.

V2는 각각 nonzero cook delta, exact query row/path hash, per-part
`collisionEnabled=false`, cleanup-before-summary, 기술/관측 분기,
`scientific verdict=null` 대 `g0a_pass=false` 분리, “structural owner
enumeration”, 일반 USD material binding 명칭, authoredness, 기대값 경계,
script-hash 선행 승인, exact Rerun path/component/timeline 계약으로
교정했다. 따라서 V2는 초안보다 증명 범위가 좁지만 거짓 PASS 가능성도
낮다.

## 9. 사전등록 결과와 해석

- 사전등록 JSON 작성: 완료
- JSON 구문 검사: PASS
- 사전등록 JSON SHA-256:
  `fc689cb1afd6108a326a73f22b8117dfdefc0bb4d8caee5bcb7470c362e96c93`
- 신규 변수: 정확히 1개
- runtime 실행: `false`
- Isaac/PhysX launch: `0`
- physics/q5/contact/cylinder: 모두 `0`
- scientific/physics verdict: `null`
- `g0a_pass=false`

D400 V2의 미래 PASS 문구:

`D400_GRIPPER_LINK_SDF_RES256_CONFIGURATION_LOAD_ADMISSION_OWNER_ENUMERATION_PREFLIGHT_PASS_NO_PHYSICS`

이는 “SDF 입력과 속성을 정확히 작성했고, stage가 수용됐으며, 전역 cook
작업이 실제 발생해 끝났고, exact Mesh path가 의도한 rigid-link query에
열거됐다”까지만 뜻한다. 특정 SDF 내부 shape를 직접 확인했다거나 관절
충돌에 실제 참여했다는 뜻이 아니며, “원통을 잡는다”는 뜻은 더더욱 아니다.

## 10. 이번 세션에 실험을 실행하지 않은 이유

AGENTS.md의 Session progress rule에 따라 실험 부재를 명시한다. 현재 승인
경계는 preregistration 작성·검토까지만 허용한다. 다음은 코드,
reviewed-hash attestation, proposed runtime hash tuple만 만드는 no-Isaac 단계이고, 실제 Isaac/PhysX
worker는 그 결과 뒤 다시 별도 명시 승인을 요구한다. 승인 범위를 넓혀
실패 가능 runtime을 임의로 실행하는 것은 Variable Ladder와 사용자 승인
경계를 위반하므로 이번 세션에는 실험을 실행하지 않았다. 나중의 actual
worker 단계에서 위 음성 대조군과 one-worker runtime이 실패 가능한 검증이
된다.

## 11. 다음 승인 경계

아직 승인되지 않은 다음 동작은 **D400 no-Isaac 구현·정적 attestation
단계**다. 정확히 두 script, reviewed-script attestation, proposed runtime
hash tuple만 작성하고 파생 USD·Isaac/PhysX는 실행하지 않는다. 그 검토가
PASS하면 네 파일 hash와 tuple 파일 SHA를 보고한다. 사용자가 그 tuple SHA를
명시해 승인한 뒤에만 actual worker 1회 실행이 허용된다. actual worker가 PASS해도
바로 제품 원통을 사용하지 않고, 별도 승인 D401에서 동일한 비제품 box
양성대조로 A64와 SDF의 실제 관절 접촉 반응을 먼저 확인한다.

commit/push는 수행하지 않았다.
