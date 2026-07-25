# 2026-07-25 — Grasp G0a D385 P34 source-hull semantic low-count redesign

## 1. 무엇을 왜 확인했는가

D379에서 P34 복합 충돌체 34개 중 17개가 PhysX cook 뒤 원 작성 형상과
일치하지 않았다. D380은 이 17개가 바깥으로 부풀지 않고 안쪽으로 깎였음을
확인했고, D384는 실패부품을 원 형상 그대로 세밀하게 쪼개면 전체가
268개 또는 558개가 되어 현재 A64 참고후보 128개보다 작아야 한다는 프로젝트
목표를 통과하지 못함을 확인했다.

D385는 그중 profile prism 9개에 대한 D384의 정확한 46-child 수리는 그대로
동결하고, 일반 3-D source hull 8개만 의미 있는 얇은 층과 길이 방향으로 다시
나누어 더 적은 수의 충돌체 후보를 만들 수 있는지 묻는 offline-only 설계
실험이다. 실제 USD를 만들거나 Isaac/PhysX를 실행하는 실험이 아니다.

이번 case의 신규 변수:

- `source_hull_semantic_thin_layer_profile_cell_partition_v1`
- `source_child_max12_vertex_budget_v1`

이 실험은 등록된 분할법이 완전한 후보를 만들지 못하면 실제로 FAIL할 수
있었고, 실제로 그 경로로 종료됐다. 따라서 “판정을 바꿀 수 없는 단순
검증”이 아니었다.

## 2. 동결한 것과 사전등록한 판정 기준

동결 입력과 SHA-256:

- D372 geometry:
  `12fd1f32c35dfb9ae36cbbb412f6a51536aa1cc07c2dc17d05a5d189f3ee83e4`
- D372 evidence:
  `d68f658089aaf838ff454e9d0b301ec3f602785a3a730b3c329aa7785010e984`
- D379 evidence:
  `8eacbad796e8532c3d856b865e90dc54481f0f2003a266c3ebfaa8e93de37af5`
- D380 evidence:
  `4c64d08e117501dd15a5836ce56ef8b963d188044beac465e645e53a17710bd1`
- D384 evidence:
  `16ed5696d7198913367806e3ee13cf17a2b3f83c0c28d139115aa1d51c40822f`

동결 구조:

- 이미 통과한 P34 부품: 17개
- D384 profile-prism 정확 수리: 46개
- 변경 가능 대상: source 3-D hull 8개뿐

등록한 구성:

1. 각 부모 convex를 원 작성 좌표에 존재하는 얇은 층 경계에서 먼저 나눈다.
2. 각 층의 넓은 면 profile을 순서가 고정된 연속 fan cell로 나눈다.
3. profile cell을 원 부모 형상과 교차해 자식 convex를 만든다.
4. 자식 하나의 원 작성 꼭짓점은 최대 12개로 제한한다.

등록한 게이트:

- source 자식 수 `<=64`
- 전체 수 `17 + 46 + source_children <128`
- 바깥 돌출과 부모 표면 미포함 `<=0.1mm`
- topology volume 상대오차 `<=0.5%`
- 자식 사이 양의 부피 겹침 `0`
- owner/role과 D372 source-coverage, void, contact-seed 의미 보존

`12`, `64`, `<128`은 NVIDIA 기본값이나 엔진 한계가 아니라 이 프로젝트가
이번 case에 정한 설계 예산이다.

## 3. 실행 순서와 관찰

### 3.1 Git 및 입력 교차검사

- 승인 기준:
  `HEAD == origin/master == 35f10e3079b19e51209ba4cf1dd66391a431b053`
- 승인 시 worktree는 clean이었다.
- 다섯 입력 파일의 실제 SHA-256은 모두 등록값과 일치했다.
- AST 검사에서 Isaac, Kit, PhysX, USD, Warp, CUDA/Torch 실행 import는
  없었다.

### 3.2 attempt1 준비단계 정지

경로:

`claudedocs/runtime_logs/grasp_track/g0a_d385/attempt1_semantic_axis_slab_low_count_redesign/`

attempt1은 출력 폴더를 만든 뒤 `git status --short`를 읽었다. 따라서 자기
자신이 방금 만든 `g0a_d385/`를 예상 밖 변경으로 잘못 판정했다.

- prepare PASS: false
- actual worker: 0
- 설계 계산: 0
- Rerun Viewer: 0
- Isaac/PhysX/physics/q5/contact: 0

이 경로는 덮어쓰지 않고 동결했다.

### 3.3 attempt2 준비순서 수리

경로:

`claudedocs/runtime_logs/grasp_track/g0a_d385/attempt2_precreate_git_status_capture_repair/`

과학/설계 변수는 바꾸지 않았다. 출력 폴더를 만들기 전에 Git 상태를 먼저
읽고, 폴더 생성 뒤에도 porcelain root가 새로 늘지 않았는지만 확인하도록
순서만 고쳤다.

- preregistration checks: 모두 PASS
- worker: 정확히 1회
- retry: 0
- return code: 0
- elapsed: `4.530759936897084s`
- watchdog: `300s`, timeout=false
- TERM/KILL: 없음
- 잔류 process-group member: 0

## 4. 정량 결과

8개 부모 중 완전 분할을 찾은 것은 4개다.

| body | 부모 | 역할 | 부모 꼭짓점 | 자식 | 자식 최대 꼭짓점 | 결과 |
|---|---|---|---:|---:|---:|---|
| gripper_link | proximal_upper_arm_hull_b | moving support | 34 | 4 | 12 | PASS |
| gripper_link | proximal_lower_arm_hull_b | moving support | 34 | 4 | 12 | PASS |
| gripper_link | moving_upper_backbone | moving-jaw backbone | 20 | 6 | 12 | PASS |
| gripper_link | moving_lower_backbone | moving-jaw backbone | 20 | 6 | 12 | PASS |

성공한 네 부모의 최대 바깥 오차는
`0.0000012291007170373014mm`, 최대 부모 표면 미포함 오차는
`0.0000006143987332785095mm`, 최대 volume 상대오차는
`1.3628896866106088e-7`이었다.

다음 4개는 “최대 12꼭짓점의 연속 fan-cell”로 해당 얇은 층을 완전히 덮는
조합이 없었다.

| body | 부모 | 실패 층 | 기각된 등록 후보 |
|---|---|---|---:|
| gripper_link | proximal_upper_arm_hull_a | `z_layer_00` | 10 |
| gripper_link | proximal_lower_arm_hull_a | `z_layer_01` | 12 |
| link5 | fixed_backbone_left | `y_layer_01` | 4 |
| link5 | fixed_backbone_right | `y_layer_00` | 10 |

따라서:

- 부분적으로 생성된 source 자식: 20개
- 완전한 8-parent source 자식 수: `null`
- 완전한 전체 충돌체 수: `null`
- `<128` 판정: 미평가
- offline design pass: false
- verdict:
  `D385_SEMANTIC_THIN_LAYER_PROFILE_CELL_NO_ADMISSIBLE_CANDIDATE_FAIL_STOP`

부분 20개는 실패 위치를 보여주는 진단 자료일 뿐이며 USD에 올릴 수 있는
완성 후보가 아니다.

## 5. 실행하지 않은 것

원 JSON의 모든 현재-scope counter는 0이다.

- asset/USD read/write 및 collider materialization/regeneration
- Isaac/Kit/PhysX launch와 live callback query
- automatic decomposition sweep
- cylinder create/write
- controlled physics step
- q5 sample 및 contact query
- target/IK/path 변경
- Warp/CUDA launch

따라서:

- `repair_materialized=false`
- `live_identity_pass=null`
- `live_gpu_compatibility_pass=null`
- `physics_or_grasp_result=null`
- `cylinder_29x50_rendered_or_measured=false`
- `p34_authored_to_cooked_identity_pass=false`
- `g0a_pass=false`

D385 FAIL은 Isaac Sim 실패, 물리 실패, 또는 원통 파지 실패가 아니다. 아직
실제 충돌체와 원통을 넣지 않은 오프라인 형상 설계 FAIL이다.

## 6. 시각화와 관찰

- 정확한 1920×1080 비교판:
  `d385_source_hull_redesign_1920x1080.png`
  (`1920x1080`, SHA-256
  `f397a407e699d0ee6881c26fd377107fc37774840f4e1a2fac6eb0c3cf3cdb32`)
- save-only recording:
  `d385_source_hull_redesign.rrd`
  (SHA-256
  `1be6eded1a1d45641dbd078d6230e47c5dcdd4c8c37cc43b0684844bebf8cb3a`)
- embedded blueprint:
  `d385_source_hull_redesign.rbl`
  (SHA-256
  `4f3647ca42c38e62bfdd0ad862292fded44156e0df723670770681574cd353a3`)
- Rerun inspection:
  `d385_rerun_inspection.png`
  (native HiDPI `3840x2160`, SHA-256
  `19ce633a390d2e6d19702814d0e8846591b62622d5c5f98a2e1f97a8192cc8b8`)

비교판에서 8개 부모가 모두 보이며, 실패 4개는 회색 원 부모만, 성공 4개는
색상 자식 조각과 함께 보인다. 하단은 부분 20개, 전체 count `NULL`, 전체
설계 FAIL을 명시한다. 수동검사 `7/7`과 RRD/RBL footer, exact entity,
timeline, component 계약이 통과했다.

Rerun 캡처 오른쪽에 restricted sandbox의
`message proxy server crashed: Operation not permitted` 알림이 보였으나
형상과 판정 패널을 가리지 않았다. 이는 RRD load/geometry 실패가 아니며,
RRD/RBL verify와 실제 형상 표시가 통과했다.

## 7. NVIDIA 공식문서 및 설치본 교차검사

적용 버전:

- Isaac Sim `5.1.0.0`
- Isaac Lab `2.3.0`
- PhysX schema/property extension `107.3.26 + Kit 107.3.3`
- Rerun SDK/CLI `0.34.1`

공식 자료:

1. [Omni Physics 107.3 — Rigid Bodies](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/rigid_bodies.html)
   — 한 rigid body 아래 여러 collider를 두는 공식 구조.
2. [Omni Physics 107.3 — Colliders](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/collision.html)
   — primitive와 mesh collider, convex hull/decomposition, cooking 의미.
3. [PhysX 5.6.1 — GPU Simulation](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/GPURigidBodies.html)
   — GPU contact용 convex는 최대 64 vertices, 64 polygons,
   32 vertices/face 경계와 GPU cooking data를 함께 요구.
4. [PhysX 107.3 source — `ConvexMesh::isGpuCompatible()`](https://raw.githubusercontent.com/NVIDIA-Omniverse/PhysX/107.3-omni-and-physx-5.6.1/physx/source/geomutils/src/convex/GuConvexMesh.cpp)
   — vertices/polygons/face width뿐 아니라 GPU edge data와
   extent-radius ratio도 함께 검사.

설치 파일:

- `PhysxSchema/resources/schema.usda:858,886,895`:
  convex-hull/decomposition `hullVertexLimit=64`,
  `maxConvexHulls=32`.
- `omni/kit/property/physx/database.py:954-957`:
  UI authoring ranges `8..64`, `1..2048`.

### post-completion marker correction

D385 evidence의 `installed_stack.schema_markers` 네 값은 false지만, 이는
설치 설정이 없다는 뜻이 아니다. 검사 코드가 USD 실제 문법 `= 64`/`= 32`
대신 `default = 64`/`default = 32`, Python 실제 문법
`InfoData(8, 64, 1)` 대신 `range=(8, 64)`를 찾은 문자열 matcher
false negative다.

이 네 diagnostic boolean은 D385 설계 gate나 verdict 계산에 사용되지 않았다.
위 설치 파일 직접 확인값이 권위이며, 향후 이 evidence field를 NVIDIA 설정
부재의 근거로 사용하지 않는다.

또한 자식당 12꼭짓점만으로 live GPU compatibility를 증명할 수 없다. 실제
PhysX cook 뒤 polygon/face/edge-data/shape-ratio callback을 읽어야 한다.

## 8. 일상어 결론과 다음 승인 경계

이번에 시험한 한 가지 분할법은 8개 어려운 구조물 중 절반만 깔끔하게
나눴다. 나머지 절반은 “조각 하나당 꼭짓점 최대 12개”를 지키면서 빈틈과
겹침 없이 나누지 못했다. 그러므로 “전체 몇 개로 줄였다”는 완성된 숫자는
아직 없다.

다음 권장 최소 후보는 미승인
`D386 [d385_minimum_admissible_vertex_budget_localization]`이다. 분할법과
형상 오차·무겹침·의미 게이트는 그대로 두고, 실패한 네 층이 실제로
필요로 하는 최소 child-vertex budget만 오프라인에서 찾는다. 그 결과를 본
뒤 한 개의 새 budget을 선택해야 한다.

D386, 다른 분할법, 12 제한 완화, 내부 겹침 허용, 자동분해 sweep,
direct-polygon bridge, USD materialization/live identity, 새 29×50mm 원통
authoring/render, physics/q5/contact/grasp는 모두 별도 명시 승인 전에는
실행하지 않는다.

