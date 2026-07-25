# Session 2026-07-25 — Grasp G0a D386 minimum admissible vertex-budget localization

## 1. 무엇을 왜 확인했는가

D385는 같은 semantic thin-layer/profile-cell 분할법으로 source 3-D hull
8개 중 4개만 완전 분할했다. 나머지 네 부모는 각각 첫 실패 층에서
`max 12 vertices/child` 조건을 만족하는 연속 fan-cell cover가 없어 멈췄다.

D386의 질문은 다음 하나였다.

> 분할법, 원 형상, 무겹침, polygon/face/volume/surface gate를 그대로 둔 채,
> D385가 처음 막힌 네 층에 꼭 필요한 최소 자식 꼭짓점 수가 얼마인가?

이번 case의 신규 변수:

`observed_no_cover_layer_exact_minimax_vertex_budget_localizer_v1`

D386은 새 충돌체 설계나 PhysX 시험이 아니다. `12`를 곧바로 완화하지 않고,
먼저 고정된 후보 그래프 안에서 필요한 임계값만 찾는 offline 진단이다.

## 2. 승인 범위와 동결 조건

공식 판정 대상은 D385 원 evidence에 기록된 다음 네 first-observed layer뿐이다.

| body | 부모 | 층 |
|---|---|---|
| `gripper_link` | `proximal_upper_arm_hull_a` | `z_layer_00` |
| `gripper_link` | `proximal_lower_arm_hull_a` | `z_layer_01` |
| `link5` | `fixed_backbone_left` | `y_layer_01` |
| `link5` | `fixed_backbone_right` | `y_layer_00` |

D385는 각 부모에서 첫 실패가 나오면 뒤쪽 층을 계산하지 않았다. 이 네 부모에
존재하는 나머지 7개 층은 이름과 존재만 기록했고 D386 계산 대상에는 넣지
않았다. 따라서 D386 값은 부모 전체나 완성 P34의 예산이 아니다.

동결한 조건:

- D379 authored Float32 point stream
- D385 semantic pre-split axis와 interval
- broad-profile fan anchor와 순서
- 연속 fan group size `1..4`
- 원 부모와의 clipping/intersection
- polygon count `<=64`
- 한 polygon의 vertices `<=32`
- positive volume
- 바깥 돌출과 부모 표면 미포함 `<=0.1mm`
- topology-volume 상대오차 `<=0.5%`
- 자식 사이 positive-volume overlap `0`
- 탐색 경로의 child vertices `12..64`

실행 금지:

- alternate partition, group size `5+`, internal overlap, tolerance relaxation
- USD/asset/collider materialization 또는 regeneration
- Isaac, Kit, PhysX, live callback, Warp, CUDA
- `29x50mm` cylinder 생성/수정
- physics step, q5, contact, grasp
- target/IK/path 및 material/mass/actuator/physics setting 변경

## 3. 실행 전 교차검사와 사전 수리

### 3.1 Git 및 입력

- `HEAD == origin/master ==`
  `35f10e3079b19e51209ba4cf1dd66391a431b053`
- subject: `D384`
- D385 forward-only 산출물은 미커밋 상태로 보존
- D385 script SHA-256:
  `ea1d76a8db9c78a3cae9de50a62e0a25283d5550346dad158e641a0da321c5ed`
- D385 evidence SHA-256:
  `4ff64045d4e2e7ecc3601927d1d6c97fd1a61b636e838241f9fded6b02e3cc00`
- D385 completion SHA-256:
  `2caf6c47ad563c9ad82b84d5c3367139943f95c98c62a590b3551a967def91c2`
- D379 evidence SHA-256:
  `8eacbad796e8532c3d856b865e90dc54481f0f2003a266c3ebfaa8e93de37af5`

### 3.2 정적 감사에서 발견해 실행 전에 고친 것

공식 output 생성 전에 독립 정적감사 세 건을 수행했다. 첫 감사에서는 다음
계약 불일치를 찾아 실행을 보류했다.

1. 전체 후보 그래프 생성 뒤 D385 `B=12` helper replay가 후보를 다시
   생성하는데도 “형상 생성 정확히 1회”라고만 적혀 있었다.
2. minimax/exhaustive/reachability 내부가 64 초과 경로도 계산한 뒤 공개값만
   `null`로 바꾸고 있었다.
3. shadowed-layer evaluation count `0`이 실제 계수가 아니라 상수였다.
4. 동결 D385 helper code가 SHA-256 확인 전에 import-time 실행됐다.

실행 전에 다음처럼 수리했다.

- target layer마다 `complete graph 1회 + frozen B12 replay 1회`로 정직하게
  분리 등록하고 실제 카운터로 검증
- minimax, exhaustive, forward/backward reachability 모두
  `vertex_count<=64`인 edge만 사용
- evaluated-layer key와 shadowed-layer key의 실제 교집합으로 evaluation `0`
  검증
- worker provenance가 script/START/input/HEAD/origin을 모두 통과한 뒤에만
  exact-SHA D385 module load
- selected, parent-wide, complete-P34 budgets를 각각 명시적 `null`로 gate
- watchdog signal은 timeout일 때 새 D386-owned worker process group에만 허용
- headless Viewer `<=1`을 worker와 finalize에서 직접 gate

수리 뒤 세 재감사가 모두 GO였다. 이때까지 D386 output과 worker invocation은
각각 `0`이었다.

## 4. 계산 방법

각 층에서 가능한 fan-cell interval geometry를 한 번 전수 생성했다.
꼭짓점 수와 별개로 polygon, face-width, positive-volume gate를 먼저 적용했다.

그 뒤 두 독립 방법으로 같은 최소 bottleneck을 계산했다.

1. 동적계획법: 완전 경로에서 가장 큰 child vertex count를 최소화한다.
2. 전수 열거: 고정 gate를 통과하는 모든 완전 경로를 직접 열거해 같은
   최소값과 canonical cut을 확인한다.

유한 최소값 `B*`는 다음 경계까지 확인해야 인정했다.

- `B=12`: D385 no-cover를 exact replay
- `B*-1`: no-cover
- `B*`: cover
- 선택된 `B*` witness가 surface/volume/polygon/face/positive-volume/overlap
  gate를 모두 통과

최소값이 없으면 `B=64`에서도 no-cover여야 한다.

## 5. 단계별 실행

Canonical path:

`claudedocs/runtime_logs/grasp_track/g0a_d386/attempt1_observed_no_cover_layer_minimum_vertex_budget_localization/`

### 5.1 prepare

- preregistration: PASS
- 신규 변수: 정확히 1개
- D385 first-failure inventory: exact `4/4`
- input hashes, HEAD/origin, Git-status before/after output-create: PASS
- forbidden runtime import roots: 없음

Preregistration SHA-256:

`d1d6b76def3c67741b0bd261ecdc52d73901ad382ba143f4757730a5148f7036`

### 5.2 actual worker

- actual worker: `1`
- retry: `0`
- return code: `0`
- elapsed: `4.933896491071209s`
- watchdog: `300s`
- timeout: false
- termination action: null
- residual process-group member: `0`

Worker stderr의 Matplotlib 메시지는 사용자 config directory가 read-only라
`/tmp/matplotlib-*` cache를 사용했다는 경고뿐이다. 계산이나 산출물 실패는
없었다.

## 6. 정량 결과

### 6.1 최소 꼭짓점 예산

| body | 부모/층 | 후보 | 비-꼭짓점 gate 통과 | 최소값 | 전 단계 | 자식 | canonical cuts |
|---|---|---:|---:|---:|---|---:|---|
| `gripper_link` | `proximal_upper_arm_hull_a/z_layer_00` | 74 | 74 | 28 | 27 FAIL | 6 | `0,1,5,9,13,17,20` |
| `gripper_link` | `proximal_lower_arm_hull_a/z_layer_01` | 82 | 40 | `null` | 64 FAIL | `null` | `null` |
| `link5` | `fixed_backbone_left/y_layer_01` | 66 | 66 | 30 | 29 FAIL | 5 | `0,4,6,10,14,18` |
| `link5` | `fixed_backbone_right/y_layer_00` | 78 | 77 | 13 | 12 FAIL | 6 | `0,1,5,9,13,17,21` |

전수 경로 수:

- upper support: `283,953`
- fixed left: `76,424`
- fixed right: `263,384`
- lower support: 완전 경로 없음

동적계획법과 전수열거의 finiteness, minimum, canonical cut은 네 층 모두
일치했다. D385 `B=12` error message와 independent no-cover도 exact `4/4`
PASS했다.

### 6.2 왜 lower support는 꼭짓점만 늘려도 해결되지 않는가

`proximal_lower_arm_hull_a/z_layer_01`:

- broad-profile vertices: `24`
- fan triangles: `22`
- 고정 group `1..4` 후보: `82`
- geometry constructed: `82`
- `polygon_count>64` 거부: `42`
- 비-꼭짓점 gate 통과: `40`
- `B=12` cover: 없음
- `B=64` cover: 없음

즉 D385 실패를 모두 “12가 너무 작다”로 설명할 수 없다. 이 층에서는
꼭짓점 제한을 64까지 올려도 frozen polygon gate를 지키는 완전 경로가
없다. D386은 polygon gate를 제거하거나 분할법을 바꾸지 않았다.

### 6.3 유한 witness 형상 gate

| 부모/층 | max child polygons | max vertices/face | outward mm | coverage mm | volume rel. error | overlap |
|---|---:|---:|---:|---:|---:|---:|
| upper support/z00 | 49 | 6 | `6.311752700355333e-7` | `3.593147385530515e-7` | `1.8635287576192464e-7` | 0 |
| fixed left/y01 | 55 | 7 | `8.184670059752097e-7` | `6.724402354385539e-7` | `2.5736050393251484e-5` | 0 |
| fixed right/y00 | 18 | 8 | `4.256505113653386e-7` | `9.678084375575047e-8` | `6.255424941142787e-6` | 0 |

세 유한 witness는 모든 frozen gate를 통과했다. 그러나 이들은 세 layer의
임계값 증거일 뿐, USD에 올릴 완성 충돌체나 채택된 예산이 아니다.

### 6.4 최종 과학 판정

- finite layers: `3/4`
- null layers: `1/4`
- three-finite diagnostic maximum: `30`
- observed-four-layer maximum: `null`
- selected vertex budget: `null`
- parent-wide vertex budget: `null`
- complete-P34 vertex budget: `null`
- application count: `0`
- complete source-child count: `null`
- complete total-part count: `null`
- materializable candidate: false
- localization pass: false
- verdict:
  `D386_OBSERVED_LAYER_VERTEX_BUDGET_NOT_LOCALIZABLE_FAIL_STOP`

`30`을 새 예산으로 적용하면 안 된다. 네 번째 층의 값이 없고, 뒤쪽 7개
층도 아직 계산하지 않았기 때문이다.

## 7. 시각화와 직접 관찰

- exact board:
  `d386_vertex_budget_localization_1920x1080.png`
  (`1920x1080`, SHA-256
  `68d1f320e5913d3bbe6edd83990494f5699807b541fbec6cc1be52d24a17d180`)
- save-only RRD:
  `d386_vertex_budget_localization.rrd`
  (SHA-256
  `d5ebf6040dcde20196251270ae5fa83d1d8c42c8eea654f6e0c458d1be7f789a`)
- embedded RBL:
  `d386_vertex_budget_localization.rbl`
  (SHA-256
  `1ecce9de6dcab237e3c3fa3990f2acfc9e433e8832799a9efbed969e767bd704`)
- headless Viewer screenshot:
  `d386_rerun_inspection.png`
  (native HiDPI `3840x2160`, SHA-256
  `c73729887dfb42008080fa3218377ec7df256d44ca84627a485a203cdc6467e5`)

Board는 네 card, `B*-1 FAIL → B* PASS`, lower-support `NULL`, 하단
“공통 예산 NULL / 전체 P34 후보 아님”을 겹침 없이 보여준다.

Rerun은 네 shifted layer와 finite witness child, decision metadata를 실제로
표시했다. sandbox에서 `message proxy server crashed: Operation not permitted`
알림과 llvmpipe software-renderer 경고가 있었지만 Viewer return은 `0`이고,
RRD/RBL footer, exact entity/component, exact `blueprint/log_time`, version
`0.34.1` 검증은 PASS다. 이 경고와 HiDPI 2배 PNG는 수동검사에 명시했고
canonical JSON 수치 권위로 사용하지 않았다.

Manual checks `7/7`, completion/observability PASS다. 이는 scientific FAIL을
정확히 기록했다는 뜻이지 설계 성공이 아니다.

## 8. 실행하지 않은 것

원 evidence의 현재-scope counters:

- alternate partition `0`
- overlap allowance/tolerance change `0/0`
- asset/USD read/write `0/0`
- collider materialization/regeneration `0`
- Isaac/Kit/PhysX/live callback `0/0/0/0`
- Warp/CUDA `0`
- cylinder create/write `0`
- controlled physics/q5/contact `0/0/0`
- target/IK/path change `0`
- material/mass/actuator/physics-setting change `0`

따라서 D386은 Isaac Sim 실패, PhysX cook 실패, 원통 접촉 실패, 또는 파지
실패가 아니다. 실제 simulator와 물리를 전혀 실행하지 않은 offline
representation diagnostic FAIL이다.

## 9. NVIDIA 자료의 적용 범위

적용 설치 버전:

- Isaac Sim `5.1.0.0` — installed, not launched
- Isaac Lab `2.3.0` — installed, not launched
- Rerun SDK/CLI `0.34.1`
- D385에서 교차검사한 PhysX schema/property extension `107.3.26`

참고한 공식 자료:

1. [Omni Physics 107.3 — Colliders](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/collision.html)
   — convex mesh와 cooking의 공식 문맥.
2. [PhysX 5.6.1 — GPU Simulation](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/GPURigidBodies.html)
   — vertices, polygons, vertices/face가 서로 다른 호환 조건임.
3. [PhysX 107.3 — `GuConvexMesh::isGpuCompatible`](https://raw.githubusercontent.com/NVIDIA-Omniverse/PhysX/107.3-omni-and-physx-5.6.1/physx/source/geomutils/src/convex/GuConvexMesh.cpp)
   — count 외에도 GPU data와 shape 조건이 별도임.

D386의 `polygon<=64`, face `<=32`는 이 공식 GPU 호환 문맥과 일치하도록
D385에서 동결한 offline precheck다. 그러나 D386은 cook/callback/GPU contact를
실행하지 않았으므로 `live_gpu_compatibility_pass=null`이다.

## 10. 증거 해시

- D386 script:
  `60b5b2d15518baa0427e44f0928a46993e78eeba45307636b234cb0b042acf8d`
- canonical evidence:
  `ae956a2b64835f4030daf104f08d239f140f8ba9b32ee9205f2b744769c51d4c`
- finite witness geometry:
  `ec5016cb5ebee9930c23093a6f3211a397466137f78d93e30357f8e10744a187`
- 300-candidate-row CSV:
  `adfb2c6007ff84e756e5d6afca260a20cbfa9d6c0cf3a180c3aaf6d458084dd2`
- Rerun validation:
  `4b80e5a7db0223eed2202d4fa46b7f6d312f81ff97295fd8843e634a264ec9bd`
- completion:
  `622c34fdb7cbd11d2b0465eda75ac1119407fcaab441a059ff40289065170b6e`

## 11. 일상어 결론과 다음 승인 경계

세 곳은 조각 하나가 꼭짓점 `13`, `28`, `30`까지 가면 현재 분할법으로
나눌 수 있었다. 하지만 한 곳은 꼭짓점 수를 `64`까지 늘려도 안 됐다.
그곳은 조각 면의 개수 제한까지 함께 걸리기 때문이다.

따라서 “12 대신 30으로 바꾸면 해결된다”는 결론은 틀리다. 지금 선택해야
할 새 전역 예산은 없다.

다음 권장 최소 후보는 아직 미승인인
`D387 [d386_shadowed_layer_fixed_graph_completion_localization]`이다. D386에서
계산하지 않은 뒤쪽 7개 층만 같은 고정 그래프와 gate로 확인해 전체 실패
지도를 완성한다. 그 뒤에야 polygon-gated layer의 partition/representation
repair를 한 변수씩 설계하는 것이 안전하다.

D387, budget 선택/적용, alternate partition, gate 완화, USD/PhysX
materialization, actual `29x50mm` cylinder, physics/q5/contact/grasp는 새 명시
승인 전 실행하지 않는다.
