# D372 교수님 안: 의미 기반 복합 충돌체

이 단계는 충돌체 후보를 오프라인에서 만든 것입니다. Isaac/PhysX 실행, 물리 스텝, q5 구동, 실제 접촉·파지는 모두 0회입니다.

## 만든 구조

- link5: 몸통 박스 1개 + 연결/회전축 3개 + 고정 접촉판 10개 + 고정 턱 뒷면 지지대 2개 = 16개
- gripper_link: 근위 지지 볼록껍질 4개 + 움직이는 접촉판 12개 + 위/아래 뒷면 지지대 2개 = 18개
- 합계: 34개 (현재 기준 128개)

## 개수 비교와 의미

| 후보 | 생성 방식 | link5 | gripper_link | 합계 | 해석 |
|---|---:|---:|---:|---:|---|
| 현재 A64 | 자동 convex decomposition | 64 | 64 | 128 | 현재 64-cap 기준 후보 |
| D371 R32 | 자동 convex decomposition | 32 | 32 | 64 | maxConvexHulls만 32로 바꾼 비교 후보 |
| D372 P34 | 의미 기반 수동 복합 충돌체 | 16 | 18 | 34 | 몸통·고정 턱·움직이는 턱을 역할별 분리한 G0a 후보 |

- 설치 스키마의 `maxConvexHulls=32`는 자동 분해의 기본값이지, 수동 child collider의 목표 개수가 아닙니다.
- D371 C1/C2는 몸통과 턱의 정확한 분할이 아니었으므로 D372 설계의 직접 대안으로 채택하지 않습니다.
- 부품 수만으로 속도, 물리 동등성, 전도 개선, 최적성을 증명할 수 없습니다.

## 빈 공간과 열린 자세

- 고정 턱 접촉층+인접 뒷면 지지대의 2D 큰 빈 공간 채움: 0.00% / 1.42%
- 움직이는 턱 접촉층+위·아래 지지대의 2D 열린 입구 채움: 8.60%
- 움직이는 턱 접촉층+위·아래 지지대의 2D 내부 창 채움(진단): 27.17%
- 위 2D 수치는 뒤쪽 connector/moving-support를 제외한 접촉층 진단입니다. 전체 P34의 관통 공간이나 3D 물리를 증명하지 않습니다.
- 동결 OPEN 정적 간격: link5 4.272683mm, 움직이는 턱 10.971460mm
- 저장된 D362 좌표 재계산의 첫 겹침 global step: link5 A64/P34=246/246, 움직이는 턱 A64/P34=232/232

## NVIDIA 공식 근거와 설치 버전

- 설치 제품: Isaac Sim 5.1.0.0, Omni PhysX extension 107.3.26
- 설치 스키마: `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.schema.physx-107.3.26+107.3.3.lx64.r.cp311.u353/plugins/PhysxSchema/resources/schema.usda:886`의 hullVertexLimit=64, `:895`의 maxConvexHulls=32 기본값
- 설치 속성 편집 UI: `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extscache/omni.kit.property.physx-107.3.26+107.3.3.cp311.u353/omni/kit/property/physx/database.py:954`에서 hullVertexLimit 8~64, maxConvexHulls 1~2048. 이는 UI 입력 범위이며 엔진 절대 한계나 최적값이 아닙니다.
- D372의 convex당 64 vertices/64 polygons/32 vertices-per-polygon는 프로젝트가 사전등록한 GPU 적격성 gate입니다. 실제 GPU 실행을 뜻하지 않습니다.
- Omni Physics 107.3 Rigid Bodies (https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/rigid_bodies.html) — one rigid body may own multiple child colliders
- Omni Physics 107.3 Colliders (https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/collision.html) — primitives first, convex meshes when needed
- Isaac Sim 5.1 Physics Simulation Fundamentals (https://docs.isaacsim.omniverse.nvidia.com/5.1.0/physics/simulation_fundamentals.html) — multiple convex shapes preserve concave openings
- Isaac Sim 5.1 Performance Optimization Handbook (https://docs.isaacsim.omniverse.nvidia.com/5.1.0/reference_material/sim_performance_optimization_handbook.html) — use the simplest collision approximation that satisfies precision
- PhysX 5.6.1 GPU Rigid Bodies (https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/GPURigidBodies.html) — supplementary convex geometry limits; not an installed SDK identity claim
- Omni Physics 107.3 PhysX Schema API (https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/schemas/physxschema.html) — version-matched schema API defaults for hullVertexLimit and maxConvexHulls

## 해석 경계

- 이 후보는 G0a 원통 경로를 위한 task-local 후보이며, 일반 목적 로봇 충돌 모델의 최적값이 아닙니다.
- D362 재계산은 저장 좌표에 새 형상을 대입한 오프라인 검사일 뿐, 새 형상이 같은 동역학을 만든다는 인과 증명이 아닙니다.
- 이 후보는 다음 live asset identity 검사 대상으로 적격인지까지만 판정합니다.
- 아직 실제 속도, 접촉 순서, 원통 전도, 파지 성공, 전역 최적성을 판정하지 않습니다.
- 후속 물리 비교에서는 link5만 교체한 경우와 gripper_link만 교체한 경우를 분리해야 원인을 구분할 수 있습니다.

## 시각자료

- 소유 링크와 분할: `claudedocs/runtime_logs/grasp_track/g0a_d372/attempt2_external_schema_path_repair/d372_ownership_and_split_1920x1080.png`
- 턱 빈 공간: `claudedocs/runtime_logs/grasp_track/g0a_d372/attempt2_external_schema_path_repair/d372_jaw_void_preservation_1920x1080.png`
- 동결 OPEN 정적 간격: `claudedocs/runtime_logs/grasp_track/g0a_d372/attempt2_external_schema_path_repair/d372_frozen_open_clearance_1920x1080.png`
- Rerun 기록: `claudedocs/runtime_logs/grasp_track/g0a_d372/attempt2_external_schema_path_repair/d372_professor_semantic_candidate.rrd`
