# 2026-07-21 Grasp G0a D372 — 교수님 안 의미 기반 복합 충돌체 오프라인 설계

## 1. 무엇을 왜 했는가

사용자가 명시한 교수님 안은 다음 세 부분을 서로 다른 소유 링크와 역할로 나누는 것이었다.

1. `link5`의 비접촉 몸통은 단순한 박스 중심으로 만든다.
2. `link5`에 붙어 있는 고정 턱은 원통 접촉면과 큰 빈 공간을 보존하도록 별도 볼록 조각으로 만든다.
3. `gripper_link`에 붙어 q5와 함께 움직이는 턱은 고정 턱과 분리하고, 열린 입구와 내부 구멍을 보존하도록 별도 볼록 조각으로 만든다.

이번 case의 신규 변수: `[semantic_owner_region_partition_v1, manual_compound_primitive_budget_v1]`

이 둘만 바꿨다. 기존 raw mesh, D350/D354 턱 표면 권위, D349 frozen-OPEN 자세,
D362 저장 trace, 원통 형상과 target/IK/path는 동결했다.

이 case는 실제 Isaac/PhysX 물리를 돌리기 전의 오프라인 후보 설계다. 새 USD나 live collider를
만들지 않았고, q5 명령·physics step·실제 contact query도 실행하지 않았다.

## 2. 부팅과 Git 경계

- 실행 전 `HEAD == origin/master == 4a1120b801e808071583136e78954c78ca941dc8`, subject `370 test`를 확인했다.
- D371 및 D372 작업물은 사용자가 아직 commit/push하지 않은 상태로 보존했다.
- `claudedocs/lab_meeting/20260715/d334_collision_table/` 세 파일은 prepare와 finalize에서 해시를 다시 비교했고 그대로였다.
- commit/push, 하드웨어 제어, signal 전송은 하지 않았다.

## 3. 앞선 화면 종료 원인과 D372 실패를 분리한 감사

사용자가 본 화면 종료는 D372 계산의 GPU 오류가 아니었다.

- `/var/log/kern.log:3485-3486`에는 2026-07-21 18:38:09 `cursor invoked oom-killer`가 기록되어 있다.
- 같은 로그 `:3541`에는 anonymous RAM 약 51.9GiB가 active/inactive로 잡혀 있고,
  `:3554-3557`에는 swap이 `0kB`라고 기록되어 있다.
- `:4754-4755`에서 커널은 global OOM으로 Cursor PID 3314170을 실제 종료했다.
- 당시 GNOME, Cursor, Claude/Codex/MCP, Python 등 여러 프로세스가 함께 있었으므로
  “D372 한 프로세스가 전부 사용했다”는 인과 증거는 없다. GPU NVRM/Xid 오류도 같은 시각에는 없다.

D372 attempt1은 그 뒤 19:11에 발생한 별개의 prepare 코드 오류다.

- 설치 영역의 NVIDIA `database.py`를 repo 상대경로로 바꾸려 해 `ValueError`가 났다.
- preregistration, invocation, measurement는 생성되지 않았고 Isaac/PhysX/q5/physics는 모두 0이다.
- 이 실패는 `g0a_d372/d372_runtime_exception.json`에 그대로 동결했다.

수정은 설치 파일 자체를 repo 입력 목록에 넣지 않고, 설치 NVIDIA 계약 안에서 절대경로와 SHA-256을
검사하도록 제한했다. 새 forward-only 경로는
`g0a_d372/attempt2_external_schema_path_repair/`다.

## 4. NVIDIA 공식 근거와 설치본 교차검사

설치 제품과 문서 버전을 먼저 맞췄다.

- Isaac Sim: `5.1.0.0`
- Omni PhysX extension: `107.3.26`
- 설치 schema `PhysxSchema/resources/schema.usda:886`: `hullVertexLimit=64`
- 같은 schema `:895`: `maxConvexHulls=32`
- 설치 property UI `database.py:954`: `hullVertexLimit 8..64`, `maxConvexHulls 1..2048`

여기서 `32`는 자동 convex decomposition의 schema 기본값이다. 수동으로 child collider를 몇 개
둘지 정하는 목표 숫자가 아니다. `2048`도 설치 UI 입력 범위의 상한이지 엔진 절대 한계나 최적값이 아니다.

사용한 NVIDIA 1차 자료:

1. Omni Physics 107.3, **Rigid Bodies**  
   https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/rigid_bodies.html  
   하나의 rigid body 아래 여러 collider를 둘 수 있다는 근거다.
2. Omni Physics 107.3, **Colliders**  
   https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/collision.html  
   형상이 허용하면 box 같은 primitive를 먼저 쓰고, 그다음 convex mesh를 쓰라는 근거다.
3. Isaac Sim 5.1.0, **Physics Simulation Fundamentals**  
   https://docs.isaacsim.omniverse.nvidia.com/5.1.0/physics/simulation_fundamentals.html  
   한 convex hull이 구멍을 메우는 반면 여러 수동 형상 또는 convex decomposition으로 빈 공간을
   보존할 수 있고, 일반적으로 hull 수가 적을수록 빠르다는 근거다.
4. Isaac Sim 5.1.0, **Performance Optimization Handbook**  
   https://docs.isaacsim.omniverse.nvidia.com/5.1.0/reference_material/sim_performance_optimization_handbook.html  
   필요한 정밀도를 만족하는 가장 단순한 근사와 적은 collider 수를 쓰라는 근거다.
5. Omni Physics 107.3, **PhysX Schema API**  
   https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/schemas/physxschema.html
6. PhysX 5.6.1, **GPU Rigid Bodies**  
   https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/GPURigidBodies.html  
   convex GPU 기하 제한의 보강 자료다. 설치 PhysX SDK exact semver를 확인하지 못했으므로 설치본과
   동일 버전이라는 주장에는 사용하지 않았다.

## 5. 실제 설계

후보 이름은 `P34_professor_semantic_compound`다.

### link5: 16개

- 비접촉 몸통 박스: 1개
- 목·회전축 연결/지지: 3개
- 고정 턱 접촉층: 10개
- 고정 턱 좌·우 뒷면 지지대: 2개

### gripper_link: 18개

- 몸쪽의 위·아래 연결부: 4개
- 움직이는 턱 접촉층: 12개
- 움직이는 턱 위·아래 뒷면 지지대: 2개

합계는 `16 + 18 = 34`다. 현재 A64 기준 `64 + 64 = 128`과 비교하면 94개,
즉 `73.4375%`를 줄인 후보다. 그러나 이는 속도나 최적성의 측정값이 아니다.

몸통을 박스로 만든 이유는 원통과 접촉시키려는 정밀 영역이 아니기 때문이다. 반대로 턱은 큰 구멍과
열린 입구를 하나의 convex hull로 감싸면 빈 공간이 가짜 고체가 되므로 여러 볼록 조각으로 나눴다.
고정 턱은 `link5`, 움직이는 턱은 URDF joint `link5_to_gripper_link`의 child인 `gripper_link`에 유지했다.

## 6. 실패 가능한 오프라인 검증

세션 진행 규칙을 충족하는 failure-capable perturbation은 다음 다섯 음성대조군이었다.

1. full-link5 AABB는 frozen OPEN에서 원통을 침범해야 한다.
2. 고정 턱 단일 외곽 convex는 큰 구멍 두 곳을 메워 탈락해야 한다.
3. 움직이는 턱 단일 외곽 convex는 열린 입구를 메워 탈락해야 한다.
4. 고정/움직이는 접촉층을 제거하면 D350/D354 인증 seed를 잃어야 한다.
5. 고정 턱과 움직이는 턱의 owner를 바꾸면 URDF parent/child 계약과 맞지 않아야 한다.

결과는 `5/5 PASS`였다. 접촉층을 제거했을 때 seed 손실 거리는 고정 턱
`3.5013612402mm`, 움직이는 턱 `2.1517001921mm`였다.

## 7. 정량 결과

### 7.1 raw 표면과 후보의 오프라인 거리

- link5 raw sample 중 후보 밖인 점의 거리: P95 `0.1795673203mm`, max `1.0937375550mm`
- gripper_link: P95 `0.1872935098mm`, max `1.1572354710mm`

이 값은 등록된 P95 `2mm`, max `3.5mm` gate를 통과했다. raw vertex 표본의 개수 비율은 tessellation
밀도에 의존하므로 전역 표면 권위로 승격하지 않았다.

### 7.2 턱 빈 공간 2차원 진단

- 고정 턱 큰 빈 공간 채움: `0.0000%`, `1.4176%`
- 움직이는 턱 열린 입구 채움: `8.6019%`
- 움직이는 턱 내부 창 채움: `27.1669%`

이 수치는 접촉층과 바로 붙은 jaw-backbone만 투영한 2차원 진단이다. connector/moving-support는
제외했으며, 전체 P34의 3차원 관통 공간 또는 물리 동등성 증명이 아니다.

### 7.3 frozen OPEN 정적 간격

- link5 P34: `4.2726834003mm`
  - D349 raw 기준과 절대 차이 `0.0000378667mm`
- gripper_link P34: `10.9714602318mm`
  - D349 raw 기준과 절대 차이 `0.2036281428mm`

둘 다 원통과 겹치지 않았고 사전등록 최소 간격 `0.1mm`를 넘었다.

### 7.4 immutable D362 저장 좌표 재계산

D362의 `q5_close_observation` 61개 저장 pose만 읽어 A64와 P34의 정적 겹침을 다시 계산했다.

- moving jaw 첫 겹침: A64/P34 모두 global step `232`; 전환창 max 거리 차이 `0.0025090084mm`
- fixed jaw 첫 겹침: A64/P34 모두 global step `246`; 전환창 max 거리 차이 `0.0270367065mm`
- P34에서 처음 겹친 선택 부품은 각각 `moving_jaw`와 `fixed_jaw` 접촉층이었다.

이는 저장된 동일 pose에 다른 형상을 대입한 counterfactual offline replay다. 실제 P34를 물리엔진에서
움직였을 때 같은 pose와 접촉순서가 발생한다는 인과 증명이 아니다.

### 7.5 실행 범위

- actual run / automatic retry: `1 / 0`
- run marker 내부 경과: `21.093865699s`
- immutable D362 rows read: `61`
- offline hppfcl static part query: `10045`
- SimulationApp/Kit, Isaac/PhysX, cook/automatic decomposition, q5 sample, physics step,
  live contact query, USD/live asset write, target/IK/path change, material/mass/actuator/physics change: 전부 `0`

## 8. 시각화와 finalize

- 정확한 `1920x1080` 교수님용 비교판 3장을 생성했다.
- RRD/RBL footer, 정확한 entity/component/timeline 계약과 Rerun `0.34.1` 검증이 PASS했다.
- Rerun Viewer의 논리 창은 `1920x1080`, HiDPI 물리 PNG는 `3840x2160`이다.
- 원본해상도 수동 검사에서 1080p 비교판 세 장은 글자 겹침·잘림 없이 읽혔다.
- Rerun 화면에는 정보성 알림 세 개와 보조 이벤트 행의 가로 잘림이 남았지만 후보 형상, 원통,
  거리 곡선, `d362_phase_step` 타임라인을 가리지 않았다. 이 제한을 manual JSON/MD에 그대로 기록했다.
- finalize는 정확히 한 번 실행했고 checks `15/15`, visualization PASS였다.

## 9. 최종 판정

최종 verdict는
`D372_PROFESSOR_SEMANTIC_COMPOUND_CANDIDATE_OFFLINE_PASS_NO_PHYSICS`다.

쉽게 말하면, 교수님 안대로 만든 `link5 몸통 + 고정 턱 + 움직이는 턱` 34개 후보는 오프라인에서
형상·소유권·간격·저장 trace·시각화 gate를 통과했다. 그러나 아직 Isaac/PhysX에 실제 collider로
올리지 않았고 원통을 잡아보지도 않았다.

다음 항목은 모두 `null`이다: live asset identity, actual GPU contact execution, physics equivalence,
D362 replay causal equivalence, runtime speed, tipping causality, grasp feasibility, global optimum.
`g0a_pass=false`다.

## 10. 다음 승인 경계

승인되지 않은 다음 최소 단계는 새 forward-only live-asset identity preflight다.

1. P34를 별도 새 USD/candidate 경로에 materialize한다.
2. physics/q5/contact는 0으로 유지한다.
3. live callback/readback에서 `link5=16`, `gripper_link=18`, owner·vertex·polygon digest가
   오프라인 후보와 정확히 같은지만 확인한다.

그 PASS 뒤에도 실제 물리 비교는 다시 별도 승인이 필요하다. 인과를 구분하려면 동일 pose/trajectory와
물리 설정에서 `A64`, `link5만 P34`, `gripper_link만 P34`, `양쪽 P34`를 한 번에 섞지 말고
forward-only case로 나누어야 한다. target/IK/path와 중앙높이/wrist 수정은 collider 효과를 본 뒤의
별도 변수다.

## 11. 주요 증거

- attempt1 prepare exception: `claudedocs/runtime_logs/grasp_track/g0a_d372/d372_runtime_exception.json`
- preregistration: `claudedocs/runtime_logs/grasp_track/g0a_d372/attempt2_external_schema_path_repair/d372_preregistration.json`
- geometry: `claudedocs/runtime_logs/grasp_track/g0a_d372/attempt2_external_schema_path_repair/d372_professor_semantic_candidate_geometry.json`
- measurement evidence: `claudedocs/runtime_logs/grasp_track/g0a_d372/attempt2_external_schema_path_repair/d372_professor_semantic_candidate_evidence.json`
- professor report and boards: `claudedocs/runtime_logs/grasp_track/g0a_d372/attempt2_external_schema_path_repair/`
- manual inspection: `d372_manual_visual_inspection.json`, `d372_manual_visual_inspection.md`
- completion: `d372_completion_summary.json` SHA-256
  `57f3ed8fe6f057d059980a78bb51be8e881d8300297a4f41def6ddf94ad0cf43`
- harness: `sim_scripts/cyl34_top_view_d372_professor_semantic_compound_collider_design_offline.py`

