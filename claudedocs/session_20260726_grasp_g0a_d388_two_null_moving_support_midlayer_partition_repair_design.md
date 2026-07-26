# D388 — two-null moving-support midlayer partition repair design

Date: 2026-07-26 KST

## 1. 무엇을 왜 확인했는가

D387은 P34의 실패 지도를 완성했고, `gripper_link`의 움직이는 지지부
가운데 층 두 개만 기존 fan graph에서 `B=64`까지 완전한 분할 경로가
없음을 확인했다. D388은 다른 아홉 층과 모든 형상 gate를 동결한 채,
두 층의 같은 반시계방향 단면에서 fan 기준점만 한 번 순환 이동하면
경로 단절이 해소되는지를 확인한 offline-only 설계 검사다.

여기서 `B`는 한 child convex hull이 허용하는 꼭짓점 수의 진단 예산이다.
충돌체 부품 수, NVIDIA 기본값, 채택한 전역 예산을 뜻하지 않는다.

이번 case의 신규 변수:

`null_middle_layer_first_blocked_triangle_reanchored_fan_graph_v1`

등록한 규칙은 하나다.

- D387의 기존 기준점 `0`에서 앞으로 도달 가능한 마지막 상태 다음
  꼭짓점을 새 기준점으로 사용한다.
- 따라서 상단은 기준점 `11`, 하단은 `10`으로 기계적으로 정해졌다.
- 점 집합과 반시계방향 순서는 바꾸지 않고
  `old_polygon[k:] + old_polygon[:k]`만 허용했다.
- 기준점 전수탐색, 다른 분할법, gate 완화, 예산 선택은 허용하지 않았다.

이 case는 성공과 실패가 모두 가능한 perturbation evaluation이므로
AGENTS.md의 session progress rule을 충족한다.

## 2. 부팅·계보·사전등록

- 실행 전 Git은
  `HEAD == origin/master == 930b41d98576a9c0bf1dce4f3eb1c0d93df8014b`
  였다. D387의 승인된 미커밋 기준선만 보존했다.
- D387 script/session/output manifest aggregate는 각각
  `39d1f9f...d9ee`, `a71d8d...0e64`, `f5f0d210...94b9`로
  재검증했다.
- D388 실행 script SHA-256은
  `7f99f80c19b4ab7e8adbae6237ed675feb738f9e1c4418049c1fa2f166c743bf`
  다.
- D385의 signal-bearing module은 import하지 않았다. 필요한 순수 형상
  함수 12개의 AST만 동결 D385 source SHA 아래 exact 비교했다.
- prepare는 `23/23` PASS했다.
- 등록 출력은 새 forward-only 경로
  `claudedocs/runtime_logs/grasp_track/g0a_d388/attempt1_two_null_moving_support_midlayer_partition_repair_design/`
  하나다.

## 3. 실행 순서

1. 상단 `p000_proximal_upper_arm_hull_a/z_layer_01`의 동결 frontier를
   읽고 새 기준점 `11`을 도출했다.
2. 같은 점·방향을 순환 이동해 후보 edge `74`개를 한 번 구성했다.
3. 동적계획법과 독립 전수열거로 `B=12..64`의 완전 경로를 비교했다.
4. 선택된 진단 경로의 polygon 수, face 폭, 양의 부피, 표면 오차,
   부피 오차, Float32 child 간 양의 부피 겹침을 검사했다.
5. 하단 `p002_proximal_lower_arm_hull_a/z_layer_01`에도 같은 규칙을
   적용해 기준점 `10`, 후보 edge `82`개를 한 번 평가했다.
6. 원본 JSON/CSV/RRD/RBL/비교판을 기록한 뒤 method/geometry 계약
   실패를 감지해 worker가 의도적으로 FAIL_STOP했다.

실제 worker는 `1`회, retry `0`, signal `0`이다. 실행시간은
`4.434133296832442s`, cooperative deadline exceeded=false,
worker exited=true, return code는 `1`이다. 이는 timeout이나 crash가
아니라 evidence 저장 후 `worker claim=false`를 감지해 낸 의도적 예외다.

## 4. 정량 결과

### 4.1 기존 NULL을 유한 경로로 바꾸었는가

| 대상 | 기존 상태 | 새 기준점 | 새 분류 | 최소 진단 B | 바로 아래 B | 진단 child |
|---|---:|---:|---|---:|---:|---:|
| 상단 moving support 중앙층 | `null through 64` | `11` | finite | `37` | `B36` no-cover | `6` |
| 하단 moving support 중앙층 | `null through 64` | `10` | finite | `35` | `B34` no-cover | `7` |

상단은 동적계획법과 전수열거가 모두
`[0,3,7,11,12,16,20]`, `B=37`을 선택했다.

하단은 두 방법 모두 최소값 `B=35`, child 수 `7`에는 동의했다. 그러나
동적계획법은 `[0,2,5,9,10,14,18,22]`, 전수열거는
`[0,1,5,9,10,14,18,22]`를 선택했다. 공통 suffix의 꼭짓점 `35`
edge가 앞부분의 차이를 덮기 때문에 최소값은 같지만, 동적계획법의
중간 상태 pruning은 전역 사전식 표준 경로를 보존하지 못했다. 따라서
하단의 정확한 canonical partition은 미확정이다.

결론적으로 기준점 이동은 두 `null`을 유한화했다. 하지만 등록 목표
`B=12`에는 둘 다 도달하지 못했고, 이 수치만으로 `37` 또는 `35`를
채택할 수 없다.

### 4.2 형상 gate

| 대상 | max polygon | max face width | 표면 outward | 표면 coverage | 부피 상대오차 | 양의 겹침 |
|---|---:|---:|---:|---:|---:|---:|
| 상단 | `56` | `16` | `4.1409200374e-7mm` | `4.1515439350e-7mm` | `5.9867967888e-9` | `5/15` pairs |
| 하단 | `60` | `13` | `1.6808909195e-6mm` | `4.3796578864e-7mm` | `1.4289129636e-7` | `6/21` pairs |

polygon, face 폭, 표면 `<=0.1mm`, 부피 상대오차 `<=0.5%`, 각 child의
양의 부피 조건은 통과했다. 그러나 등록된 Float32 pairwise overlap
검사에서 상단의 인접 seam `5`쌍과 하단의 인접 seam `6`쌍이 양의
교집합 부피를 냈다. 계산 실패는 `0`이다.

양의 교집합 합은 다음과 같다.

- 상단 `1.0732770688656094e-14m^3`
  (`1.0732770688656094e-5mm^3`)
- 하단 `3.0558646686052954e-13m^3`
  (`3.0558646686052954e-4mm^3`)

이는 현재 동결된 positive-volume gate에서는 명백한 FAIL이다. 그러나
이 검사에는 halfspace 안쪽 판정의 `5nm` 선형 허용오차가 들어간다.
별도 read-only 사후감사의 derived inference로, 관측 최대 부피는 parent
AABB 최대 면적에 `5nm`를 곱한 거친 규모 비교값보다 작았고, 양성은
모두 인접 seam에서만 나왔다.
따라서 수치 허용오차 띠와 Float32 반올림 가설을 지지하지만, 실제
침투가 아님을 증명한 것은 아니다.

정확한 표현은 다음과 같다.

- 확정: 등록된 5nm 경계 허용오차를 사용한 현재 overlap gate는 FAIL.
- 미확정: 실제 Float32 hull이 물리적으로 5nm 이상 관통하는지 여부.

### 4.3 음성대조군과 동결 범위

- duplicate child는 양의 겹침을 검출했다.
- 부분 겹침 box는 실제 clipping과 예상 양의 부피를 검출했다.
- 공유 면만 닿는 box는 부피 `0`으로 판정했다.
- 역순 polygon, old/new 기준점 splice, 마지막 fan triangle 누락을
  모두 거부했다.
- 다른 아홉 층 evaluation/mutation은 `0/0`; 그 map은 exact hash로
  상속했다.
- asset/USD read/write, collider materialization, Isaac/Kit/PhysX,
  Warp/CUDA, 원통, physics step, q5, contact, grasp, target/IK/path와
  설정 변경은 모두 `0`이다.

## 5. 종료 판정

Canonical design verdict:

`D388_REANCHOR_PARTITION_CONTRACT_FAIL_STOP`

Operational verdict:

`D388_ATTEMPT1_OFFLINE_WORKER_CLAIM_FAIL_STOP_NO_FINALIZE`

이유는 세 가지를 분리해야 한다.

1. 긍정적 진단: 두 old-graph `null`은 한 번의 등록 기준점 이동 뒤
   각각 `B=37`, `B=35`의 유한 경로가 됐다.
2. 채택 실패: 둘 다 `B=12`가 아니고, 현재 overlap gate에서
   `5/6`개 양성 인접쌍이 나와 geometry witness가 실패했다.
3. 방법 계약 실패: 하단은 최소값 `35`에는 합의했지만 canonical cut
   경로가 일치하지 않았다.

따라서 “기준점 이동이 아무 효과가 없었다”도 틀리고, “수리 성공”도
틀리다. 정확한 결론은 **NULL 유한화에는 성공했지만 채택 가능한
B12·무겹침·알고리즘 일치 설계에는 실패**다.

global/common, selected, adopted, complete-P34 budget은 모두 `null`,
application은 `0`, complete part count는 `null`,
`materializable_candidate=false`, live/physics/grasp는 `null`,
`p34_authored_to_cooked_identity_pass=false`, `g0a_pass=false`다.

## 6. 시각화 검사

- 결정 보드:
  `d388_two_null_partition_repair_1920x1080.png`
  (`1920x1080`, SHA-256
  `9d22ae27eee3ebb273c91a3635570b24d66879ab70d87d8d7f0d163a03a44dea`)
- Rerun screenshot:
  `d388_rerun_inspection.png`
  (`3840x2160` HiDPI 2x, SHA-256
  `f392fd1f74ae0cd22fec77519710fbd86ba5aa1f9b3cdde991bb68402480403c`)
- save-only RRD/RBL과 entity/timeline/component/footer 검증은 PASS했다.
- 수동 검사는 `8/9`, overall FAIL이다. 보드는 핵심 수치를 읽을 수
  있지만 Rerun 형상이 격자에 비해 너무 작아 child `6/7`개를 하나씩
  대응시킬 수 없고, 한글 glyph box, message-proxy 경고, loading 알림이
  남았다.

따라서 Rerun 구조 PASS를 육안 가독성 PASS로 보고하지 않는다.
canonical JSON/CSV와 SHA-256이 수치 권위다. completion summary는 만들지
않았고 finalize하지 않는다.

## 7. 발견한 비결정적 기록 결함

Evidence의 `official_sources[].applicability` 두 문장은 D388이 아니라
`D386`이라고 적힌 stale label을 상속했다. URL과 버전 및 이번 offline
nonclaim은 맞지만 case label은 잘못됐다. 원 evidence를 고치지 않고
여기에 명시한다. 이 문자열은 어떤 geometry/method gate에도 쓰이지
않았으므로 verdict를 바꾸지 않는다.

## 8. 다음 승인 경계

아직 미승인인 최소 후보:

`D389 [d388_overlap_gate_numeric_provenance_and_canonical_tie_audit]`

새 offline-only case에서 immutable D388 JSON/CSV/geometry만 읽고 두
항목을 분리 감사한다.

1. 하단 `B=35`의 모든 완성 경로를 전역 canonical key로 재정렬해
   정확한 tie-break를 확정한다.
2. 11개 인접 seam에 대해 pre-Float32와 post-Float32의 signed
   halfspace penetration, epsilon `0` 교집합, 동결 `5nm` 교집합을
   나란히 계산한다. 비인접 pair는 음성대조군으로 유지한다.

D388을 재실행하거나 gate/허용오차를 완화하지 않는다. 이 감사 전에는
`B=37/35` 선택, partition 변경, USD/PhysX materialization, 실제
`29x50mm` 원통, 물리·접촉·파지로 진행하지 않는다.

## 9. 근거

- `claudedocs/runtime_logs/grasp_track/g0a_d388/attempt1_two_null_moving_support_midlayer_partition_repair_design/d388_preregistration.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d388/attempt1_two_null_moving_support_midlayer_partition_repair_design/d388_two_null_reanchor_design_evidence.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d388/attempt1_two_null_moving_support_midlayer_partition_repair_design/d388_two_null_reanchor_witness_geometry.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d388/attempt1_two_null_moving_support_midlayer_partition_repair_design/d388_reanchored_candidate_cell_metrics.csv`
- `claudedocs/runtime_logs/grasp_track/g0a_d388/attempt1_two_null_moving_support_midlayer_partition_repair_design/d388_offline_worker_supervisor.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d388/attempt1_two_null_moving_support_midlayer_partition_repair_design/d388_offline_worker_claim.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d388/attempt1_two_null_moving_support_midlayer_partition_repair_design/d388_rerun_validation.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d388/attempt1_two_null_moving_support_midlayer_partition_repair_design/d388_manual_visual_inspection.json`
- `sim_scripts/cyl34_top_view_d388_d387_two_null_moving_support_midlayer_partition_repair_design.py`
- NVIDIA Omni Physics 107.3, `Colliders`:
  https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/collision.html
- NVIDIA PhysX 5.6.1, `GPU Rigid Bodies`:
  https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/GPURigidBodies.html
- NVIDIA PhysX 107.3, `GuConvexMesh::isGpuCompatible`:
  https://raw.githubusercontent.com/NVIDIA-Omniverse/PhysX/107.3-omni-and-physx-5.6.1/physx/source/geomutils/src/convex/GuConvexMesh.cpp

NVIDIA 자료는 convex의 vertex, polygon, face-width 조건이 서로 다른
조건이라는 version-matched 배경 근거다. D388 자체는 PhysX cook이나
GPU 호환성 판정을 실행하지 않았다.
