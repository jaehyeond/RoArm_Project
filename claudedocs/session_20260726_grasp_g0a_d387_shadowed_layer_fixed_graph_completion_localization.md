# Session 2026-07-26 — Grasp G0a D387 shadowed-layer fixed-graph completion localization

## 1. 무엇을 왜 확인했는가

D386은 D385가 각 실패 부모에서 **처음 막힌 층 4개**만 계산했다. 그 결과
3개 층은 꼭짓점 한도를 `13/28/30`으로 높이면 같은 분할 그래프 안에서
완전 경로가 생겼지만, 움직이는 아래 지지부의 중앙 층 하나는 `64`까지도
완전 경로가 없었다.

그러나 D386에서는 같은 네 부모에 남아 있던 7개 층을 계산하지 않았다.
이 중 lower `z_layer_00`과 fixed-left `y_layer_00`은 D385의 부모 단위
조기중단 전에 내부적으로 `B=12`를 통과했지만, 독립 지도 항목으로
보존되지 않았다. 따라서 어느 부분만 분할법을 수리해야 하는지 전체
지도를 알 수 없었다.

D387의 질문은 다음 하나였다.

> D386에서 이름만 기록하고 계산하지 않은 7개 층에도 같은 분할 그래프와
> 같은 판정 기준을 적용하면, D385의 실패 부모 4개에 속한 11개 층 전체가
> 어디서 통과하고 어디서 막히는가?

이번 case의 신규 변수:

`d386_shadowed_layer_fixed_graph_evaluation_set_v1`

D387은 새 충돌체를 만들거나 꼭짓점 예산을 선택하는 case가 아니다.
기존 4개 결과는 해시로 상속하고, 미계산 7개만 같은 방법으로 채워
11개 층의 실패 지도를 완성하는 offline 검사다.

## 2. 승인 범위와 동결 조건

새 계산 대상은 D386 evidence에 기록된 다음 7개와 정확히 일치했다.

| body | 부모 | 새로 계산한 층 |
|---|---|---|
| `gripper_link` | `proximal_upper_arm_hull_a` | `z_layer_01`, `z_layer_02` |
| `gripper_link` | `proximal_lower_arm_hull_a` | `z_layer_00`, `z_layer_02` |
| `link5` | `fixed_backbone_left` | `y_layer_00` |
| `link5` | `fixed_backbone_right` | `y_layer_01`, `y_layer_02` |

D386에서 상속한 4개:

- `proximal_upper_arm_hull_a/z_layer_00 = 28`
- `proximal_lower_arm_hull_a/z_layer_01 = null`
- `fixed_backbone_left/y_layer_01 = 30`
- `fixed_backbone_right/y_layer_00 = 13`

동결한 조건:

- D379 authored Float32 point stream
- D385 semantic thin-layer interval과 profile fan 순서
- 연속 fan group size `1..4`
- D385 원 부모와의 clipping/intersection
- 등록 분류 범위 `12..64`
- polygon count `<=64`
- 한 polygon의 vertices `<=32`
- positive volume
- 바깥 돌출과 부모 표면 미포함 `<=0.1mm`
- topology-volume 상대오차 `<=0.5%`
- 자식 사이 positive-volume overlap `0`

실행하지 않은 것:

- alternate partition 또는 group size `5+`
- 내부 겹침 허용, tolerance/gate 완화
- 꼭짓점 예산 선택·적용
- asset/USD/collider 생성·수정
- Isaac Sim, Kit, PhysX, live callback, Warp, CUDA
- `29x50mm` 원통 생성·측정
- physics step, q5, contact, grasp
- target/IK/path 또는 material/mass/actuator/physics setting 변경

## 3. Git와 동결 입력

사용자가 D385 이름으로 commit/push한 뒤 실제 Git은 다음과 같았다.

- `HEAD == origin/master ==`
  `930b41d98576a9c0bf1dce4f3eb1c0d93df8014b`
- subject: `D385`
- D387 승인 시 worktree: clean

동결 입력 SHA-256:

- D379 evidence:
  `8eacbad796e8532c3d856b865e90dc54481f0f2003a266c3ebfaa8e93de37af5`
- D385 script:
  `ea1d76a8db9c78a3cae9de50a62e0a25283d5550346dad158e641a0da321c5ed`
- D385 evidence:
  `4ff64045d4e2e7ecc3601927d1d6c97fd1a61b636e838241f9fded6b02e3cc00`
- D385 completion:
  `2caf6c47ad563c9ad82b84d5c3367139943f95c98c62a590b3551a967def91c2`
- D386 script:
  `60b5b2d15518baa0427e44f0928a46993e78eeba45307636b234cb0b042acf8d`
- D386 evidence:
  `ae956a2b64835f4030daf104f08d239f140f8ba9b32ee9205f2b744769c51d4c`
- D386 geometry:
  `ec5016cb5ebee9930c23093a6f3211a397466137f78d93e30357f8e10744a187`
- D386 completion:
  `622c34fdb7cbd11d2b0465eda75ac1119407fcaab441a059ff40289065170b6e`

최종 실행 script SHA-256:

`39d1f9f33a3f6b36b07fdb7ae30b5f89afdd1646f5c98531e2274043cc72d9ee`

사전등록 당시 `START_HERE.md` SHA-256:

`aa85a5f63c222c5599d84efd5bfb0b90b5d6b468ad80d90e28300341cab0b450`

## 4. 실행 전 감사와 사전등록

공식 output 경로를 만들기 전에 독립 정적 감사 세 건을 수행했다. 초기
감사에서 발견한 다음 모순을 실제 worker 실행 전에 수리했다.

1. `B=12`에서 이미 통과하는 층의 raw minimax가 `12` 미만일 수 있는데,
   이를 새 선택 예산처럼 오해할 수 있었다.
2. Rerun 판정 패널이 실패 경로에서도 PASS 문구를 표시할 수 있었다.
3. bounded watchdog를 등록하면서 START_HERE는 모든 signal을 금지해
   timeout-only 소유 process-group 권한과 모순됐다.
4. 단계 표식이 존재했지만 exact-count/order 완료 gate가 없었다.
5. Rerun HiDPI 화면이 다시 4배 크기로 커지는 것을 차단하는 물리 픽셀
   계약이 없었다.
6. D386에서 복사된 미사용 구현이 남아 실제 실행 알고리즘을 혼동시켰다.

수리 결과:

- raw minimax `<12`는 참고값만 기록하고 등록 분류는 `B=12 baseline cover`
- selected/adopted sub-12 budget은 항상 없음
- Rerun PASS/FAIL 문구를 실제 map verdict에 조건부 연결
- timeout 시에만 새 D387-owned worker process group에 signal 허용
- 새 7개 층마다 graph/helper/DP/exhaustive start/end exact 1회 gate
- 유한 층만 geometry-gate start/end exact 1회 gate
- Viewer 요청 `1920x1080`, native PNG 허용은 `1920x1080` 또는
  HiDPI `3840x2160`뿐
- 미사용 D386 복사 구현 제거

최종 독립 감사 세 건은 모두 GO였다. 이때까지 D387 output과 worker
invocation은 각각 `0`이었다.

Preregistration:

- checks `23/23 PASS`
- 신규 변수 `1`
- 새 대상 `7`, 상속 대상 `4`, 교집합 `0`, 합집합 `11`
- forbidden runtime import root 없음
- worker `1`, retry `0`, watchdog `300s`, Viewer 최대 `1`
- SHA-256:
  `188edae6fd1218ebef455fdf90e1a76408385d0570af906a29a57a8032376864`

## 5. 계산 방법

각 새 층마다 다음 순서로 계산했다.

1. D385의 exact thin-layer를 다시 추출한다.
2. 연속 fan group `1..4`의 전체 후보 그래프를 정확히 한 번 만든다.
3. 동결 D385 `B=12` helper 결과를 독립 그래프의 `B=12` cover와 대조한다.
4. `vertex_count<=64`인 edge만 사용해 minimax 동적계획법을 계산한다.
5. 같은 고정 그래프의 모든 완전 경로를 독립적으로 전수 열거한다.
6. reachability 결과와 두 알고리즘의 finiteness/minimum/canonical cut을
   대조한다.
7. 유한 결과만 surface/volume/polygon/face/positive-volume/overlap gate로
   다시 확인한다.

등록 분류:

- `BASELINE_B12_COVER`: `B=12`에서 완전 경로가 있음
- `FINITE_RELAXATION_THRESHOLD_13_TO_64`: `B*-1`은 실패하고 `B*`에서
  처음 완전 경로가 있음
- `NO_COVER_THROUGH_64`: 같은 graph/gate로 `B=64`까지 완전 경로가 없음

마지막 분류도 유효한 지도 항목이다. 즉 “어디가 막혔는지”는 확정할 수
있지만, 그 값을 전역 예산으로 바꾸거나 완성 후보라고 부를 수는 없다.

## 6. 새로 계산한 7개 층 결과

| 부모/층 | 후보 | 비-꼭짓점 gate 통과 | raw minimax | 등록 분류/임계값 | 자식 |
|---|---:|---:|---:|---|---:|
| upper `_a` / `z_layer_01` | 74 | 37 | `null` | `NO_COVER_THROUGH_64` | `null` |
| upper `_a` / `z_layer_02` | 82 | 82 | 8 | `B=12` | 6 |
| lower `_a` / `z_layer_00` | 82 | 82 | 8 | `B=12` | 6 |
| lower `_a` / `z_layer_02` | 74 | 74 | 28 | `B=28` | 6 |
| fixed-left / `y_layer_00` | 70 | 70 | 9 | `B=12` | 7 |
| fixed-right / `y_layer_01` | 86 | 86 | 8 | `B=12` | 6 |
| fixed-right / `y_layer_02` | 74 | 74 | 8 | `B=12` | 5 |

새 7개 분류 합계:

- `B=12` baseline cover: `5`
- `13..64` 유한 임계값: `1` (`28`)
- `64`까지 no-cover: `1`

CSV 후보 행과 constructed geometry는 `542/542`이고, 7개 그래프의 후보
수 합 `542`와 정확히 일치했다. 비-꼭짓점 gate 통과는 `505`, 거부는
upper `z_layer_01`의 `polygon_count>64` `37`건이었다. 유한 6개 층의
독립 전수열거 완전 경로 합은 `4,858,898`이다.

`raw minimax=8/9`는 “이 그래프가 수학적으로 요구하는 가장 큰 자식
꼭짓점 수” 참고값이다. D387 의사결정 범위의 바닥은 `12`이므로 새 예산
`8/9`를 선택하거나 적용하지 않았다.

## 7. 상속 4개를 합친 11개 층 지도

| 부모 | layer 00 | layer 01 | layer 02 |
|---|---|---|---|
| upper moving support `_a` | `28` (D386) | `null` (D387) | `12` (D387) |
| lower moving support `_a` | `12` (D387) | `null` (D386) | `28` (D387) |
| fixed backbone left | `12` (D387) | `30` (D386) | N/A |
| fixed backbone right | `13` (D386) | `12` (D387) | `12` (D387) |

11개 전체 분류:

- `B=12`: `5`
- `13..64` 유한 임계값: `4` (`13, 28, 28, 30`)
- `null`: `2`

모든 11개 층은 유효한 분류를 가져 map은 `11/11` 완성됐다. 그러나 두
moving-support `_a` 부모의 **중앙 `z_layer_01`**이 각각 `null`이므로:

- combined-all-eleven finite: false
- global common vertex budget: `null`
- adopted parent-wide budget: `null`
- selected vertex budget: `null`
- selected budget application count: `0`
- complete-P34 budget: `null`
- complete source-child/total-part count: `null/null`
- materializable candidate: false
- global semantic preservation: `null`

유한값의 최대 `30`은 지도 요약용 참고값일 뿐 채택 예산이 아니다.

## 8. 왜 두 중앙 층이 막혔는가

새 upper moving-support `z_layer_01`:

- broad-profile vertices `22`
- fan triangles `20`
- candidate geometry `74/74` constructed
- frozen `polygon_count<=64` 통과 `37`
- `polygon_count>64` 거부 `37`
- 시작에서 도달 가능한 state `0..10`
- 끝으로 연결 가능한 state `17..20`
- 두 집합을 잇는 완전 경로 없음

상속 lower moving-support `z_layer_01`:

- candidate geometry `82/82` constructed
- frozen `polygon_count<=64` 통과 `40`
- `polygon_count>64` 거부 `42`
- 완전 경로 없음

따라서 D385의 실패를 “12 vertices가 너무 작아서”라고 하나로 설명할 수
없다. 두 중앙 층은 꼭짓점 수를 `64`까지 허용해도, 현재 fan grouping과
polygon gate를 함께 만족하는 연결 경로가 없다.

이 결과는 polygon gate를 없애라는 뜻도 아니다. 다음 수리는 두 중앙 층의
representation/partition을 바꾸어 polygon 수를 줄이면서 표면·부피·무겹침
gate를 유지할 수 있는지 별도 case로 검증해야 한다.

## 9. 유한 witness 형상 gate

새 유한 6개 층은 모두 등록 형상 gate를 통과했다.

| 부모/층 | 임계값 | max polygons | max vertices/face | outward mm | coverage mm | volume rel. error | overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| upper/z02 | 12 | 11 | 6 | `3.47e-15` | `4.23e-7` | `8.13e-10` | 0 |
| lower/z00 | 12 | 11 | 6 | `3.47e-15` | `4.23e-7` | `8.13e-10` | 0 |
| lower/z02 | 28 | 49 | 6 | `9.35e-7` | `8.75e-7` | `1.47e-7` | 0 |
| fixed-left/y00 | 12 | 13 | 6 | `3.63e-7` | `6.69e-8` | `7.39e-8` | 0 |
| fixed-right/y01 | 12 | 12 | 6 | `1.67e-8` | `1.39e-14` | `8.10e-9` | 0 |
| fixed-right/y02 | 12 | 12 | 6 | `2.30e-6` | `1.39e-14` | `1.64e-8` | 0 |

이 witness는 각 층의 offline 분류 증거다. USD에 올릴 완성 충돌체나
PhysX/GPU compatibility 증거가 아니다.

## 10. 실제 실행 계약

Canonical path:

`claudedocs/runtime_logs/grasp_track/g0a_d387/attempt1_shadowed_layer_fixed_graph_completion_localization/`

- actual worker: `1`
- retry: `0`
- return code: `0`
- elapsed: `9.548573459964246s`
- watchdog: `300s`
- timeout: false
- termination action: null
- residual process-group member: `0`
- phase records: `77`
- global phase order/count/monotonic contract: PASS
- per-layer phase contract: `7/7 PASS`
- method contract checks: `19/19 PASS`
- completion checks: `32/32 PASS`

Worker stderr의 Matplotlib config 임시-cache와 equal-aspect x-limit 메시지는
표시용 경고다. 원 JSON/CSV 계산과 board layout gate는 통과했다.

## 11. 시각화와 직접 육안검사

Decision board:

- path: `d387_fixed_graph_layer_map_1920x1080.png`
- exact size: `1920x1080`
- SHA-256:
  `11df068251cc036165f65b0543f95b73f675317a79b4853f343eb28babbda439`
- layout checks `77/77`, synthetic negatives `3/3`

직접 관찰:

- 4행×3열에 populated 11칸과 N/A 1칸이 모두 보임
- D386 상속 4칸은 점선, D387 신규 7칸은 실선
- finite는 초록색, `null`은 빨간색
- 두 중앙 `z_layer_01` null, 각 임계값, 우측 부모 요약, 하단
  `11/11 / global null / selected null/0 / P34 아님`이 겹침 없이 읽힘

Rerun:

- RRD SHA-256:
  `cc4c25332d691d6ceff65f624ed7056cffca3b9794915f9096dc4eb6b8bc7814`
- RBL SHA-256:
  `b375879e57dc8ca11d3aeb32320a7fabeaf7888c395fccc276d6ad667aeb11ae`
- screenshot: native HiDPI `3840x2160`
- screenshot SHA-256:
  `d7b6b2f3b2041b4ccc443bb063bbe6055a095acadf7c06ae508da2a7a396ed75`
- Rerun SDK/CLI `0.34.1`
- exact entities/components `130/130`
- exact timelines `{blueprint, log_time}`
- footer/RRD/RBL/headless Viewer return `0`
- manual checks `8/8 PASS`

Rerun headless 화면은 sandbox에서 llvmpipe CPU software renderer를 사용했고,
analytics read-only 및 message-proxy `Operation not permitted` 경고를
표시했다. RRD load, exact contract validation, screenshot 저장은 성공했다.
이 경고는 표시 성능/환경 경고이며 canonical JSON 수치 판정을 바꾸지 않는다.

## 12. NVIDIA 공식 문서 적용 범위

D387은 Isaac/PhysX를 실행하지 않았고 새 NVIDIA limit을 주장하지 않았다.
D386에서 동결해 상속한 조건의 의미만 다음 version-matched 1차 자료와
교차유지했다.

- NVIDIA Omni Physics 107.3, **Colliders**:
  https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/collision.html
- NVIDIA PhysX 5.6.1, **GPU Rigid Bodies**:
  https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/GPURigidBodies.html
- NVIDIA-Omniverse PhysX `107.3-omni-and-physx-5.6.1`,
  `GuConvexMesh::isGpuCompatible`:
  https://raw.githubusercontent.com/NVIDIA-Omniverse/PhysX/107.3-omni-and-physx-5.6.1/physx/source/geomutils/src/convex/GuConvexMesh.cpp

이 자료는 vertices, polygon count, face width 등이 서로 다른 조건이라는
맥락을 제공한다. D387의 `12`, `64`, `32`는 동결된 프로젝트 gate이며,
offline PASS를 live PhysX GPU compatibility PASS로 부르지 않는다.

## 13. 최종 판정과 다음 승인 경계

수치 판정:

`D387_SHADOWED_LAYER_FIXED_GRAPH_MAP_COMPLETION_PASS_GLOBAL_BUDGET_NULL`

뜻:

- 성공한 것: D385 실패 부모 4개의 11개 층 지도를 같은 방법으로 완성
- 실패/미완인 것: 현 분할법으로 완성 P34 후보 생성
- 두 중앙 moving-support `z_layer_01`은 `64`까지도 no-cover
- 전역/선택/적용/complete-P34 budget은 계속 `null/null/0/null`
- collider design pass: false
- P34 authored-to-cooked identity: false
- `29x50mm` 원통: 미생성·미측정
- physics/contact/grasp: `null`
- `g0a_pass=false`

`B=12`를 모든 source child의 하드 설계 목표로 계속 유지할지에 따라 다음
수리 범위가 달라진다.

- **최소 분할 변경 경로**: `13/28/30` 유한층은 진단값으로 보존하고,
  우선 완전 경로 자체가 없는 두 중앙 null 층만 수리한다. 이 경로도
  `30` 또는 다른 예산을 채택했다는 뜻은 아니며, 예산 정책은 별도 승인이다.
- **B=12 하드 목표 경로**: 두 null뿐 아니라 `12`를 넘는 `13/28/28/30`
  네 층도 함께 재설계해야 하므로 대상은 총 6개 층이다.

최소 분할 변경 경로의 다음 권장 후보는 아직 미승인인:

`D388 [two_null_moving_support_midlayer_partition_repair_design]`

범위 제안:

- immutable D385-D387 evidence만 읽는 offline-only case
- 정확히 두 `z_layer_01` null 층만 대상
- polygon gate를 제거하지 않고, 표면 `0.1mm`, volume `0.5%`,
  positive-volume overlap `0`, semantic ownership을 유지
- 새 partition/representation 변수는 1개만 사전등록
- budget 선택·적용, USD/Isaac/PhysX/live identity, `29x50mm` 원통,
  physics/q5/contact/grasp는 포함하지 않음

D388 결과를 검토한 뒤에만 repaired asset materialization/live identity를
별도 승인한다. 실제 제품 원통 authoring/시각화와 물리 파지는 그 이후에도
서로 분리된 승인 case다.

반대로 모든 층 `B=12`를 계속 하드 목표로 둘 경우에는 위 D388 범위를
사용하지 않고, 6개 non-B12 층을 대상으로 하는 별도 case를 먼저
사전등록해야 한다. 어느 정책도 D387이 자동으로 선택하지 않았다.

D387 attempt1은 동결한다. 재실행하거나 덮어쓰지 않는다.
