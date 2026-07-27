# Session 2026-07-27 — Grasp G0a D398 greedy-BSP dead-end provenance localization

## 1. 무엇을 왜 확인했는가

D397은 여덟 source parent 중 두 개만 완성했고 여섯 개는
`no_admissible_shared_plane_split`에서 멈췄다. D398의 질문은 형상을 다시
설계하는 것이 아니라, 동결된 여섯 partial tree에서 다음 두 사실만
국소화하는 것이었다.

1. 각 최초 막힌 leaf의 x/y/z 인접 Float32 중간점 후보가 어느 판정
   단계에서 처음 탈락했는가?
2. 그 leaf에 이르는 조상에서 D397이 선택하지 않은 다른 admissible
   후보가 하나라도 있었는가?

이번 case의 신규 변수:

`six_failed_parent_axis_midpoint_option_rejection_provenance_v1`

## 2. 승인·동결 경계

허용:

- immutable D397 attempt2 evidence/geometry/worker claim과 동결 구현만 읽기
- 이미 선택된 D397 경로의 ephemeral in-memory 재생
- 막힌 leaf의 후보를 판정하기 위한 ephemeral split 계산
- 조상별 미선택 admissible 후보 존재 여부 Boolean 기록

금지:

- 새 branch 선택, backtracking, depth-2 search
- 후보 child 형상 저장 또는 채택
- 꼭짓점 예산, 평면군, 허용오차, geometry gate 변경
- USD/asset/collider, Isaac/Kit/PhysX/Warp/CUDA
- 원통, physics step, q5, contact, target/IK/path/settings
- hardware, signal, commit, push

실행 전 Git은 `HEAD == origin/master ==
7736c73910aa5756ef1560ee55640ba005faa012`였고 worktree는 clean이었다.
D398 변경은 commit/push하지 않았다.

## 3. 입력 계보와 사전검사

동결 입력 SHA-256:

- D397 evidence:
  `ea7fd61c38f12b9e03f4e7154536579b831c6f85703bfd4d14e34807cdf327b6`
- D397 geometry:
  `b9a44d430f647e45292fe71804bd17e6f53bf37eea28913389316beac60fa623`
- D397 worker claim:
  `2bac06043e35e095660ed3a0562930f98425a9eba436dc5b72e58f313ed1ed79`
- D397 base script:
  `52745beab46bc695467dd8d676a06b30fa3ea873c7dcad685861e65cfecf4b36`
- D397 attempt2 wrapper:
  `bd95aa1cadb21e4f192c3171596585cc01002dfd951537c68035afd88230286a`

D398 script SHA-256:

`cc92fa379cfbf299b0fd48a16fe137029951c9d17c2a83a14a751fffa8f6dba3`

Prepare는 모든 등록 check를 통과했다. 두 독립 정적 검토도 D397과 같은
후보 생성·분할·seam/volume·strict-reduction 순서, frozen selected-path
재생, Boolean-only 공개 조상 계보, 금지 범위 0 gate, 실제 변조
음성대조군을 확인했다.

## 4. 관찰 가능한 실행 순서

1. D397의 여섯 final forest와 leaf payload/path identity를 재생해
   bit-exact 일치를 확인했다.
2. D397 selector와 같은 순서로 각 final forest의 최초 막힌 leaf를
   다시 골랐다.
3. 그 leaf의 모든 raw x/y/z 인접 중간점 후보를 다음 순서로 분류했다.
   - midpoint candidate generation
   - paired split creation
   - seam/volume validity
   - strict vertex reduction
   - admissible
4. 독립 trace와 frozen D397 `_candidate_splits`의 빈 admissible set이
   일치하는지 확인했다.
5. 선택 경로 조상에서는 공개 결과를
   `node_id + unselected_admissible_option_exists`로만 제한했다.
6. 누락 row, 거짓 admissible 승격, leaf hash 변조, 조상 Boolean 반전,
   selected cut bit 변조의 음성대조군 `5/5`가 모두 거부되는지 확인했다.

Worker는 정확히 `1`회, retry `0`, signal `0`으로 실행됐다. 계산
elapsed는 `5.820609834045172s`, supervisor elapsed는
`6.199380008969456s`, return code는 `0`이었다.

## 5. 원 JSON 수치 결과

| 실패 부모 | 최초 막힌 leaf 꼭짓점 | raw 후보 | 최초 탈락: strict vertex reduction | 조상 수 | 미선택 admissible 후보가 있던 조상 |
|---|---:|---:|---:|---:|---:|
| `proximal_upper_arm_hull_a` | 31 | 38 | 38 | 1 | 1 |
| `proximal_lower_arm_hull_a` | 28 | 45 | 45 | 3 | 3 |
| `moving_upper_backbone` | 16 | 18 | 18 | 1 | 1 |
| `moving_lower_backbone` | 16 | 18 | 18 | 1 | 1 |
| `fixed_backbone_left` | 21 | 30 | 30 | 4 | 4 |
| `fixed_backbone_right` | 22 | 32 | 32 | 4 | 4 |
| 합계 | — | 181 | 181 | 14 | 14 |

단계별 합계:

- midpoint 생성 탈락 `0`
- paired split 생성 탈락 `0`
- seam/volume validity 탈락 `0`
- strict vertex reduction 탈락 `181`
- admissible `0`

즉, 막힌 leaf 자체에서는 181개 후보 모두 분할체와 경계/부피
판정까지 갔지만, 어느 후보도 “가장 복잡한 자식의 꼭짓점 수가 부모보다
엄격히 감소”하지 못했다.

반대로 그 leaf에 이르는 조상 14개 모두에는 D397이 선택하지 않은 다른
admissible 후보가 있었다. 따라서 D397의 greedy 경로는 locally forced가
아니었다. 그러나 다른 후보를 따라가지 않았으므로 그것이 전체 tree를
완성하는지는 `null`이다.

평가 accounting:

- trace raw split evaluations `991`
- frozen-parity raw split evaluations `991`
- ephemeral total `1982`
- replay diagnostic cells `30`
- 새 branch/저장·채택 후보 형상 `0/0`

수치 verdict:

`D398_SIX_FAILED_PARENT_GREEDY_BSP_DEAD_END_PROVENANCE_LOCALIZED`

진단 결론:

`AT_LEAST_ONE_FROZEN_GREEDY_ANCESTOR_HAD_AN_UNSELECTED_ADMISSIBLE_OPTION_COMPLETION_FEASIBILITY_NULL`

## 6. 시각자료와 최종화

수치 비교판:

`claudedocs/runtime_logs/grasp_track/g0a_d398/attempt1_six_failed_parent_greedy_bsp_dead_end_provenance_localization/d398_greedy_dead_end_provenance_1920x1080.png`

- exact `1920x1080`
- 여섯 부모, 후보 수, 탈락 단계, 조상 Boolean, null 경계가 읽힘
- 글자 겹침·잘림 없음

Rerun:

- RRD/RBL/validation 자동 계약 PASS
- Viewer/retry `1/0`
- screenshot `3840x2160`
- 동결 source parent와 최초 막힌 leaf 형상은 보임
- 새 대안 후보 형상은 없음

수동검사에서 Rerun의 `fixed_backbone_left/right`와 일부
moving/proximal 빨간 이름표가 겹쳤다. 우측 상단에는 비핵심
`message proxy server crashed: Operation not permitted` 경고도 보였다.
따라서 필수 항목 `no_text_overlap_or_clipping`을 거짓으로 기록했다.

수치 결과와 금지 범위는 PASS지만 전체 completion은 다음으로 정직하게
중단했다.

`D398_COMPLETION_INTEGRITY_FAIL_STOP`

재실행이나 같은 경로 덮어쓰기는 하지 않는다.

## 7. 해석과 다음 승인 경계

- D398은 “max-12가 너무 작다” 또는 “axis midpoint 평면군이 불가능하다”를
  증명하지 않았다.
- 현재 최초 막힌 leaf의 직접 원인은 strict-reduction 조건이다.
- 더 중요한 새 정보는 모든 선택 경로 조상에 다른 admissible 후보가
  있었다는 점이다. 이것은 D397의 greedy 선택이 유일하지 않았다는
  증거이지, backtracking 성공 증거는 아니다.
- 완성 collider, materializable candidate, live identity, 원통 물리,
  contact, grasp는 모두 여전히 `null`; `g0a_pass=false`.

다음 최소 후보는 별도 승인
`D399 [d398_rerun_label_deconfliction_observability_repair]`이다. immutable
D398 evidence/display만 읽어 label 위치와 발표 화면만 수리하고
science worker, branch search, geometry, USD/Isaac/PhysX/physics를 `0`으로
유지해야 한다.

D399 이후 별도 승인을 받을 경우에만, max-12·평면군·gate를 동결한 채
조상 대안 branch를 따라가는 bounded completion search를 하나의 새
변수로 시험할 수 있다.

## 8. NVIDIA 공식 문서에 비춘 collider 분해 의미

설치 기준은 Isaac Sim `5.1.0.0`, Isaac Lab `2.3.0`, Kit `107.3.3`,
Omni PhysX `107.3.26`이다.

NVIDIA 문서의 일반 순서는 다음과 같다.

1. 필요한 형상을 충분히 표현하면 box/sphere/capsule 같은 primitive를
   먼저 쓴다.
2. 다음으로 효율적인 선택은 convex mesh다.
3. 구멍·입구·오목한 접촉면을 보존해야 할 때만 여러 수동 collider 또는
   convex decomposition을 쓴다.
4. 한 rigid body 아래의 여러 child collider는 하나의 rigid body로
   동작한다.
5. hull 수가 적을수록 일반적으로 빠르므로, contact/clearance 판정을
   바꾸지 않는 가장 단순한 표현을 찾는다.
6. 고정밀 동적 비볼록 접촉이 꼭 필요하면 SDF가 별도 선택지지만,
   자동으로 기본 선택할 이유는 없다.

따라서 D397처럼 모든 seam과 leaf를 세밀하게 추적하는 것은 보통의 최종
robot collider 제작 절차가 아니다. 지금 프로젝트에서는 저개수 후보가
턱의 열린 공간을 가짜 고체로 메우거나 false contact를 만들었던 이력이
있어, 어느 단순화가 문제를 만들었는지 진단하는 연구 절차다. 최종
production 후보는 link5 비접촉 몸통을 단순 primitive/few convex로 두고,
고정 턱과 움직이는 턱의 실제 안쪽 접촉면·끝단·입구만 필요한 만큼
세분화한 뒤 물리 비교로 최소 개수를 결정하는 편이 공식 권고와 맞는다.

공식 자료:

- NVIDIA Isaac Sim 5.1.0, *Physics Simulation Fundamentals*:
  https://docs.isaacsim.omniverse.nvidia.com/5.1.0/physics/simulation_fundamentals.html
- NVIDIA Isaac Sim 5.1.0, *Performance Optimization Handbook*:
  https://docs.isaacsim.omniverse.nvidia.com/5.1.0/reference_material/sim_performance_optimization_handbook.html
- NVIDIA Omni Physics 107.3, *Colliders*:
  https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/collision.html
- NVIDIA PhysX 5.1.3, *PxConvexMeshDesc*:
  https://nvidia-omniverse.github.io/PhysX/physx/5.1.3/_build/physx/latest/class_px_convex_mesh_desc.html

설치 Omni PhysX extension은 `107.3.26`이지만 그 내부 PhysX SDK exact
semver를 이번 offline case에서 별도로 판정하지 않았다. 따라서 PhysX
5.1.3 API 문서는 버전 일치 주장 없이 GPU 의미를 보강하는 자료로만
사용한다. 그 문서의 GPU-compatible convex 제한 `64`
vertices/polygons는 한 convex hull의 GPU 호환성 제한이다. D397의
`12 vertices/child`는 프로젝트가 만든 분해 gate이며 NVIDIA
기본값·최적값·GPU 한계가 아니다.

## 9. 핵심 증거

- `claudedocs/runtime_logs/grasp_track/g0a_d398/attempt1_six_failed_parent_greedy_bsp_dead_end_provenance_localization/d398_greedy_dead_end_provenance_evidence.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d398/attempt1_six_failed_parent_greedy_bsp_dead_end_provenance_localization/d398_offline_worker_supervisor.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d398/attempt1_six_failed_parent_greedy_bsp_dead_end_provenance_localization/d398_manual_visual_inspection.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d398/attempt1_six_failed_parent_greedy_bsp_dead_end_provenance_localization/d398_completion_summary.json`
- `sim_scripts/cyl34_top_view_d398_d397_six_failed_parent_greedy_bsp_dead_end_provenance_localization.py`
