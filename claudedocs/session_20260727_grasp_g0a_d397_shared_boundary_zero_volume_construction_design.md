# Session 2026-07-27 — Grasp G0a D397 shared-boundary zero-volume construction design

## 1. 질문과 승인 범위

질문:

> D396이 기각한 D388의 겹치는 분할을 버리고, 하나의 Float32 경계를
> 양쪽 폐쇄 볼록체가 그대로 공유하게 만들면 P34의 실패한 source 부모
> 8개를 모두 꼭짓점 12개 이하로 다시 나눌 수 있는가?

과학 case의 신규 변수:

1. `float32_canonical_shared_plane_balanced_bsp_v1`

관측 보정 변수:

1. attempt3:
   `per_parent_failure_board_and_label_suppressed_exploded_rerun_v1`
2. attempt4:
   `annotation_safe_board_and_tighter_exploded_camera_v1`

사용자는 D397 오프라인 설계부터, 그 PASS 뒤에만 USD/PhysX identity,
`29x50mm` 원통 zero-step, 실물 물성, pose/contact 시험을 순서대로
승인했다. 따라서 D397 FAIL은 뒤 단계를 자동으로 중단한다.

## 2. 기준 상태와 입력 계보

- Git `HEAD == origin/master ==
  d354d46134fe002073642441a7d24c99fe579edd` (subject D388).
- D389-D396의 미커밋 forward-only 산출물은 사용자 소유 상태로 보존했다.
- D396은 D388 후보의 실제 완료된 두 pre-Float32 overlap을 각각
  `6.4038856253626914e-15m^3`와
  `2.4130456372851684e-15m^3`로 확정해 그 후보를 기각했다.
- D397 base는 동결된 통과 부품 `17`, exact-profile 자식 `46`, 합계
  base `63`을 상속하고, 실패한 source 부모 8개만 새 방식으로 만들었다.
- 동결된 과학 evidence와 geometry의 최종 SHA-256:
  - evidence:
    `ea7fd61c38f12b9e03f4e7154536579b831c6f85703bfd4d14e34807cdf327b6`
  - geometry:
    `b9a44d430f647e45292fe71804bd17e6f53bf37eea28913389316beac60fa623`
- base script SHA-256:
  `52745beab46bc695467dd8d676a06b30fa3ea873c7dcad685861e65cfecf4b36`

핵심 입력:

- `claudedocs/runtime_logs/grasp_track/g0a_d397/attempt2_phase_marker_payload_key_repair/d397_shared_boundary_design_evidence.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d397/attempt2_phase_marker_payload_key_repair/d397_shared_boundary_candidate_geometry.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d397/attempt2_phase_marker_payload_key_repair/d397_offline_worker_claim.json`

## 3. 사전등록한 생성법과 게이트

각 볼록 source 부모에 대해 다음 규칙을 사용했다.

1. 현재 cell의 x/y/z 좌표에서 인접 고유값의 Float32 midpoint만 후보
   plane으로 만든다.
2. 한 split의 교차점과 seam polygon을 한 번만 계산한다.
3. 그 동일한 Float32 seam 꼭짓점을 양쪽 폐쇄 sibling에 재사용한다.
4. 한 sibling은 `axis <= cut`, 다른 sibling은 `axis >= cut`에 둔다.
5. 선택한 split은 현재 cell보다 최대 child vertex count를 엄격히
   줄여야 한다.
6. 통과 후보를 최대 child 꼭짓점, 부피 불균형, 총 꼭짓점, 축/cut bits
   순으로 정렬해 첫 후보 하나만 선택하며 backtracking은 하지 않는다.
7. 이 greedy 경로의 모든 leaf가 꼭짓점 `<=12`가 되면 종료한다.

동결 게이트:

- 한 source 부모의 child `<=64`
- 전체 part `<128`
- child 꼭짓점 `<=12`
- polygon `<=64`, 한 polygon 꼭짓점 `<=32`
- parent surface/bounds 오차 `<=0.1mm`
- 부피 상대오차 `<=0.5%`
- 같은 source 부모를 대체하는 child 사이 새 양의 부피 겹침 `0`
- D372 void, frozen-OPEN clearance, contact seed, raw-surface 계약

음성대조군은 duplicate leaf, child 제거, 인위적 overlap, D396의 두
overlap witness, source count 65, total count 128의 `6/6`이었다.

## 4. 실행 전 정적 검토

독립 정적 검토가 다음 문제를 worker 전에 잡아 수리했다.

1. D385보다 줄어든 surface sampling
2. D372와 다른 seed 거리 계산
3. 실제 상한이 아닌 raw-surface proxy
4. perturbation 없는 상수 음성대조군
5. separator proof 실패를 양의 겹침으로 오해하는 표현
6. source parent/partial leaf/seam 시각자료 누락
7. Active Case와 출력 경로 검증 누락

수리 뒤 입력 해시, 환경 pin, 실제 perturbation 대조군, 실패 시 null
의미론, source/leaf/seam 계층과 forward-only 경계를 다시 정적 PASS했다.

## 5. attempt1 — 과학 계산 전 운영 실패

경로:

`claudedocs/runtime_logs/grasp_track/g0a_d397/attempt1_shared_boundary_zero_volume_construction_design/`

- preregistration `18/18`, worker preflight `10/10`: PASS
- OpenUSD version: `24.05`
- 첫 source-parent 계산 전에:
  `TypeError("_phase() got multiple values for argument 'name'")`
- 원인: phase helper의 positional parameter `name`과 payload keyword
  `name=`이 충돌했다.
- source-parent start/end: `0/0`
- geometry evaluation: `0`
- process signal: `0`

따라서 attempt1의 설계 PASS/FAIL은 `null`이다. 같은 경로를 재실행하지
않고 helper parameter만 `name -> phase_name`으로 바꿨으며 payload
field `name`은 그대로 유지했다.

## 6. attempt2 — 유일한 D397 과학 실행

경로:

`claudedocs/runtime_logs/grasp_track/g0a_d397/attempt2_phase_marker_payload_key_repair/`

- prepare: `18/18` PASS
- operational repair attestation: `6/6` PASS
- worker/retry/signal: `1/0/0`
- worker elapsed: `3.019634233787656s`
- source-parent construction attempts: `8`
- 최종 과학 verdict:
  `D397_SHARED_BOUNDARY_ZERO_VOLUME_CONSTRUCTION_FAIL_STOP`

### 6.1 완성된 두 부모

| 부모 | child | split | max child vertices | max polygons | max vertices/face | outward mm | surface coverage mm | volume relative error | positive overlap / tested pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `proximal_upper_arm_hull_b` | 8 | 7 | 12 | 8 | 6 | `5.183687541709947e-7` | `1.015403238921464e-7` | `8.481391131044211e-9` | 0 / 28 |
| `proximal_lower_arm_hull_b` | 8 | 7 | 12 | 8 | 6 | `5.183687541709947e-7` | `1.015403238921464e-7` | `8.481391520557598e-9` | 0 / 28 |

두 부모는 동일 sibling separator proof로 양의 부피 겹침 `0`을
인증했다. 이것은 두 부분 결과의 PASS이지 전체 후보의 PASS가 아니다.

### 6.2 중단된 여섯 부모

모두 `no_admissible_shared_plane_split`에서 중단했다.

- `proximal_upper_arm_hull_a`
- `proximal_lower_arm_hull_a`
- `moving_upper_backbone`
- `moving_lower_backbone`
- `fixed_backbone_left`
- `fixed_backbone_right`

마지막 diagnostic 상태:

- source parents: `8`
- partial diagnostic leaves: `46`
- registered shared seams: `38`
- 부모별 leaf:
  - fixed left/right `7/7`
  - proximal upper a/b `4/8`
  - proximal lower a/b `8/8`
  - moving upper/lower backbone `2/2`

failed terminal leaf의 diagnostic vertex 범위는 PUA `26-31`, PLA `19-28`,
MUB/MLB `16/16`, FBL `14-21`, FBR `15-22`였다. 완성 PUB/PLB는
각각 `10-12`였다. 따라서 이 결과만으로 전역 예산을 단순히 `12 -> 13`으로
올릴 근거는 없다.

### 6.3 왜 다른 게이트가 null인가

8개 부모가 모두 완성되지 않았으므로 materializable inventory가 없다.
따라서 다음 값은 FAIL 측정값이 아니라 **미실행 `null`**이다.

- source child count / total part count
- body별 count와 bounds
- jaw void
- frozen-OPEN clearance
- raw-surface direct measurement
- contact seed
- `29x50mm` cylinder
- live PhysX callback identity
- physics/contact/grasp

scope counter:

- immutable authoring USD reads: `0`
- collider materialization / USD write: `0/0`
- Isaac / Kit / PhysX / Warp-CUDA launch: `0/0/0/0`
- physics step / q5 sample / contact query: `0/0/0`
- cylinder create/write: `0`
- target/IK/path or material/mass/actuator/physics change: `0`

즉 이번 FAIL은 Isaac timeout, GPU, PhysX cook 실패가 아니다. Isaac을
아예 실행하지 않은 오프라인 생성법의 형상 FAIL이다.

## 7. 관측성과 육안검사

attempt2 자동 RRD 검증은 통과했지만, 실제 검사에서 missing glyph,
작은 exploded geometry, seam label overlap이 보여 presentation은 FAIL했다.

attempt3:

- 과학 worker `0`
- exact board `1920x1080`
- RRD/RBL 자동 계약 PASS, Viewer `1`
- manual FAIL: 중복 빨간 실패 문구가 axis text와 겹치고 Rerun 형상이
  여전히 작았다.
- operational:
  `D397_ATTEMPT3_COMPLETION_INTEGRITY_FAIL_STOP`

attempt4는 attempt2 과학 evidence와 attempt3 실패 계보만 읽었다.

- 신규 표시 변수:
  `annotation_safe_board_and_tighter_exploded_camera_v1`
- 과학 worker / geometry gate change / Isaac-PhysX / signal: `0/0/0/0`
- board: exact `1920x1080`, SHA-256
  `d1adfb3f460b77fc9f1eabe9c0f78c8813938b9ee67af9dbeebee488a9355d7f`
- RRD/RBL strict contract: PASS
- Viewer/retry: `1/0`
- RRD entities:
  source parent `8`, diagnostic leaf `46`, consolidated seam points `8`,
  total mesh `54`
- manual inspection: `8/8` PASS
- RRD SHA-256:
  `b55948c0d2ae085d70a93ba8c44b878952e73ffbd25887f4f3f6794502ed8837`
- RBL SHA-256:
  `05adf01f7be376c3819362a3a4f130efc065f8bd7f7ed0c8b07249cdc7ef278f`
- completion SHA-256:
  `4265909d583f7e31654402744d04702d50e722995250dafa2f87ce7047185c07`

Rerun screenshot에는 headless `message proxy`/loading toast가 보이지만,
geometry load와 entity/component/timeline/footer 검증을 막지 않았고
결정판을 가리지도 않았다. 이 경고는 과학 권위가 아니다.

관측 운영 verdict:

`D397_FAILURE_PRESENTATION_REPAIRED_COMPLETE`

## 8. 최종 판정

과학:

`D397_SHARED_BOUNDARY_ZERO_VOLUME_CONSTRUCTION_FAIL_STOP`

일상어로는 다음과 같다.

- 같은 경계를 양쪽 조각이 공유하게 만드는 원리는 두 부모에서는
  제대로 작동했다.
- 그러나 현재의 “축 정렬 midpoint 후보 + 매 단계 즉시 꼭짓점 수 감소
  + 최종 12개 이하” 후보군에서 결정론적으로 첫 후보만 고른 greedy
  경로는 나머지 여섯 부모를 끝까지 나누지 못했다. 다른 분기까지 모두
  기각한 결과는 아니다.
- 따라서 완성된 새 충돌체는 없고, USD에 올릴 것도 없다.
- 이것은 원통을 못 잡았다는 물리 결과가 아니다. 원통·물리·접촉은
  실행하지 않았다.
- `materializable_candidate=false`, `g0a_pass=false`.

## 9. Session progress rule 충족

D397은 실제로 FAIL 가능한 오프라인 geometry perturbation/design
evaluation이었다. 8개 부모 생성과 실제 perturbation 음성대조군 `6/6`을
실행했고, 본 후보가 2/8만 완성되어 실제 FAIL했다. attempt2 이후의
control/관측 보정은 그 실제 실패와 presentation 실패에 대한 reactive
hardening이었다.

## 10. 다음 승인 경계

USD/PhysX cook/readback, `29x50mm` 원통, 실물 물성, pose/contact 단계는
D397 PASS 조건을 충족하지 못했으므로 실행하지 않는다.

다음 최소 후보는 아직 미승인:

`D398 [d397_six_failed_parent_greedy_bsp_dead_end_provenance_localization]`

목적:

- 동결된 D397의 여섯 failed parent와 partial tree만 읽는다.
- 각 최초 stuck leaf의 모든 축/midpoint 후보가 paired split 생성,
  shared-seam/volume, strict vertex-reduction 중 어느 단계에서 탈락했는지
  계수한다.
- 각 ancestor에 greedy 선택에서 제외된 다른 admissible option이
  존재했는지만 기록한다. 새 경로를 선택하거나 geometry를 만들지 않는다.
- backtracking/depth-2 search, geometry adoption, vertex budget 변경,
  non-axis plane, USD/PhysX, cylinder/physics/q5/contact는 모두 `0`으로 둔다.

이 결과 뒤에만 “greedy branch만 바꿀지”, “backtracking을 허용할지”,
“plane family나 꼭짓점 예산을 별도 재검토할지”를 한 변수씩 선택한다.

## 11. 주요 산출물

- clean decision board:
  `claudedocs/runtime_logs/grasp_track/g0a_d397/attempt4_manual_visual_clarity_repair/d397_failure_by_parent_clean_1920x1080.png`
- replayable RRD:
  `claudedocs/runtime_logs/grasp_track/g0a_d397/attempt4_manual_visual_clarity_repair/d397_failure_exploded_tight.rerun.rrd`
- fixed blueprint:
  `claudedocs/runtime_logs/grasp_track/g0a_d397/attempt4_manual_visual_clarity_repair/d397_failure_exploded_tight.rerun.rbl`
- canonical science evidence:
  `claudedocs/runtime_logs/grasp_track/g0a_d397/attempt2_phase_marker_payload_key_repair/d397_shared_boundary_design_evidence.json`
- canonical diagnostic geometry:
  `claudedocs/runtime_logs/grasp_track/g0a_d397/attempt2_phase_marker_payload_key_repair/d397_shared_boundary_candidate_geometry.json`
- final manual inspection:
  `claudedocs/runtime_logs/grasp_track/g0a_d397/attempt4_manual_visual_clarity_repair/d397_attempt4_manual_visual_inspection.json`
- final completion:
  `claudedocs/runtime_logs/grasp_track/g0a_d397/attempt4_manual_visual_clarity_repair/d397_attempt4_completion_summary.json`
