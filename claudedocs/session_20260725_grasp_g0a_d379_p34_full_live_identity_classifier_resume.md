# Session 2026-07-25 — D379 P34 full live identity classifier resume

## 1. 무엇을 왜 확인했는가

D372에서 교수님 안을 따라 수동으로 만든 P34 충돌체 후보는 link5 `16개`,
gripper_link `18개`, 합계 `34개`다. D373-D377은 이 34개가 USD와 PhysX callback
경로에 모두 존재하고 property query가 유효하다는 데까지 도달했다. 그러나 “34개가
있다”와 “PhysX가 실제 충돌에 쓰는 조리된(cooked) 표면이 우리가 만든 표면과 같은가”는
다른 질문이었다.

D379의 목적은 새 Isaac 실행 없이 immutable D377 callback을 D372 authored geometry와
비교하여 그 마지막 identity 질문에 답하는 것이었다.

이번 case의 신규 변수:

- `p34_full_live_identity_classifier_resume_v1`

승인 범위:

- immutable D372/D373/D377/D378 증거를 읽는 offline audit `1`
- actual offline worker `1`, automatic retry `0`, bounded watchdog
- exact `1920x1080` board와 save-only RRD/RBL

금지 범위:

- Isaac/Kit/PhysX 새 실행, USD read/write, collider 생성/재생성
- cylinder 생성/변경, physics step, q5 command/sample, contact query
- target/IK/path/pose 또는 material/mass/actuator/physics setting 변경

## 2. 부팅과 Git

- `HEAD == origin/master ==
  2acb5b99567946d343e95e61087357193da0826c`
- subject: `D377(376case)`
- D379 시작 전 worktree에는 승인된 D378 미커밋 변경만 있었다.
- D379는 그 dirty baseline을 exact preregistration 대상으로 보존했다.
- commit/push는 하지 않았다.

## 3. 버전과 NVIDIA 공식 근거

이 case 자체는 NVIDIA 모듈을 새로 import하거나 실행하지 않았다. 적용 대상은 이전 live
worker에서 확인한 설치 스택이다.

- Isaac Sim `5.1.0.0`
- Isaac Lab `2.3.0`
- Omni PhysX/schema `107.3.26`
- Kit `107.3.3`
- NVIDIA driver `580.159.03`
- RTX 4090 Laptop GPU, compute capability `8.9`, VRAM `16,376MiB`

버전 근거:

- `claudedocs/session_20260722_grasp_g0a_d375_p34_live_asset_identity_contract_repair_fail_stop.md:36-37`
- `claudedocs/runtime_logs/grasp_track/g0a_d375/d375_external_gpu_attestation.json`

공식 문서:

1. NVIDIA, **Omni Physics 107.3 — Colliders**  
   https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/collision.html  
   적용 버전: installed Omni PhysX `107.3.26`. Primitive collider는 해당 geometry에
   정밀하게 대응하고, mesh의 `convexHull`/`convexDecomposition`은 implied collision
   approximation을 만든다. mesh data에서 collision approximation을 만드는 과정을
   cooking이라고 설명한다.
2. NVIDIA, **Omni Physics 107.3 — Query the Mass and Volume**  
   https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/mass_inertia_queries.html  
   적용 버전: installed Omni PhysX `107.3.26`. D379 property-volume 및 rigid-body
   mass-state readback 의미론을 확인하는 근거다.
3. NVIDIA, **Isaac Sim 5.1.0 — Physics Simulation Fundamentals**  
   https://docs.isaacsim.omniverse.nvidia.com/5.1.0/physics/simulation_fundamentals.html  
   적용 버전: installed Isaac Sim `5.1.0.0`.

NVIDIA 문서는 D379의 특정 `0.684mm` 오차나 17개 실패를 말하지 않는다. 그 수치는 이
repo의 callback 원배열로 측정했다. 문서와 로컬 결과를 합치면 “mesh collision
approximation의 cook 결과를 authored mesh와 별도로 검사해야 한다”는 해석만 할 수 있다.

## 4. attempt1 — prepare-only 정지

경로:

`claudedocs/runtime_logs/grasp_track/g0a_d379/attempt1_p34_full_live_identity_classifier_resume/`

사전검사에서 코드가 D372 evidence의 top-level `pass`를 기대했다. 실제 D372 권위 field는
`measurement_pass=true`였으므로 `d372_evidence_pass=false` 한 항목으로 fail-closed했다.

- stage: `prepare`
- identity result: `null`
- classifier/board/Rerun: `0/0/0`
- Isaac/PhysX: `0`

고정 SHA-256:

- preregistration:
  `7d02329010f3622fc8b6e738d7ca5d255e070957329a1a48ec5ef6df3f909ba9`
- exception:
  `5c8c5778bbfb4341d7d5f03c6c685e31ca9686f31142a3142f23e6597eef665a`
- phase markers:
  `7659744965f56be5d3193e0bc1037ed0f98f513557a0c7bbb10c389a4d87d227`

attempt1은 수정하거나 재사용하지 않았다.

## 5. attempt2 — 실행 절차

경로:

`claudedocs/runtime_logs/grasp_track/g0a_d379/attempt2_d372_measurement_field_repair/`

1. D372를 `measurement_pass=true`와 exact verdict로 검사하도록 field 계약만 고쳤다.
2. preregistration `22/22`가 모두 PASS했다.
3. D343 Float32 readback `34행`, D372 Float64 geometry `34개`, D373 frozen callback,
   D377 clean-run callback/property evidence의 경로와 SHA를 재확인했다.
4. offline child를 정확히 한 번 실행했다.
5. D375에서 동결한 gate를 tolerance 변경 없이 적용했다.
6. callback polygon 하나 삭제와 mass perturbation을 포함한 failure-capable negative
   controls `6/6`을 통과했다.
7. original JSON을 먼저 확정한 뒤 board/RRD/RBL을 만들고 원본 해상도로 육안 검사했다.

실행 결과:

- worker/retry: `1/0`
- return code: `0`
- elapsed: `3.7800336838699877s`
- timeout/SIGTERM/SIGKILL/process-group residue: 모두 false
- Isaac/PhysX worker: `0`
- physics/q5/contact/cylinder/USD/pose-setting change: 모두 `0`

따라서 D379의 형상 FAIL은 runtime timeout이나 Isaac launch 실패가 아니다.

## 6. 통과한 것

다음은 모두 PASS했다.

- direct authored parts: `34/34`
- proxy-aware live inventory: `34/34`
- typed Float32 authored readback: `34/34`
- D373 frozen callback == D377 clean-run callback payload: `34/34`
- property collider rows: link5 `17`, gripper_link `19`, 모두 `VALID`
- callback polygon topology: closed/oriented `34/34`
- callback topology-volume == PhysX property-volume:
  - `34/34`
  - max relative delta `6.443186705889919e-8`
- mass/COM/inertia/principal axes:
  - both bodies PASS
  - max absolute delta `1.1368683772161603e-13`
- negative controls: `6/6`

이 결과의 뜻은 “P34 경로와 binding은 정확하고 callback 재현성도 있으며, property query가
그 callback collision volume을 읽고 있다”는 것이다.

## 7. 실패한 것

모든 identity gate를 동시에 통과한 부품은 `17/34`뿐이었다.

### body별

| Body | PASS | Total |
|---|---:|---:|
| link5 | 12 | 16 |
| gripper_link | 5 | 18 |

### 역할별

| Role | PASS | Total |
|---|---:|---:|
| structural_body | 1 | 1 |
| connector_support | 3 | 3 |
| fixed_jaw | 8 | 10 |
| fixed_jaw_backbone | 0 | 2 |
| moving_support | 0 | 4 |
| moving_jaw | 5 | 12 |
| moving_jaw_backbone | 0 | 2 |

### gate별

| Gate | PASS | FAIL | Limit | Worst |
|---|---:|---:|---:|---:|
| symmetric surface | 17 | 17 | `0.1mm` | `0.684166832184637mm` |
| bounds | 32 | 2 | `0.1mm` | `0.2500005066394806mm` |
| authored↔callback topology volume | 19 | 15 | `0.5%` | `6.677679161440082%` |
| polygon-plane residual | 33 | 1 | `1e-5m` | `7.737191610970862e-5m` |

최악 부품:

- surface: link5 `p014_fixed_backbone_right`
- bounds: gripper_link `p013_moving_jaw_09_moving_brace_04`와
  `p014_moving_jaw_10_moving_brace_05` 동률
- authored↔callback volume 및 plane residual:
  gripper_link `p002_proximal_lower_arm_hull_a`

판정:

`D379_P34_FULL_LIVE_IDENTITY_CLASSIFIER_RESUME_FAIL_STOP`

쉽게 말하면, 34개 파일/경로는 모두 맞게 연결됐지만 PhysX가 실제 충돌용으로 조리한 일부
볼록 형상은 우리가 authored한 표면과 허용오차 안에서 같지 않았다. 이 상태에서는 P34
물리시험을 “교수님 안의 정확한 34개 형상을 시험했다”고 부를 수 없다.

## 8. 시각자료와 Rerun

정확한 비교판:

`claudedocs/runtime_logs/grasp_track/g0a_d379/attempt2_d372_measurement_field_repair/d379_p34_full_live_identity_1920x1080.png`

- SHA-256:
  `788e400799698b1df5c7add30c4561a1e61f9a4bdc53125c4ec6f7602bb19be0`
- exact `1920x1080`
- D372 Float64 source, USD Float32 readback, D373 callback, D377 callback을
  link5/gripper_link 각각 나란히 보여준다.
- 원본 해상도에서 글자와 네 채널이 읽혔고 JSON verdict와 일치했다.

Rerun:

- RRD SHA-256:
  `772ac72fec0b49323efad7a2b1f76acd146251ad043bd33c0f6120153b6a8e6e`
- RBL SHA-256:
  `3b27373016c5fed7f2bfabfe6473b54e2c7e4b71cd61737f4b494152d8fcfe34`
- strict verify와 headless Viewer return `0`
- inspection PNG SHA-256:
  `8df39e72c700beb4106d30af862131e7bc0eb2f7e1c1bf2fd2692651fae1a7a3`

육안검사 FAIL 사유:

- 하단 metric panel이 `Unknown timeline`을 표시했다.
- Viewer notification이 네 번째 상단 제목을 가렸다.
- verdict row가 오른쪽에서 잘렸다.

따라서 presentation completion은
`D379_OBSERVABILITY_OR_COMPLETION_INTEGRITY_FAIL_STOP`이다. 이것은 JSON이 확정한 P34
identity FAIL을 바꾸지 않는다.

## 9. 실제 제품 29x50mm로 언제 바꿀 것인가

판매자 페이지:

https://byhandmall.com/product/%EC%9B%90%EA%B8%B0%EB%91%A5-%EB%B0%9C%ED%96%A5%EB%AA%A9-%EB%94%94%ED%93%A8%EC%A0%80-%EB%B0%9C%ED%96%A5%EA%B8%B0-%ED%99%95%EC%82%B0-%EC%9A%B0%EB%93%9C-%ED%99%95%EC%82%B0%EB%AA%A9/4519/

확인된 명목값:

- 지름 `29mm`
- 높이 `50mm`
- 재질 선택: 느티나무 또는 호두나무

페이지에 없는 값:

- 질량
- 치수 공차
- 질량중심과 관성
- 마찰계수
- 밑면 평탄도와 실제 원형도

따라서 simulation target은 먼저 `UsdGeom.Cylinder` primitive로
radius/height `14.5/50mm`만 정확히 바꾸는 것이 맞다. NVIDIA 107.3 문서는 primitive
cylinder collision representation이 geometry에 정밀하게 대응한다고 설명한다. 실제
제품에 잡기에 영향을 주는 홈, 테이퍼, 모따기 등이 측정되기 전에는 target을 별도로 convex
decomposition할 이유가 없다.

현재 table plane z는 `-12.117481868996972mm`다. 명목 50mm 물체를 그 위에 세우면:

- center z: `+12.882518131003028mm`
- top z: `+37.88251813100303mm`

D362의 moving-jaw confirmation point z는 `77.78545469045639mm`였다. 이는 새 물체
명목 top보다 `39.90293655945336mm` 위다. 따라서 과거 90mm 원통 자세를 그대로 쓰면 새
물체를 잡는 시험이 아니라 위쪽 허공을 지나갈 가능성이 높다.

형상비도 다르다.

- 역사 원통 height/diameter: `90/34 = 2.6471`
- 실제 제품 명목 height/diameter: `50/29 = 1.7241`

새 물체가 더 짧고 중심이 낮으므로 중앙 높이에서 잡으면 전도에는 기하학적으로 유리할
가능성이 있다. 그러나 질량과 마찰, 접촉력, 동역학을 모르므로 아직 “안 쓰러진다”거나
“잡힌다”고 판정할 수 없다.

## 10. 권장 순서

1. **17개 실패부품의 cook provenance와 semantic 영향 감사**  
   authored part가 실제로 convex인지, PhysX callback이 어느 표면/부피를 바꿨는지, 그
   변화가 jaw 접촉면·입구·내부 빈 공간·OPEN clearance를 바꾸는지 immutable 증거로 먼저
   검사한다. 기존 `0.1mm/0.5%` gate는 완화하지 않는다.
2. **감사 결과에 따른 P34 표현 결정**  
   cooked 결과가 semantic gate를 모두 보존하면 authored P34와 구분되는 새 cooked
   candidate로 명명한다. 그렇지 않으면 몸통/단순 지지는 box 같은 exact primitive,
   jaw는 authored↔cooked fixed-point가 되는 split convex parts로 수리한다.
3. **P34 live identity 재검증**  
   별도 승인을 받아 결정된 asset을 한 번 live load/cook/readback하고 `34/34` identity
   PASS를 요구한다.
4. **29x50 target geometry rebase**  
   primitive cylinder radius/height `14.5/50mm`만 zero-step으로 반영한다. mass, friction,
   q5, pose는 아직 바꾸지 않는다.
5. **실물 도착 후 계측**  
   여러 방향/높이의 지름, 전체 높이, 건조 질량, 밑면 평탄도, 표면/모따기를 측정한다.
6. **질량·관성·마찰 계약**  
   실측 질량을 넣고, 균일 원통 COM/inertia는 가정이면 가정이라고 등록한다. 마찰은 별도
   측정 또는 benchmark 값으로 분리한다.
7. **중앙 높이 자세, height-only**  
   fixed jaw를 물체 옆에 유지하고, 접촉 높이를 바닥에서 약 `25mm`로 낮춘다. wrist는
   동결한다. radius `14.5mm`에 맞춰 radial offset과 link4/jaw clearance를 zero-step으로
   검사한다.
8. **wrist-only 대조**  
   height PASS 뒤에만 손목 정렬을 한 변수로 바꾼다.
9. **물리 비교**  
   같은 target/pose/settings에서 A64 baseline, link5-only P34, moving-jaw-only P34, both
   P34를 각각 별도 실행한다. contact body/order, q5 접촉각, 접촉 높이 `z/H`, force/impulse,
   cylinder 이동/기울기와 step time을 비교한다.
10. **hold/lift**  
   양쪽 jaw가 안정적으로 닫힌 것이 확인된 뒤에만 든다.

29x50 geometry를 일찍 준비하는 것은 맞지만, D379에서 identity가 실패한 P34로 바로 물리를
돌리면 collider 오류와 물체 규격/자세 효과가 섞인다. 따라서 현재 임계경로는 P34 표현
차이의 provenance와 semantic 영향을 먼저 감사하는 것이다.

## 11. 종료 경계

- D379 attempt1/attempt2를 동결하고 재실행하거나 덮어쓰지 않는다.
- `g0a_pass=false`.
- P34 physical equivalence/speed, 29x50 contact/tipping, q5 closure, grasp feasibility,
  target/IK/path justification은 모두 `null`.
- 다음 최소 후보는 미승인
  `D380 [p34_failed_part_cook_provenance_and_semantic_impact_audit]`.
- D380, target geometry, 실제 물리시험은 각각 별도 명시 승인이 필요하다.

## 12. 주요 원증거

- `sim_scripts/cyl34_top_view_d379_p34_full_live_identity_classifier_resume.py`
  - final SHA-256:
    `9e4801817c12b0629b1b187cb3f035e64d4cf01d7058f1d3d607937220869f7a`
- attempt2 preregistration:
  `173c97d848139fdfe4fb538d22d50c8b3cd32250c3d7b89cff424328d921e59f`
- identity evidence:
  `8eacbad796e8532c3d856b865e90dc54481f0f2003a266c3ebfaa8e93de37af5`
- manual inspection:
  `68c5767f1b721243c8b93cc5e6747ab5001ecbfc3ac5531e890bede6f1edc6e0`
- completion summary:
  `630efec1a9815ed67d39edd296eca80fe59eb63e4e7ab3a0411b5a9fb55f8f8e`
