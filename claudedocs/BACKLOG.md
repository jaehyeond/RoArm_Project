# Backlog

These are deferred ideas. They are not active unless `START_HERE.md` names them
as the current Active Case.

- TCP/EEF dual recording.
- Caliper mm correction before real transfer.
- Render resolution: store 448x448, train/input 224x224.
- Direction diversity / goal-conditioned primitive. See `claudedocs/design_d321_goal_conditioned_primitive.md`.
- Upper friction bin remains an RL target reserve.
- Overshoot 167ep HER relabel.
- Data spec v1.
- VLA training deferred.

- 2026-07-11 (D331): `tool_surface_union` (D231) — 가동 조 `gripper_link`
  collision이 4mm proxy(convexHull, pxr 직접 확인)라 G0a alignment PASS 후에도
  G0b 파지 물리 불성립. G0b 진입 전 필수 collision 재저작 항목. (착수 금지,
  기록만.)

- 2026-07-12 (D335): `target_family_continuous_feasibility` — registered finite
  r/t set에서 raw-clear `0/2629`; 연속 domain 전체 불가능 증명은 아님. 새 물리
  변수 전에 승인된 별도 case로 top basins의 continuous/finer discriminator를
  실행할 수 있음. 현재 착수 금지.
- 2026-07-12 (D335): `target_family_orientation` — finer feasibility도 실패할
  경우 동일 anti-retreat r/t domain에 reachable wrist/tool-orientation 변수 1건을
  추가하는 non-retreat 후속안. D323 strict-axis 재도입 금지; 사용자 case 승인 전
  착수 금지.
- 2026-07-12 (D335): `grasp_depth_semantics_r_gt_17mm` — `r>17mm`는 TCP를 near
  face 밖으로 후퇴시키므로 clearance만 green이 되는 퇴행 가능성. 명시적인 새
  grasp-depth/bracketing 계약과 사용자 승인 없이는 domain 확장 금지.

- 2026-07-12 (D336 감사): `rerun_pipeline_upgrade` — 감사 결과 rerun은 D325 이후
  전 런에서 .rrd를 저장하지만 (12개) ①live viewer 0회 사용 (D326-D330은
  `--live_viewer` 플래그 보유, D332-D336은 False 하드코딩) ②D333 이후 RRD를
  열어본 증거 없음 (ok+nonzero gate만) ③물체가 solid로 추적 안 됨 (frame axes
  마커만) ④접촉력이 TextDocument로 묻힘 (Arrows3D/rr.Scalars 미사용) ⑤10-trial
  중 trial 1만 기록 ⑥D333 400 step 중 target 200만 RRD 수록. 업그레이드 후보:
  물리 런은 `rr.set_sinks(GrpcSink, FileSink)` 하이브리드(현업 표준), 원통
  solid 추적, 힘 화살표, 스칼라 타임시리즈, RRD 열람 확인 의무화. rerun-sdk
  버전 pin 주의 (D326 numpy 사건, .rrd는 인접 minor 호환만). 사용자 승인 전
  착수 금지, 기록만.

- 2026-07-14 (사용자·교수님 방향 재확인): `sim_first_cylinder_material_contract`
  — **[SUPERSEDED 2026-08-03: 사용자 결정으로 real-first 역전.** sim 사양 기준
  실물 주문제작 경로 기각. 실물 원통(명목 D29×H50, 사용자 실측 24.83g)이
  권위이며 sim 사양을 실물 실측에 rebase한다. 상세:
  `claudedocs/session_20260803_grasp_g0a_real_first_funnel_decisions_state_update.md`
  + `direction_20260708_grasp_pivot.md` 2026-08-03 섹션. 아래 원문은 보존.**]**
  — G0a에서 로봇 충돌 형상을 먼저 검증한 뒤, G0b 진입 전에 시뮬레이터의 원통
  사양을 제작 기준으로 명시적으로 동결한다. D34 x H90 치수, 질량 또는 밀도,
  원통-그리퍼와 원통-테이블의 표면 마찰, 반발계수와 필요한 경우 접촉 순응성을
  실제 readback 값과 함께 기록하고, 그 사양에 맞춰 실물 원통의 재료·충전·추·
  표면 코팅을 설계한다. 현재 `0.72kg`, 정지/동적 마찰 `1.5/1.2`는 Isaac이
  자동 선정한 재질값이 아니라 코드에 명시된 임시값이다. 엔진 기본값을 제작
  기준으로 채택할 수는 있지만, 버전 의존적인 "자동" 상태로 두지 말고 한 번
  판독한 수치를 명시 계약으로 고정해야 한다. 충돌 형상 오차는 재질/마찰로
  보상하지 않는다. D345에는 도입하지 않으며 별도 사용자 승인 전 착수 금지.

- 2026-07-15 (D351 runtime STOP): `d351_validate_phase_localization_watchdog`
  — D351 attempt2는 preflight `20/20 PASS`와 실제 GUI launch 뒤 live-binding
  artifact 전에 `3693.302s` 장기 실행됐다. 다음 후보는 과학 변수 없이
  `_make_runtime_env`/reset/corrected audit/live part `0..127`/zero-step bridge에
  forward-only phase marker와 bounded wall-clock watchdog를 추가해 장기 실행 위치를
  먼저 국소화하는 별도 D352 case다. 동일 attempt 재실행, target/IK/gate 변경,
  settle/RL 승격은 금지하며 사용자 별도 승인 전 착수하지 않는다. D352 자체는
  localization-only이고 q5 science 재개는 그 결과 뒤 별도 명시 승인 대상이다.

- 2026-07-19 (D366 pre-step STOP): `timeline_play_pending_state_commit_bridge`
  — D366은 initial PAUSE bridge와 baseline까지 통과했지만 raw `timeline.play()` 직후
  `playing_not_stopped=false`, physics no-advance에서 pose write 전에 멈췄다. 별도 승인
  zero-step control case는 explicit `Timeline.commit()` after PLAY 한 변수만 검증하고,
  before/request/post raw timeline tuple·clock·callback·joint/object bits와 cleanup close
  start/end marker를 보존한다. Cylinder write, physics step, public forward, q5/contact/science는
  전부 `0`으로 유지한다. 이 bridge 뒤 one-write/one-step/one-forward 재개도 다시 별도 승인
  대상이며, 현재 착수 금지.

- 2026-07-19 (D367 결과): 위 `timeline_play_pending_state_commit_bridge`의 제한된 제어
  질문은 PASS로 해소됐다. 남은 별도 후보는
  `simulation_app_terminal_close_attestation_contract_repair`다. Immutable D367 evidence와
  설치 소스만 읽어 terminal/non-returning `SimulationApp.close()` completion authority를
  pre-close durable sentinel + supervisor exit/no-watchdog/no-residue로 perturbation 검증한다.
  Isaac invocation, q5, physics, contact는 전부 `0`; 사용자 승인 전 착수 금지.
- 2026-07-19 (D367 결과): `tensor_step_fabric_visibility_commit_resume`는 D367 bridge PASS를
  상속해 D366의 frozen cylinder one-write/one-step/one-forward 질문을 새 forward-only
  경로에서 다시 측정하는 후보다. D367 overall completion FAIL을 사용자가 명시적으로 구분해
  수용하고 새 case를 승인하기 전에는 착수 금지. q5/contact/cap-rim/grasp science와
  target/IK/path·asset/physics 설정 변경은 포함하지 않는다.

- 2026-07-25 (D385 결과): `d385_minimum_admissible_vertex_budget_localization`
  — D385의 thin-layer/profile-cell 분할은 8개 source parent 중 4개만 완결했고,
  나머지 4개는 max-12 vertex gate에서 no-cover였다. 동일 partition,
  surface/topology-volume/overlap/semantic gate를 동결한 채 실패한 네 layer가
  요구하는 최소 child-vertex budget만 offline에서 국소화하는 다음 후보.
  자동 후보 sweep이나 gate 완화 PASS가 아니며, 결과 뒤 한 budget을 별도
  선택해야 한다. 다른 partition, internal overlap, USD/live identity,
  Isaac/PhysX/cylinder/physics/q5/contact는 사용자 승인 전 착수 금지.

- 2026-07-25 (D386 결과): 위
  `d385_minimum_admissible_vertex_budget_localization`은 완료됐다. 세
  first-observed layer의 minimum은 `13/28/30`이지만 lower moving-support
  `z_layer_01`은 `B=64`에서도 no-cover다. `82` candidates 중 `42`가 frozen
  polygon-count gate를 넘고 나머지 `40`도 complete path를 만들지 못했다.
  따라서 D386 단계에서 `30`을 즉시 전역 budget으로 선택·적용하는 것은
  정당화되지 않았고 진행하지 않았다.
- 2026-07-25 (D386 결과): 다음 최소 후보
  `d386_shadowed_layer_fixed_graph_completion_localization` — D386에서
  inventory만 한 later/shadowed `7` layers에 같은 fixed graph, `12..64`
  bounded localizer와 frozen non-vertex gates를 적용해 전체 실패 지도를
  완성한다. 새 partition이나 gate relaxation이 아니며 offline-only다.
  사용자 승인 전 착수 금지.
- 2026-07-25 (D386 이후 reserve):
  `polygon_gated_layer_minimum_partition_representation_repair_design` —
  shadowed-layer map이 완성된 뒤에만 polygon-gated layer를 위한 한두 변수
  partition/representation repair를 설계한다. polygon gate 제거, internal
  overlap, USD/live identity, actual `29x50mm` cylinder와 physics/grasp를
  결합하지 않는다. 사용자 승인 전 착수 금지.

- 2026-07-26 (D387 결과): 위
  `d386_shadowed_layer_fixed_graph_completion_localization`은 완료됐다.
  전체 11개 지도는 upper `[28,null,12]`, lower `[12,null,28]`,
  fixed-left `[12,30]`, fixed-right `[13,12,12]`이다. 두 proximal
  moving-support 중앙 `z_layer_01`만 `B=64`까지 no-cover이며, finite
  maximum `30`은 diagnostic only다. 전역/채택/선택/P34 budget과 완성
  count는 null, application `0`, materializable=false다.
- 2026-07-26 (D387 이후 조건부 최소 후보):
  `two_null_moving_support_midlayer_partition_repair_design` — 최소 분할 변경
  정책을 사용자가 선택할 때만, exact 두 null `z_layer_01`을 대상으로
  offline one-variable representation/partition repair를 설계한다. 나머지
  9개 지도 항목과 polygon/face/surface/volume/no-overlap gate는 동결한다.
  `30` 또는 다른 budget 선택·적용, USD/PhysX, 실제 원통, 물리/접촉/파지는
  포함하지 않는다. 사용자 승인 전 착수 금지.
- 2026-07-26 (D387 이후 정책 분기):
  `all_non_b12_layer_redesign` — 모든 source child에 `B=12`를 계속 하드
  목표로 둘 경우 수리 대상은 두 null이 아니라 `13/28/28/30` 네 층까지
  포함한 non-B12 6개다. 이는 최소 분할 변경 경로와 다른 설계 정책이므로
  사용자 명시 선택과 별도 case 승인 전 착수 금지.

- 2026-07-26 (D388 결과): 위
  `two_null_moving_support_midlayer_partition_repair_design`은 등록한 한 번의
  cyclic re-anchor로 두 old-graph null을 diagnostic `B=37/B=35` 유한
  경로로 바꿨다. 그러나 둘 다 B12가 아니고, 등록된 5nm halfspace
  tolerance overlap gate에서 인접 seam `5/6`쌍이 양의 부피로 판정됐으며,
  하단은 DP와 전수열거의 최소값35는 같아도 canonical cuts가 달랐다.
  따라서 budget/partition 채택은 `null`, materializable=false다.
- 2026-07-26 (D388 이후 최소 후보):
  `d388_overlap_gate_numeric_provenance_and_canonical_tie_audit` — immutable
  D388 JSON/CSV/geometry만 읽는 offline-only case에서 (1) 하단 B35의
  global canonical tie-break와 (2) 11개 인접 seam의 pre/post-Float32
  signed penetration, epsilon0 대 frozen-5nm 교집합을 독립 감사한다.
  D388 재실행, 새 partition, tolerance/overlap gate 완화, budget 선택,
  USD/PhysX/Isaac, 실제 원통, physics/q5/contact/grasp는 사용자 별도 승인
  전 착수 금지.

- 2026-07-27 (D398 resume 분석, 미승인): `sdf_collider_representation_reeval`
  — D384–D398이 막힌 "적은 볼록조각으로 오목 그리퍼 정확 재현" 문제를
  구조적으로 우회하는 대안. 설치 `PhysxSchema 107.3.26`에
  `PhysxSDFMeshCollisionAPI`(`schema.usda:1043`, `sdfResolution=256`)와
  cone/cylinder 정확 `physxCollisionCustomGeometry` 토큰이 이미 존재하고,
  버전일치 Omni Physics 107.3 Colliders 문서상 SDF는 동적·kinematic
  비볼록 형상을 보존하는 공식 옵션 중 하나다. 목표는 authored↔cooked 0.1mm identity 완성이
  아니라 D362 전도(밀어 쓰러뜨림)가 실제로 개선되는지 **물리로 처음 재측정**.
  단 SDF도 memory/perf·thin-feature 접촉·articulation-link 적용에 자체 검증
  필요(무비판 채택 금지). 신규 case 승인 + preregistration(신규 변수/gate/
  산출물 경로) 제시 전 착수 금지. 상세:
  `claudedocs/session_20260727_grasp_g0a_d398_resume_verification_and_sdf_reevaluation.md`.

- 2026-07-27 후속 교정: 사용자는 SDF 방향과 D400 번호를 선택했고,
  preregistration은 작성·검토 완료됐다. 다음 경계는 두 D400 script,
  reviewed-script hash attestation, proposed runtime hash tuple만 만드는
  no-Isaac 구현·정적검토다. actual runtime은 그 tuple 파일 SHA를 사용자가
  명시해 승인한 뒤의 별도 단계다.
  D400은 `gripper_link`만 A64→SDF(resolution 256)로 바꾸는
  configuration/load-admission/global-cook-drain/rigid-owner-enumeration
  preflight이며, `link5=A64`, product
  cylinder/q5/contact/controlled physics=0으로 제한한다. 전역 cook queue와
  property query는 per-prim SDF identity 또는 실제 articulation collision
  participation 권위가 아니다. 65/66 property-query 행수도 아직 실측값이
  아니라 미래 worker가 실패 가능하게 검증할 예상 계약이다. 이 권위 교정과
  구현→runtime 승인 분리는 `DECISIONS.md` D400-P1/D400-P2가 담당한다.
  D401은 non-product known-box contact-positive articulation response gate
  (cylinder 사용 금지), nominal
  `29x50mm`와 height/radial pose는 D402 zero-step, 실측 mass 뒤 A64 physics
  baseline은 D403, 동일 계약의 gripper-only SDF 비교는 D404로 분리한다.
  D362 cylinder는 analytic일 가능성이 높지만 runtime carb 값과 PhysX
  geometry type 미기록으로 exact-confirmed가 아니다. 또한 위 154-156행의
  “유일 옵션”은 너무 넓은 표현이고, D398 resume ledger의 “convex gate가
  likely unsatisfiable”도 미입증이다. 안전한 권위는 SDF가 공식 호환표 안의
  동적 오목형상 보존 경로 중 하나이며, D398은 선택된 greedy/max-12 계보의
  dead end만 국소화했다는 것이다. D398의 “label repair first” 순서만
  사용자 후속 선택으로 보류했으며 D398 verdict와 D399 예약은 유지한다
  (`DECISIONS.md` D398-F1). 상세:
  `claudedocs/session_20260727_grasp_g0a_vertex12_cylinder_d400_scope_review.md`,
  `claudedocs/session_20260727_grasp_g0a_d400_gripper_sdf_live_cook_articulation_preregistration.md`.

- 2026-07-29 (D407 결과): approved A/B physics는 A64 control과 gripper
  SDF res256 treatment 모두 500-step trace를 완주했고 두 leg에서
  moving-jaw contact 뒤 object motion을 관측했다. 그러나 필수 live manual
  JSON이 300초 안에 게시되지 않았고 Rerun capture의 notification/hover
  overlay 및 B force-arrow 침범도 확인되어 전체 D407은
  `manual_inspection` FAIL-STOP으로 동결됐다. B step 500은 아직 force와
  velocity가 큰 비정착 상태이므로 더 작은 최종 tilt를 stability 개선으로
  해석하지 않는다.
- 2026-07-29 (D407 이후 조건부 최소 후보):
  `d407_manual_observability_completion_repair` — immutable D407 trace/RRD만
  읽는 observability-only case에서 overlay 없는 Rerun capture, bounded
  force-arrow 표시, runtime 전 pre-armed atomic writer와 controller
  PID/phase handshake를 검증한다. D407 completion/manual JSON 덮어쓰기,
  D407 attempt 재실행, Isaac/PhysX physics step, q5/contact query, 새 물리
  표본, stable-grasp/SDF-improvement 주장은 포함하지 않는다. 사용자 별도
  승인 전 착수 금지.
- 2026-07-29 (D408 승인): 위 후보를 D408 Active Case로 승격했다. 승인
  범위는 설계·정적 준비·attestation·승인용 SHA tuple까지이며, actual
  read-only replay/capture는 tuple-file SHA를 인용한 별도 승인 전 실행하지
  않는다. 신규 과학 변수는 0, 운영·관측성 변수는
  `d407_clean_view_capture_and_bounded_force_arrow_repair_v1`과
  `prearmed_atomic_manual_writer_pid_phase_handshake_v1` 두 개다.
- 2026-07-29 (D408 정적 준비 완료): A~M 13/13, checks 73/73,
  negative fixture 26/26 PASS와 reviewed attestation을 거쳐 tuple-file
  SHA `97c7ca51f8116053fcdc59aa9572669231d4abeb66022ed4e59c9e61af28e1ff`를
  고정했다. 실제 read-only replay/capture는 이 SHA를 인용한 사용자
  별도 승인 전 실행 금지다.
- 2026-07-29 (D408 actual 완료): 위 tuple 승인으로 controller 1회,
  software Rerun viewer A/B 각 1회, production writer 1회, retry 0을
  실행했다. 실제 5개 화면의 11/11 수동 판독과 atomic publication이
  완료되어 observability repair는 PASS했다. D407 FAIL, science null,
  `g0a_pass=false`는 그대로이며 D408 attempt1은 동결한다. 새 물리 질문은
  새 case·tuple·사용자 승인으로 분리한다.

- 2026-08-03 (사용자 결정 ④ 이연): `isaac_grasping_sdg_grasp_editor_evaluation`
  — NVIDIA 공식 grasp 라벨 도구(isaacsim.replicator.grasping Grasping SDG,
  Grasp Editor, GraspGen/GraspDataGen) 채택 평가. flying-gripper 가정이라
  5-DOF reachable family 필터(핵심 rung 1)를 대체 못 하고, 설치 Kit 107.3
  에서의 확장 존재/버전 정합 미확인, isaaclab env 설치 시 D326 numpy/psutil
  pin 리스크. 라벨 대량 생산이 필요해지는 G2(형태 다양화) 진입 시 재검토.
  사용자 승인 전 착수 금지.
- 2026-08-03 (장기 기록 항목 — 사용자 결정 4건에는 미포함, 세션 doc §6
  참조): `closed_loop_grasp_deployment` —
  배포 단계에서 열거-라벨 기반 open-loop 재생만으로는 취약하다는
  GG-CNN/QT-Opt 계보 교훈. real-first 열거-라벨 funnel(D409~) 완성 후의
  후순위 항목. 착수 금지, 기록만.

- 2026-08-04 (사용자 제안 — 기록만, 착수 금지):
  `object_agnostic_best_two_region_selection` — "물체마다 가장 잘 잡히는 두
  접촉영역(A=고정 조 / B=가동 조)을 먼저 고르고, 그 영역을 기준으로
  파지·hold·이송을 학습한다"는 일반화 구상. 계보 = Nguyen 1988 독립 접촉영역
  / Ferrari-Canny 1992 / Dex-Net / GraspGen (2026-08-03 문헌 조사 41건,
  `session_20260803_grasp_g0a_real_first_funnel_decisions_state_update.md` §1).
  현재 착수 불가 사유 4건: ① 두 영역 판정식이 파지 성공을 예측한다는 증거가
  0 — D409 attempt1은 FULL 0 / A∧B 0, 실물·sim 통틀어 성공 파지 1건 미확보
  (`g0a_pass=false`). ② 판정식이 평면 물체에서 축퇴 — 면-대-면 밀착(최선)과
  모서리 찍기(최악)가 둘 다 top-edge 여유 0.000mm (D411 ③). 재설계 선행 필요.
  ③ 기하 라벨 단독 학습 승격 금지 (Kappler 2015 / Rubert; START_HERE
  Open Risks P3). ④ 비대칭 그리퍼에서 A/B는 물체만의 함수가 아니라
  (물체 × 가동 조 스윙 스윕)의 함수 (D410) → "물체에 고유한 좋은 두 영역"
  라벨은 성립하지 않으며, 그리퍼가 바뀌면 라벨이 무효가 된다.
  진입 시점 = 단일 물체 실물 파지 1건 성공 + 판정식 재설계 이후 G2(형태
  다양화). 사용자 승인 전 착수 금지.

## 2026-08-05 (19th) — 실리콘 테이프 부착 (실기 전이 시점까지 보류, 사용자 지시)

- 사용자 2026-08-05: "실리콘 테이프는 구매했으나 미부착 — 시뮬 트랙 미반영, 실기 전이
  시점 BACKLOG." 8/04 "누수방지 실리콘도 붙이니 hold도 더 잘되고"(14th doc:10)는
  구매·계획 단계 발화로 정리(사용자 명시 재해석, HARD RULE #18) — 14th doc:15-17
  "실리콘 감기전에 ... 잡아봤는데 잡혔어"가 원통 파지 = 맨 그리퍼임을 뒷받침.
- 트리거: T4(실물 재현) 진입 시 부착 여부 결정. 부착 시 선행 2건:
  ① D414 ① 무효화 범위 검토(조 형상 변경 → 동결 증거 무효), ② `g0b_d420`
  `t3_mass_friction_contract.md` §2 전면 재저작(맨 그리퍼 기준 마찰 계약 무효).
- sim 트랙(T2/T3)은 맨 그리퍼 기준 유지 — 테이프 기하 비용은 원통 기준 미계산
  (D417-R1 ② 재인용 금지 조건 유지).

- 2026-08-13 (57th, D444): `rim_pinch_tilt_case` — **교수님 기움 허용 컨펌 대기,
  컨펌 전 착수 금지.** 변수 2개: ① 접근 기움 θ∈[6°,35°] (방향은 D432의 팔 수직 평면,
  φ/depth는 동결 n8/n9 산출물 핀), ② 닫힘 목표 q5 14~22° + 24° 근방 세밀화 (D431 ⑥).
  근거: D431 (θ 6°부터 bite 양수, T1 실물 물림 0~12 mm 밴드 ↔ θ 6~29° 정량 대응,
  최적 θ35°에서 +14.97 mm) + D432 (기운 자세 IK 도달 가능). D419 "수직 상부 접근"이
  교수님 지시라 HARD RULE #18상 단독 변경 불가 — 설득 논거는 D431 ③ ("실물이 실제로
  한 0~35° 기움을 sim에 되돌려 놓는 것") + `g0b_d444` fg1 결과.
- 2026-08-13 (57th, 상태 갱신): 2026-08-03 항목
  `isaac_grasping_sdg_grasp_editor_evaluation`은 부분 해소 — 확장 존재/버전 정합은
  `isaacsim.replicator.grasping 1.0.9`로 확인 완료(D326 pin 무손상), antipodal sampler는
  `g0b_d420/t3s_side_sdg1·sdg2`에서 실사용, flying-gripper 물리 평가는 `g0b_d444`
  active case로 승격 (D444). 잔존 부분 = "라벨 대량 생산" 용도 평가(G2 진입 시 재검토).

- 2026-08-13 (61st 종료 후 재개 세션): `isaaclab_parallel_grasp_env` — ba1의
  SETTLE→APPROACH→CLOSE→LIFT→HOLD 파지 시퀀스를 Isaac Lab 병렬 env(RL 학습 또는
  대량 demo 생성)로 확장하는 아이디어 (사용자 질문에서 발생, Variable Ladder상
  즉시 구현 금지 → 등재만). 성립 전제: ① B601 수리본 USD 1회 저작(현재 p20의
  런타임 수리 — world-bake+resetXformStack + split2 convex 충돌 이식 — 를 자산
  파일로 굳힘), ② GPU PhysX 파이프라인 semantics 검증(해석적 Cylinder collider,
  Isaac Lab ContactSensor vs CPU contact report 이벤트 — NVIDIA 공식 문서 버전
  대조 필수), ③ env별 물체 pose 랜덤화 시 offline IK 1개로 불가 → env별
  differential IK 또는 RL policy로 재설계(j5 ±90° 도달성 여유 좁음 — standoff
  0.10 전 방위 불가, D448), ④ 4090 Laptop 15.6GB VRAM에서 env 수 실측(state 기반
  RL 수백~수천 env 예상, 비전 병렬은 급격히 무거움; B200 반납 — HARD RULE #27),
  ⑤ 3월 Isaac Lab 포기 원인 재독(critical_analysis_isaaclab_vla_scaling_20260326.md).
  RoArm 그리퍼 병렬화는 무의미(fg1 0/13 — 기하 병목, 시도 횟수 문제 아님) —
  B601 자산 전제 = 사실상 구매 결정과 연동. **사용자 승인 전 착수 금지.**

- 2026-08-13 (61st 종료 후 재개 세션, 추가): `b601_stacking_long_horizon_ladder` —
  원통 여러 개를 집어 위로 쌓는 장기(long-horizon) task 사다리 (사용자 질문에서
  발생, 등재만). 전제 = B601 자산(fg1 0/13로 RoArm 그리퍼는 사다리 자체 불성립)
  + 물체=원통 유지(스펀지 # tower로 전환 시 HARD RULE #19/#20 원문 재독 의무).
  사다리: Step A `ba2` place+release(신규 변수: place phase/목표 지점 — 사전 해석
  필수: side 파지 시 블레이드 하단 클리어런스) → Step B `ba3` 2단 쌓기(물체 2개/
  스택 위 place — 동심 오차·전도·도달성 재스캔, j5 ±90 여유 좁음) → Step C N≥3
  체인 — **여기까지 전부 Isaac Sim 단일 env 스크립트로 가능, Isaac Lab 불필요** →
  Step D 자율화(여기서부터 Isaac Lab 필수): D-RL(병렬 RL — 장기 sparse reward
  최난도, sub-skill 분해/커리큘럼 전제) vs D-Demo(스크립트 expert 병렬 랜덤화 →
  demo 대량 생성 → VLA 학습 — HARD RULE #2 파이프라인 접속, 이력상 더 현실적)
  → Step E 실기 전이(비주장 영역, D448 ④). [[isaaclab_parallel_grasp_env]] 전제
  ①~⑤ 공유. **사용자 승인 전 착수 금지, case당 신규 변수 1~2개 유지.**

- 2026-08-14 (62nd, 상태 갱신): `b601_stacking_long_horizon_ladder` Step A는
  `g0d_d449` `ba2`로 실행됨 — verdict `BA2_TCP_TRACK_FAIL`(D449): 물리 게이트
  전부 PASS(안착 실재)였으나 손목 후행 체적이 이웃 pedestal A에 얹혀(60.4 N)
  release 전 TCP 19.82 mm. Step A 재시도 = **ba3**(B 재배치 + 전 지지물 × 손목
  후행 체적 클리어런스 스윕 + D449 ③ 계측 수정) — **사용자 승인 대기**. Step B
  (원통 위 원통)는 ba3 성공 전 진입 금지.
