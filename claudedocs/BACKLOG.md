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
  따라서 `30`을 전역 budget으로 선택/적용하는 방향은 폐기한다.
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
