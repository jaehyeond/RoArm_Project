# START_HERE.md

Last updated: 2026-07-29 KST. **D408 승인된 read-only observability
replay는 controller 1회, software Rerun viewer A/B 각 1회, manual writer
1회, retry 0으로 PASS했다. D408 attempt1은 동결했다. 이 PASS는 화면·수동
판정 전달 경계의 수리 성공이며, D407 물리 과학 verdict는 여전히
FAIL-STOP이고 `g0a_pass=false`다.**

## Current Truth

- Pivot: RoArm cylinder grasp-track G0a. `q5=0` CLOSED, frozen OPEN
  `1.5413 rad`.
- D407은 frozen D362 500-step 계약으로 A64 control(A)과 gripper
  SDF res256(B)을 실제 재측정했다.
- controller final:
  `D407_SDF_PHYSICS_AB_TIPPING_REMEASURE_FAIL_STOP`,
  classification `manual_inspection`, `pass=false`, `g0a_pass=false`.
- A/B physics worker는 각각 exit 0, 500 finite rows, operational PASS.
  두 제한적 physical sub-verdict는
  `D407_MOVING_JAW_CONTACT_AND_OBJECT_MOTION_OBSERVED`.
- FAIL은 physics worker 실패가 아니라 필수 live manual JSON의 300초
  timeout이다. 사후검사로 소급 PASS하지 않는다.
- D408 `[d407_manual_observability_completion_repair]`는 D407의 44개
  산출물을 불변 입력으로 replay했고, 11/11 사람이 확인한 항목과
  원자적 manual JSON publication을 완료했다.
- D408 terminal:
  `D408_D407_MANUAL_OBSERVABILITY_COMPLETION_REPAIR_PASS`,
  `observability_repair_pass=true`.
- D408에서도 Isaac/Kit/PhysX/GPU physics runtime, physics step,
  q5/contact/cylinder 실행은 모두 0이다.
- D408 attempt1은 완료·동결됐고, 승인된 후속 runtime이나 새 과학 case는
  현재 없다.
- 최신 상세 세션:
  `claudedocs/session_20260729_grasp_g0a_d408_actual_read_only_replay_observability_pass.md`.

## Completed Active Case — D408 [d407_manual_observability_completion_repair]

이번 case의 신규 과학 변수: 0.

운영·관측성 변수: 2 —
`d407_clean_view_capture_and_bounded_force_arrow_repair_v1`,
`prearmed_atomic_manual_writer_pid_phase_handshake_v1`.

동결 산출 경로:
`claudedocs/runtime_logs/grasp_track/g0a_d408/attempt1_d407_manual_observability_completion_repair/`

- 실행 범위: controller 1회, software Rerun viewer leg A/B 각 1회,
  manual writer 1회, retry 0.
- 입력: frozen D407 attempt root의 44개 regular file. 시작·중간·종료마다
  전체 manifest를 재해시했고 어떤 D407 파일도 쓰지 않았다.
- 수행 내용: 기존 A/B RRD·trace의 읽기 전용 replay,
  overlay-free spatial capture, 표시 전용 bounded force glyph, 정직한
  수동 판정의 pre-armed atomic publication.
- 수행하지 않은 내용: Isaac/Kit/PhysX/GPU physics runtime, physics
  step, q5/contact/cylinder 재실행, D407 manual JSON 사후 생성.

## D408 Frozen Static Authority

- prereg `0c0f1c03…c8d0d`; static results `bfb0f057…2dab`;
  attestation `fa5a3cf2…50dd`.
- controller `00f4317c…fce`; manual writer `f69d4221…edf7`.
- tuple-file:
  `97c7ca51f8116053fcdc59aa9572669231d4abeb66022ed4e59c9e61af28e1ff`.
- D407 input 44 files/2 dirs, A/B trace 500+500, RRD/RBL footer 4/4,
  stages 13/13, checks 73/73, negative 26/26, allowed dirty 161.

## D408 Actual Runtime Authority

- manual 11/11 true, `pass=true`, SHA `bf917eb4…af18`; terminal SHA
  `48626366…dd37`; phase exact 10/10, final-row SHA `ead440b5…b31`.
- source checkpoint 5/5, screenshot checkpoint 4/4, D407 rehash 44/44 PASS.
- D408 root 32 files/2 dirs; symlink/special/pending 0, nlink 전부 1,
  terminal summary가 마지막 write다.
- Rerun `0.34.1`, CPU Vulkan llvmpipe. A/B force sample 각 2,000,
  bbox row 각 8,000; hardware GPU와 physics counter는 모두 0.
- writer fsync prompt+72.655초, controller 수신 +72.798초,
  deadline margin 522.345초.
- controller exit 0, worker/viewer/writer 잔류 process 0, 새 repo
  `__pycache__` 0.

## Frozen Source — D407 [sdf_physics_ab_d362_remeasure]

- 신규 변수 1: `gripper_link_collision_representation_a64_to_sdf_res256_v1`.
- A는 link5/gripper A64 64+64, B는 link5 A64 64 + gripper SDF mesh 1;
  seed 33201, dt 0.005, OPEN 200 + close 300, 각 500 step, retry 0.
- A/B final XY `60.6190/46.1839mm`, tilt `89.9978/58.1622°`,
  z `-28.0005/+1.7210mm`; peak gripper `43.8583/464.0025N`,
  link5 `23.2279/357.1754N`.
- B step 500은 q5 velocity `3.1403rad/s`, gap `8.542mm`인 비정착 상태다.
  화면 overlay와 300.254초 manual timeout으로 overall FAIL-STOP이며,
  D407 manual JSON은 없고 attempt 44파일은 동결됐다.

## Next Concrete Action — Stop / New Authorization Boundary

D408 승인 범위는 소진됐고 attempt1은 동결했다. 여기서 정지한다.
새 과학 변수, 새 Isaac/PhysX 실행, D407/D408 retry는 승인되지 않았다.
다음 case가 필요하면 먼저 별도의 설계·정적 준비·새 tuple을 만들고,
그 tuple SHA를 인용한 사용자 승인 경계를 다시 세워야 한다.

## Open Risks / Claim Limits

- B step 500은 settled가 아니므로 SDF stability/tipping improvement 불명.
- B derivative link5 scope도 `instanceable=false`; link5 B−A의 순수 인과
  귀속 금지.
- stable grasp, force closure, grasp feasibility, exact manifold/face,
  cap/rim/barrel 순서, per-prim cooked SDF identity, SDF 일반 우월성,
  다른 cylinder 전이는 null.
- B live ContactSensor/SDF binding 경로는 이번이 첫 live 관측이다.
- generic `Failed to clone in Fabric` 1행/leg은 보존하지만 이번 실패의
  단일 원인으로 확정하지 않는다.
- Rerun `RrdReader`는 experimental API이고 notification detector는
  휴리스틱이다. D408은 실제 11개 boolean을 사람이 판정했지만,
  이는 D407 과학 verdict를 바꾸지 않는다.

## Frozen — Do Not Retry or Overwrite

- D400~D408 attempt1, D362 33파일, D334 sidecar 수정·재실행 금지.
- D408 산출물 수정, manual 재작성, replay retry 금지.
- target/IK/path, geometry, material/mass/actuator/physics, `isaaclab`
  env와 numpy/psutil/rerun pin 변경 금지.
- HANDOFF.md, TASKS.md, `/half-clone`, commit/push 금지.

## Must Read First

1. `AGENTS.md`
2. this file
3. `claudedocs/DECISIONS.md` D407, D407-R1, D408, D408-R1
4. `claudedocs/EXPERIMENT_LEDGER.md` tail
5. D408 actual session:
   `claudedocs/session_20260729_grasp_g0a_d408_actual_read_only_replay_observability_pass.md`
6. `claudedocs/session_20260729_grasp_g0a_d408_manual_observability_completion_repair_static_prep.md`
7. `claudedocs/session_20260729_grasp_g0a_d407_actual_runtime_manual_inspection_fail_stop.md`

## Git

- `HEAD == origin/master ==
  a69a96d36219268e4bc5e25065cc234da9d99674`.
- runtime 직전 dirty 131은 prereg exact 기대와 일치했다. terminal 직후
  dirty 151은 allowlist 161 안이었다.
- 필수 사후 세션 문서 추가 뒤 dirty 152이며, prereg allowlist 밖 1개는
  attempt root 밖의 새 actual-session 문서뿐이다.
- commit/push하지 않았다.
