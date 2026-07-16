# D360 — current-pose bounded PhysX jaw-close contact/motion test

Date: 2026-07-16 KST

Case: `g0a_d360`

이번 case의 신규 변수:

1. `bounded_time_evolved_q5_close_execution`
2. `anti_occlusion_contact_force_motion_observation`

## 1. 무엇을 왜 확인하는가

D354는 물리 시간을 전혀 진행하지 않은 채 현재 자세에서 q5를 수치적으로 바꾸어
moving jaw와 원통의 거리를 계산했다. raw/live 모두 같은 매우 좁은 q5 접촉 구간을
찾았지만, 마지막 비접촉점이 원통 윗면과 옆면이 만나는 정확한 경계에 놓였기 때문에
`barrel-first`를 확정하지 못했다. D357은 실제 Isaac 화면을 남겼지만 정지 자세만
표시했고 moving jaw가 원통 뒤에 가려졌다. 따라서 지금까지는 실제 턱을 시간에 따라
닫았을 때 접촉력이 생기고 원통이 움직이는지 시험하지 않았다.

D359는 별도의 해시 불일치 원인을 복구했다. 같은 geometry를 original source point ID
순서와 coordinate lexicographic 순서로 다르게 번호 매긴 것이 원인이었다. 즉 D359는
증거 생성 계보를 고쳤지만 물리 접촉을 증명한 case가 아니다.

D360의 질문은 다음 하나다.

> D354에서 동결한 q0-q4와 원통 자세를 그대로 두고, q5 drive target만 OPEN에서
> CLOSED로 바꾸어 실제 PhysX 시간을 제한적으로 진행하면, moving gripper body의
> 양의 접촉력과 그 뒤 원통의 이동/기울어짐이 관찰되는가?

여기서 `양의 접촉력`은 접촉 센서가 moving gripper body에 귀속한 힘 크기가 기존
D332/D333 기준 `0.1 N` 이상으로 2개 연속 physics step에서 측정되는 것을 뜻한다.
`원통 운동`은 open-baseline 최종 자세에 비해 XY 이동 `0.5 mm` 이상 또는 기울기
`1 deg` 이상이 2개 연속 step에서 나타나는 것을 뜻한다. 이 기준은 새로 만든 성공
기준이 아니라 D332/D333에서 이미 사용한 disturbance 진단 기준을 그대로 상속한다.

### 1.1 왜 이전 D333 결과만으로 충분하지 않은가

D333은 같은 q0-q4와 `q5=0` CLOSED 자세를 원통 옆에 즉시 써 넣은 다음 첫 물리
sample을 얻었다. 그 첫 sample에서 `gripper_link=76.41275491914837 N`, 원통 XY 이동
`1.4549372392421325 mm`, 기울기 `1.821227067988108 deg`가 이미 관찰됐고, 200-step
최대 XY 이동은 `12.598178941303035 mm`였다. 이는 current-pose closed 상태가 실제
moving-gripper body와 강하게 상호작용할 가능성을 보여주는 중요한 선행 증거다.

하지만 접촉 시작이 기록의 첫 sample인 step 0보다 앞에 있었으므로 D333 자체가
`contact_onset_left_censored=true`라고 판정했다. 즉 카메라가 이미 충돌한 뒤부터
녹화를 시작한 것과 같아서, OPEN 상태가 깨끗했는지와 q5가 어느 위치를 지날 때
접촉이 시작됐는지를 알 수 없다. D360은 OPEN baseline부터 연속 기록하고 q5 target을
그 뒤 한 번만 바꾸어 이 시간 순서의 빈칸을 직접 메운다.

## 2. 동결 입력과 물리 조건

- Base before D359/D360 edits: `HEAD == origin/master ==
  d4671d4bdefa4f6e5ef1f2f28b8e318c100b7cb5`.
- Seed: `33201`; environments: `1`; device: `cuda:0`.
- Frozen q0-q4 rad (Float32):
  `[0.03750238195061684, 0.542945146560669, 1.9687392711639404,
  0.18299327790737152, 0.0]`.
- q5 OPEN/CLOSED: `1.5413000583648682 / 0.0 rad`.
- D354 raw/live shared last-clear/first-overlap bracket:
  `1.0269782543182373 / 1.0269775390625 rad`.
- Cylinder center `[0.30000001192092896, 0.0, 0.03288299962878227] m`,
  quaternion `[1,0,0,0]`, radius `0.017 m`, height `0.090 m`, mass `0.72 kg`.
- Material: static/dynamic friction `1.5/1.2`, restitution `0.0`.
- TapTable top `z=-0.012117 m`; redundant global ground collision disabled.
- Robot asset/decomposition: frozen D339/D348/D354 asset, link5 `64` +
  gripper_link `64` actual collider parts; corrected callback-topology audit
  `128/128` parts (`256/256` channels) must pass before motion.
- Physics `dt=0.005 s`, solver settings unchanged.
- Actuators unchanged: stiffness `80`, damping `4`, effort limit `2.5`,
  velocity limit `3.14` for the gripper drive as authored by the inherited env.
- Renderer remains `balanced`; GUI is `headless=false`, `DISPLAY=:1`.

No asset, decomposition, mass, material, contact gate, actuator, solver, target,
IK, path, q0-q4, object initial state, or dependency may be changed in this case.
The one allowed q5 command change is a single OPEN-to-CLOSED position-target step.
Its initial proportional error would be about `80*1.5413 = 123.3` before damping,
well above the frozen `2.5` effort limit, so this is expected to begin as an
effort-saturated close. It must not be described as a gentle or speed-shaped
closure, and the applied q5 effort/saturation state is recorded every step.

## 3. 사전등록된 실행 순서

1. `--stage prepare` checks the Git base/scope, harness and frozen input hashes,
   D359 completion, D334 user-sidecar immutability, `numpy==1.26.0`,
   `psutil==5.9.8`, exact Isaac Python, `DISPLAY=:1`, CUDA device identity,
   at least `8192 MiB` free VRAM and `10 GiB` available RAM. It writes only
   prepare/preregistration artifacts.
2. `--stage run` creates one exclusive invocation marker. Automatic retry and
   reuse/overwrite of the output folder are forbidden. A supervisor records
   worker output, phase heartbeats, GPU/RAM telemetry, and bounded inactivity/
   total watchdogs (`300 s / 900 s`). The inactivity bound is longer than the
   registered `180 s` headless Rerun validation timeout, and RRD generation
   emits progress markers.
3. The worker launches one actual `headless=false` Isaac application on
   `cuda:0`, creates the frozen D333 sole-support cylinder environment, resets
   with seed 33201, and verifies the TapTable-only support domain, exact sensor
   tensor/filter map, D348 corrected audit, and 64+64 live binding. Reset 전후의
   environment counter, SimulationContext clock, Timeline 상태를 별도 기록하며,
   이 reset 내부 전이는 D360 controlled physics step 수에 포함하지 않는다.
4. It writes the exact D354 current pose with q5 OPEN and zero joint/object
   velocity. The file-only Rerun sink is attached before the first decision
   subject is logged.
5. It advances exactly `200` open-baseline steps (`1.0 s`). The baseline hard
   gate inherits the D333 inequalities exactly: first post-step absolute object-z
   correction `<=0.5 mm`; last-50 TapTable world-z force median `>1 N`;
   last-50 absolute cylinder-bottom/table-top gap maximum `<=0.5 mm`; each of
   link4/link5/gripper baseline maximum force `<0.1 N`; maximum XY displacement
   `<0.5 mm`; maximum tilt `<1 deg`; robot-root position/rotation drift each
   `<=1e-6 m/rad`; and the frozen stage/sensor/filter contracts PASS. If any
   item fails, q5 closure is not executed.
6. After a passing baseline, it changes only the q5 position target to exactly
   `0.0 rad`, once. q0-q4 targets remain frozen. It then advances at most `300`
   closure steps (`1.5 s`). If no operational exception occurs, all exactly
   `300` closure steps are executed even after a contact/motion event; there is
   no success early-stop, state mutation, or pose teleport. `At most` only
   covers an operational FAIL_STOP before the horizon.
7. Every controlled physics step records: step/time; q5 target/actual/velocity/
   error/applied effort and effort saturation; maximum q0-q4 drift; cylinder pose, linear/angular velocity, XY
   displacement and tilt; TapTable/link4/link5/gripper forces; filtered contact
   point; and the contact/motion event masks. q0-q4 targets must remain bit-exact
   to the frozen values. Actual q0-q4 at initial, baseline-end, first contact,
   first motion, and final rows plus the full-trace maximum drift are reported.
   No new unregistered q0-q4 drift tolerance is introduced: non-finite state is
   an operational failure, while finite tracking drift is a disclosed diagnostic
   that conditions comparison with D354's static-pose bracket.
   Closure tilt motion is the cylinder-axis change from the actual
   open-baseline final quaternion, not merely its absolute world tilt.
   The Timeline must remain PLAY/not-STOP and may not move backward at every
   controlled step. `SimulationContext` step index and `0.005 s` increment are
   the physics-clock authority; Timeline time is retained as a diagnostic
   because the two APIs do not share a preregistered bit-equality contract.
   The fixed robot root must remain within the inherited D333 bounds
   (`<=1e-6 m` position and `<=1e-6 rad` rotation) across both baseline and
   closure; this scene-integrity gate is distinct from allowed finite q0-q4
   articulation tracking drift. 모든 2-step event mask는 같은 phase 안에서만
   연속으로 인정한다. 따라서 OPEN baseline 마지막 행과 closure 첫 행을 합쳐
   접촉 또는 운동으로 오판하지 않는다. 한 body의 힘 event가 사라졌다 다시
   나타나면 각 event의 첫 두 contact-point sample을 모두 유한성 검사한다.
8. Actual Isaac PNGs are captured without an extra physics step at open
   baseline, first positive robot contact confirmation if present, first threshold
   object-motion confirmation if present, and final state. 여기서 event PNG는
   2개 연속 sample 중 두 번째인 confirmation 상태를 보여 주고, 원 JSON에는
   첫 번째 onset step과 confirmation step을 함께 기록한다. baseline에서 이미
   contact/motion이 나오더라도 그 실패 상태를 같은 방식으로 저장한다. The
   primary camera reuses D351/D354's
   already successful oblique view, eye `[0.49,-0.32,0.28]`, target
   `[0.285,0.0,0.055]`; it replaces D357's failed low side view
   `[0.285,-0.42,0.09]`. The PNG/manual gate must explicitly report whether the
   moving jaw–cylinder interface is truly visible. Because D354 certified the
   two jaw groups but did not certify that exact interface, each requested state
   is captured from both the primary view and a no-step symmetric opposite
   oblique view, eye `[0.49,0.32,0.28]`, target `[0.285,0.0,0.055]`. Moving the
   inspection camera between these captures is not a physics or pose change.
   Manual visibility passes when, for every captured physical state, at least
   one of the two registered views clearly exposes the interface; both views
   are still inspected and recorded independently.
9. The finalized RRD contains the complete 500-step maximum timeline, actual
   robot/jaw and cylinder state, force vectors/contact points, event scalars and
   decision labels. RRD footer verification, exact inventory/timeline checks,
   fixed RBL export, headless screenshot, and original-resolution manual
   inspection are required.
10. Finalization cross-checks controlled step counts, phase order, output
    hashes, sidecar immutability, supervisor exit/watchdog, PNGs, RRD/RBL, and
    separates operational, contact, motion, and visualization verdicts. Supervisor는
    worker 종료 직후 실제 파일 inventory와 SHA-256 manifest, forward-only phase
    sequence를 고정한다. Finalizer는 그 manifest와 phase stream을 다시 계산한다.
    누락된 시각화는 물리 판정을 보존한 observability failure지만, unexpected/core
    file, hash, phase 또는 frozen-input 무결성 실패는 물리 판정을 `null`로 남기는
    별도 post-run integrity failure다.

The maximum controlled horizon is therefore `200 + 300 = 500` steps or
`2.5 s` simulated time. Reset-internal warm-up is reported separately and is
not mislabeled as a controlled D360 sample.

## 4. 판정 규칙

- `D360_MOVING_JAW_CONTACT_AND_OBJECT_MOTION_OBSERVED`:
  baseline hard gate PASS, q5 actual responds in the CLOSED direction, moving
  `gripper_link` force is at least `0.1 N` for 2 consecutive steps, threshold
  object motion begins in the same sampled solver row or later, and the
  interface/recording visual contracts PASS. The first qualifying
  `gripper_link` onset must be strictly earlier than any qualifying link4/link5
  onset. A later link5 onset is reported separately as a fixed-jaw secondary
  contact and is not retroactively called the moving-jaw first contact.
  Reaching the frozen D354 static overlap bracket is diagnostic, not mandatory:
  PhysX contact offset or object motion can create force before that static
  zero-step coordinate and can then block further q5 travel.
  The force stream alone is the body-contact authority. A missing/invalid
  aggregate contact-point coordinate is reported as an observability failure;
  it must not erase an otherwise force-positive physical witness.
- `D360_MOVING_JAW_CONTACT_WITHOUT_THRESHOLD_OBJECT_MOTION`:
  the same moving-body force event is positive, but no inherited-threshold
  object motion occurs within the bounded horizon.
- `D360_OTHER_ROBOT_BODY_CONTACT_OR_PRECONTACT_CONFOUND_FAIL_STOP`:
  any open-baseline robot contact, threshold object motion before the qualifying
  moving-gripper onset, or a qualifying link4/link5 onset earlier than or in the
  same solver row as the moving-gripper onset prevents clean moving-jaw
  attribution. A strictly later link5 onset is not this confound; it is a
  separately reported fixed-jaw secondary-contact event.
- `D360_NO_POSITIVE_CONTACT_WITNESS_UNRESOLVED`:
  the q5 drive reaches the registered geometric bracket but no qualifying
  robot-body sensor event is present. Absence is not promoted to proof of no
  contact because the table positive control does not independently excite
  every robot filter.
- `D360_CONTROL_HORIZON_OR_BASELINE_FAIL_STOP`:
  the exact baseline gates, q5 response, any forbidden target/state write,
  non-finite state, or step bound fails. Finite actual q0-q4 tracking drift is
  reported rather than compared with an invented threshold. Static bracket
  reachability is required only to interpret a no-contact
  trace: if qualifying moving-gripper contact appears first and physically
  blocks q5 before the registered static bracket, that is not a control failure.
- `D360_OBSERVABILITY_FAIL_STOP`:
  the physical trace exists but required PNG/RRD/RBL/manual interface visibility
  does not pass.
- `D360_POSTRUN_INTEGRITY_OR_INVENTORY_FAIL_STOP`:
  worker가 만든 물리 trace가 있더라도 unexpected/core artifact, post-worker
  SHA-256 manifest, phase 순서, frozen input 또는 D334 sidecar 재검사가 실패하면
  해당 trace의 계보를 신뢰할 수 없으므로 최종 physical verdict는 `null`이다.
- Any input/launch/watchdog failure is operational FAIL_STOP and leaves the
  physical question null.

Even the strongest positive D360 result means only that this body-level current
pose creates a real contact that moves the cylinder. It does **not** determine
the exact triangle/face, cap-versus-rim-versus-barrel first-contact feature,
simultaneous two-jaw force closure, stable grasp, hold, lift, or G0a success.
Every D360 verdict keeps `g0a_pass=false`.

## 5. 시각화와 초보자용 결과물

The result package must make four facts visually separable:

1. 시작할 때 moving jaw와 원통 사이가 떨어져 있었는지;
2. q5가 실제로 OPEN에서 CLOSED 방향으로 움직였는지;
3. 어떤 robot body의 contact force가 먼저 기준을 넘었는지;
4. 그 뒤 원통 중심/기울기가 기준 이상 변했는지.

Isaac screenshots show the actual rendered/collider state. Rerun shows the full
time sequence and numeric overlays. Neither display copy is hashed back into
the scientific equality gate; original Float64/JSON/CSV and PhysX sensor arrays
remain the metric authority.

The beginner sheet must display the initial and baseline-end q5 actual values
prominently, so a reader can verify that the clean baseline really remained at
the OPEN condition before the single target change. It also shows the first
robot-body confirmation body/phase/step, failed baseline checks, per-body peak
forces, maximum object XY/tilt motion, and the exact verdict. Thus a negative or
confounded run produces a visual failure explanation rather than only a text code.

## 6. 금지 범위

- exact cap/rim/barrel discriminator or moving-face hash gate: `0`
- target/IK/path change or pose search: `0`
- asset/cook/decomposition/gate/material/mass/actuator/solver/physics change: `0`
- object pose write after controlled motion begins: `0`
- lift, hold, G0b, settle promotion, ten-trial, RL/PPO, VLA, ladder: `0`
- real robot/hardware, B200/SSH, package install: `0`
- D351-D359 overwrite/rerun, D334 sidecar write, commit/push: `0`

## 7. Session-progress rule

The open-baseline and q5-close perturbation can fail and directly changes whether
body-level physical interaction is supported. It is therefore a failure-capable
perturbation evaluation, not validation that cannot change a decision.

## 8. Preregistration-time status

This document section was written before D360 `prepare` and before the sole
Isaac/PhysX invocation. No D360 physics step, q5 target change, screenshot, RRD,
or physical verdict exists yet. Actual execution/results will be appended below
in observable order.

## 9. 등록 실행 핀

Harness:
`sim_scripts/cyl34_top_view_d360_current_pose_bounded_physx_contact_motion.py`

Preregistered harness SHA-256:
`86bd2af855effb3bc31f067fd6cc7a4cb7088c422da3ae828d48f4e31d92fd5a`

Registered commands:

```bash
DISPLAY=:1 OMNI_KIT_ACCEPT_EULA=YES \
  /home/cgxr/miniconda3/envs/isaaclab/bin/python -B \
  sim_scripts/cyl34_top_view_d360_current_pose_bounded_physx_contact_motion.py \
  --stage prepare

DISPLAY=:1 OMNI_KIT_ACCEPT_EULA=YES \
  /home/cgxr/miniconda3/envs/isaaclab/bin/python -B \
  sim_scripts/cyl34_top_view_d360_current_pose_bounded_physx_contact_motion.py \
  --stage run

DISPLAY=:1 OMNI_KIT_ACCEPT_EULA=YES \
  /home/cgxr/miniconda3/envs/isaaclab/bin/python -B \
  sim_scripts/cyl34_top_view_d360_current_pose_bounded_physx_contact_motion.py \
  --stage finalize
```

Only `--stage run` consumes the single Isaac/PhysX invocation. `finalize` is
permitted only after the generated JSON/CSV and every registered image have
been inspected at original resolution and a forward-only manual-inspection
artifact has been written.

## 10. Prepare 결과

`--stage prepare`는 Isaac/PhysX를 시작하기 전에 실행됐고 `16/16` check가 모두
PASS했다. 등록 Git 기준은 `HEAD == origin/master ==
d4671d4bdefa4f6e5ef1f2f28b8e318c100b7cb5`였으며, harness SHA-256은
`86bd2af855effb3bc31f067fd6cc7a4cb7088c422da3ae828d48f4e31d92fd5a`로
사전등록값과 일치했다. `numpy==1.26.0`, `psutil==5.9.8`, Rerun SDK/CLI
`0.34.1`, exact Isaac Python과 `DISPLAY=:1`도 모두 확인했다.

준비 시점 장치는 `NVIDIA GeForce RTX 4090 Laptop GPU`, compute capability
`8.9`, 총 VRAM `16,376 MiB`, free `13,894 MiB`, used `2,050 MiB`, GPU
utilization `0%`였다. 사용 가능 system RAM은 `17,264,902,144 bytes`였다.
이는 실행 전 자원 gate를 통과했다는 뜻이지, 이후 과학 결과를 미리 보장한다는 뜻은
아니다.

Sources:

- `claudedocs/runtime_logs/grasp_track/g0a_d360/d360_prepare_preflight.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d360/d360_preregistration.json`

## 11. 유일한 실제 Isaac/PhysX 실행

등록한 `--stage run`을 정확히 한 번 실행했다. worker는 실제
`headless=false`, `cuda:0`, `DISPLAY=:1` Isaac GUI를 열었고, worker preflight
`14/14`와 다음 runtime prerequisite를 통과했다.

- D348 callback-topology corrected audit `128/128` parts (`256/256` channels)
- live collider binding `64 link5 + 64 gripper_link`
- frozen object/actuator/stage contract
- TapTable sole-support domain과 네 filter의 sensor map

실행 순서는 다음처럼 보존됐다.

1. 초기 OPEN 상태의 두 camera view를 counter `0`에서 저장했다.
2. OPEN baseline을 `200/200` physics step까지 완료했다.
3. baseline 뒤 두 camera view를 counter `200`에서 저장했다.
4. program order상 `baseline["pass"]`가 true인 경우에만 진입하는 분기에서 q5
   target을 `0.0 rad`로 정확히 한 번 바꿨다. q0-q4 target bit는 변하지 않았다.
5. worker exception은 최종 completed controlled-step count `243`, q5 target
   update count `1`을 남겼다. 따라서 baseline 200개 뒤 closure row 43개가
   메모리에서 계산된 상태까지는 도달했다.

중요한 제한이 있다. baseline의 개별 수치와 243개 per-step row는 실행 끝에서 한꺼번에
trace로 쓰도록 구현돼 있었다. GPU 오류가 그 전에 발생했기 때문에, baseline gate가
메모리 안에서 true였다는 분기 사실만 남았고 각 baseline 수치, closure q5 actual,
힘, 이동량, 접촉점은 파일로 보존되지 않았다.

Sources:

- `claudedocs/runtime_logs/grasp_track/g0a_d360/d360_worker_preflight.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d360/d360_runtime_prerequisites.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d360/d360_phase_markers.jsonl`
- `claudedocs/runtime_logs/grasp_track/g0a_d360/d360_worker_exception.json`
- `sim_scripts/cyl34_top_view_d360_current_pose_bounded_physx_contact_motion.py:2459-2534`

## 12. crash 전에 남은 두 임시 trigger의 정확한 의미

Phase marker에는 두 event capture가 남았다. 이것은 **실제 값과 body label까지 포함한
최종 과학 판정이 아니라**, 그 capture 분기에 들어갔다는 program-order 증거다.

1. Contact capture는 global counter `233`에서 primary/opposite 두 장이 저장됐다.
   해당 코드는 최근 두 closure row 모두에서 `{link4, link5, gripper_link}` 중 적어도
   한 body가 `0.1 N` 이상일 때만 실행된다. 따라서 closure phase onset step `31`,
   confirmation step `32`의 **어떤 monitored robot body threshold trigger**만
   provisional하게 지지한다.
2. Motion capture는 global counter `243`에서 두 장이 저장됐다. 해당 코드는 최근 두
   row 모두에서 baseline-end 대비 cylinder XY `0.5 mm` 이상 또는 tilt `1 deg`
   이상일 때만 실행된다. 따라서 closure phase onset step `41`, confirmation step
   `42`의 **object-motion threshold trigger**만 provisional하게 지지한다.

따라서 trace 없이 보존된 가장 강한 시간 순서는
`provisional any-monitored-robot trigger -> provisional object-motion trigger`다.
어떤 body가 기준을 넘었는지, moving gripper였는지, 힘이 몇 N이었는지, 원통이 몇
mm/deg 움직였는지, 접촉점이 어디였는지는 모두 `null`이다. 이를 “moving jaw가
접촉했다” 또는 “원통이 실제로 밀렸다”로 승격하지 않는다.

## 13. 중단 원인: contact-point capacity 초과와 CUDA 범위 오류

원인 증거는 시간 순서가 분명하다.

1. D360이 상속한 D333 ContactSensor는 `track_contact_points=True`와
   `max_contact_data_count_per_prim=16`을 사용한다.
2. 설치된 IsaacLab 문서는 이 값이 filter별 16개가 아니라 **모든 body와 environment에
   걸친 총 한도**라고 설명한다. 실제 구현도 `16 * sensor body count 1 * env count 1
   = 16`으로 PhysX contact view를 만든다. 네 filter는 이 capacity의 배수가 아니다.
3. raw worker log는 먼저 `maxContactDataCount = 16`보다 접촉점이 더 많아서 incomplete
   contact data를 반환한다는 경고를 기록했다.
4. 바로 다음 줄부터 PyTorch ATen `indexSelectLargeIndex`의
   `srcIndex < srcSelectDimSize` assertion이 실패했고, 이어 PhysX tensor CUDA
   device-side assert가 연속 발생했다.
5. worker Python 예외는 다음 `inner.scene.update()`의 sensor update 경로에서 표면화됐다.
   CUDA 오류는 비동기로 보고될 수 있으므로 traceback의 마지막 Python 줄을 최초
   오류 지점이라고 단정하지 않는다. 그러나 `>16 contact points` 경고 직후 bounded
   contact-point buffer를 읽는 인덱스 assert가 발생한 것이 가장 직접적인 근접
   원인 계보다.

즉 이번 중단은 기하 판정이 FAIL한 것이 아니라, contact-rich 상태에서 관측용 접촉점
버퍼가 너무 작아 과학 trace를 끝까지 저장하지 못한 **runtime/observability failure**다.
정확한 실제 접촉점 개수는 기록되지 않았으므로 `16 초과`까지만 말한다.

Sources:

- `sim_scripts/cyl34_top_view_d333_grasp_g0a_sole_support_static_retest.py:121-131`
- `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor_cfg.py:26-35`
- `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py:282-288`
- `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py:380-405`
- `claudedocs/runtime_logs/grasp_track/g0a_d360/d360_worker_stdout_stderr.log:367-410`

## 14. 왜 GPU 메모리, Warp 또는 SM 효율 문제가 아닌가

Supervisor는 102개의 GPU/RAM telemetry sample을 남겼다. 전체 실행에서 GPU used
VRAM 최대값은 `7,703 MiB`, free VRAM 최소값은 `8,241 MiB`, GPU utilization
최대값은 `43%`, 사용 가능 RAM 최소값은 `11,063,463,936 bytes`였다. 오류와 가장
가까운 elapsed `22.685602700919844s` sample도 used/free VRAM
`7,695/8,249 MiB`, utilization `43%`, temperature `51 C`, RAM available
`11,283,103,744 bytes`였다.

OOM/allocation-failed 기록은 없고 최초 범위 오류는 Warp kernel이 아니라 ATen
index-select와 PhysX tensor 경로다. 따라서 SM occupancy를 높이거나 Warp 설정을
조절하는 것은 범위를 벗어난 인덱스를 유효하게 만들지 못하며, 이번 원인에 맞는
수정도 아니다.

Worker는 exception JSON을 쓴 뒤 정상 cleanup을 완료하지 못했고 supervisor가
`worker_exit_code=-9`를 기록했다. Watchdog는 작동하지 않았다. Repository에 보존된
증거에는 SIGKILL 발신자가 없으므로 user, watchdog 또는 OOM killer 중 하나로
추정하지 않는다.

## 15. 실제 Isaac 렌더 8장에 대한 원본 해상도 검사

`1280x720 RGBA` PNG 여덟 장은 모두 decode됐고 원본 해상도로 직접 검사했다.

- initial OPEN: primary/opposite 2장
- 200-step OPEN precommand: primary/opposite 2장
- provisional contact trigger capture: primary/opposite 2장
- provisional motion trigger capture: primary/opposite 2장

두 camera 모두 원통과 end-effector 일부를 실제 Isaac 장면으로 보여준다. 그러나
moving-jaw/cylinder 접촉면은 명확히 드러나지 않고, contact/motion capture는
precommand와 눈으로 구분하기 어려울 정도로 비슷하다. 화면에는 body label, force,
q5 actual 또는 이동량 overlay도 없다. 따라서 이 PNG들은 “그 시점의 장면이
저장됐다”는 실패 증거는 되지만, body identity나 실제 접촉/이동량의 독립 판정
근거가 되지 못한다. D360은 숫자 trace 보존과 interface-visible visualization을 모두
완료하지 못했다.

주요 PNG SHA-256:

- initial primary/opposite:
  `d5779e106fa65c3b3f29bc258cfa017525adcfd66be63099794f93dd20fff762` /
  `3f132aceadc6fee793223131edd082944f291cd08ab7fe2634cf371356399b4a`
- precommand primary/opposite:
  `b4fc2e2ac5249ea8eec87dfb5b3187d99e5939fe3874ad26396fa2648f7c3fd7` /
  `1d5262eeb549994801883d174d6c9973f498116bf8108694d1df923b01e1943b`
- contact primary/opposite:
  `77aeeaeecdaff702acf917a1495751d3bf1670a3217cf879dfc4b23fc9ea1743` /
  `11b10f6d52578eb1280df926ecd556f9236b71d724a56438ea43b998ea6eaa10`
- motion primary/opposite:
  `4fe30b9e92483a569ce60a8da972fbbe5b708035b8e6b71e4cc0eac9ccf80f80` /
  `b180dde71faa890ca06626b280eac0fe140551827c2d14f06f906f272785c6ae`

## 16. 최종 판정과 누락 산출물

최종 operational verdict:

`D360_SINGLE_INVOCATION_CUDA_DEVICE_ASSERT_AFTER_243_CONTROLLED_STEPS_PRE_TRACE_FAIL_STOP`

최종 판정 필드:

- `physical_verdict=null`
- `body_identity=null`
- `moving_gripper_contact_supported=null`
- `object_motion_after_moving_gripper_contact_supported=null`
- `g0a_pass=false`
- `retry_authorized=false`
- `finalize_authorized=false`

완료되지 않은 것은 closure horizon `300/300`, physics trace JSON/CSV, worker summary,
final PNG, RRD/RBL, Rerun validation/screenshot/manual artifact, beginner sheet,
automated/manual/completion summary다. `finalize`는 이 입력들을 무조건 요구하므로 실행할
수 없고, 실행하지 않았다. Supervisor의 “unexpected artifacts” 목록은 expected inventory
자체가 worker summary 부재로 구성되지 못한 기계적 결과이며 외부 파일 유입을 뜻하지
않는다.

Supervisor가 고정한 post-worker 산출물 16개와 그 뒤 기록된 supervisor summary 1개,
즉 `g0a_d360/`의 현재 17개 파일을 실패 시점 그대로 동결한다. 파일 추가, 덮어쓰기,
이름 변경, retry, finalize를 하지 않는다. 핵심 SHA-256은 다음과 같다.

- supervisor: `54bb8a80569048e0299183ae8ed86e81b82a527ec67329c43d4c9e8cb12f026c`
- phase markers: `381c9ae67a72558c9ed99362c30bd93221dbce18a73e54c7256ddfac58d66971`
- worker exception: `1dfb7ffe863f77ece3dc8acafc09b134325b147f1a392907ad5692a9fd650fb1`
- raw worker log: `1bd0aa5a6060da283c8f84b83a0608156624de68aea47eba8c08b5878fc3ecf5`
- runtime prerequisites: `bfd20dc2ab678d9929c0dd8f54323f8f4b17707c6c89d099b1374b814c467dae`

## 17. 다음 승인 경계

가장 좁은 다음 후보는 별도 forward-only D361
`[contact_point_capacity_and_prefix_trace_repair]`다. 정확한 capacity 값은 새 case의
정적 budget 근거와 함께 사전등록해야 한다. 추가로 매 step의 최소 결정 row와 event
body/value를 append-only로 즉시 fsync해, 이후 GPU 오류가 나도 완료 prefix가 남게
해야 한다.

이 후보는 아직 승인되지 않았다. D360 retry나 actual q5/PhysX science rerun도 승인되지
않았다. 새 승인이 있더라도 target/IK/path, asset/decomposition, gate, material, mass,
actuator, solver, physics 설정, initial q0-q5/object pose는 계속 동결하고, exact
cap/rim/barrel face, force closure, grasp, hold/lift, G0b, repeated trial, RL/PPO/VLA,
ladder로 자동 확대하지 않는다.
