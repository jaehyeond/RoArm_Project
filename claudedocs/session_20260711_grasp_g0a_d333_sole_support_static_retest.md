# Session 2026-07-11 (late) - D333: sole-support static retest

Status before runtime: `D333_PRE_REGISTERED_RUNTIME_PENDING`

이번 case의 신규 변수:
`[support_domain_global_ground_collision_disabled]`

This section was written before the D333 physics run. Runtime results must be
appended below without changing these gates.

## Research question

With the correct D34 x H90 cylinder held at the same D332 reset pose and the
same canonical G0a command, does the alignment-only object disturbance remain
after disabling only the redundant global-ground collider and leaving the
TapTable as the sole support?

This run does not decide the swept D330 approach, G0a PASS, collision-mesh
repair, or G0b promotion.

## Frozen comparison contract

- Source: `claudedocs/runtime_logs/grasp_track/g0a_d332/g0a_d332_static_collision_summary.json`.
- Seed: `33201` (same as D332).
- Environments: `1`.
- Physics: `dt=0.005s`, baseline `200` steps, target settle `200` steps.
- Cylinder: center `[0.300, 0.000, 0.032883]m`, radius `0.017m`, height
  `0.090m`, mass placeholder `0.72kg`, friction `1.5/1.2`.
- Canonical command, target, HOME reset, open gripper, reporter threshold,
  object pose, object velocity reset, and diagnostic thresholds stay frozen.
- Seed plus robot USD and URDF SHA-256 values must match the D332 artifact;
  CLI overrides that change them fail before scene creation.
- The only physical scene change is setting
  `/World/ground/terrain/GroundPlane/CollisionPlane.physics:collisionEnabled`
  to `false` before PLAY.

## Pre-baseline hard contract

1. The exact global-ground collider is the only CollisionAPI prim below
   `/World/ground`, is disabled before PLAY, and remains disabled after PLAY.
2. `/World/envs/env_0/TapTable/geometry/mesh` is the sole TapTable collider,
   remains enabled, and its world top differs from `TABLE_Z=-0.012117m` by no
   more than `0.01mm`.
3. The cylinder-owned pre-PLAY ContactSensor resolves one-to-one filters for
   TapTable, link4, link5, and gripper_link; reporter threshold and the known
   sleep-threshold side effect are both `0`.
4. Runtime articulation reports `is_fixed_base=True`.

Any failure writes the pre-baseline contract artifact and stops before the
baseline. No contact/body conclusion is allowed.

## Baseline 200-step hard gate

The target settle runs only if all conditions pass:

- first post-step absolute object z correction `<=0.5mm`;
- last-50 median TapTable filtered world-z force `>1N`;
- last-50 maximum absolute cylinder-bottom to table-top gap `<=0.5mm`;
- maximum link4/link5/gripper_link baseline filtered force `<0.1N`;
- robot root position drift `<=1e-6m` and rotation drift `<=1e-6rad`;
- baseline object XY displacement `<0.5mm` and tilt `<1deg`;
- stage and sensor structural contracts pass.

Failure verdict: `D333_G0A_SCENE_SUPPORT_CONTRACT_FAIL_STOP`. The target
settle is not executed.

## Target 200-step reading contract

- Keep D332 target and exact teleport state unchanged.
- Record full cylinder position, quaternion, linear/angular velocity, TCP,
  joints, root drift, TapTable/link4/link5/gripper forces, contact points, and
  cylinder-bottom/table gap at every step.
- Disturbance onset is the first two-consecutive-step event from either:
  D332 XY `>=0.5mm` or tilt `>=1deg`, or D333 absolute z delta/table gap
  `>=0.5mm`.
- A link5 event supports immediate static interaction only when its
  two-consecutive-step onset is step `0`, object disturbance exists, and the
  link5 onset is no later than disturbance onset plus one physics step.

Pre-registered outcomes:

1. Clean support + no robot event and no object disturbance:
   `D333_G0A_D332_STATIC_EVENT_GROUND_ARTIFACT_SUPPORTED`. This reassigns only
   the D332 static event, not the D330 swept approach or all D330 failures.
2. Clean support + immediate timing-compatible link5 event and disturbance:
   `D333_G0A_CLEAN_STATIC_LINK5_INTERACTION_SUPPORTED`. This supports the
   mirror hypothesis but does not prove live-collider ownership or link5-only
   causality.
3. Clean support + other/late/absent link5 attribution with any event or
   disturbance: `D333_G0A_CLEAN_STATIC_BODY_ATTRIBUTION_MIXED_STOP`.
4. Target fixed-root contract failure:
   `D333_G0A_TARGET_RUNTIME_CONTRACT_FAIL_STOP`.

Every branch stops after D333. `g0a_pass=false`, no ladder promotion, and no
collision repair authorization.

## Artifacts and Visualization DoD

- Output: `claudedocs/runtime_logs/grasp_track/g0a_d333/`.
- Baseline CSV is written before target execution.
- Target CSV and a provisional core JSON are written before visualization.
- PNG count is `1..3`, with target/actual TCP, link5, gripper_link, cylinder,
  and contact/proxy witness frames when target runs.
- Exactly one non-empty RRD is required. Snapshot, marker, or RRD failure
  changes the final verdict to
  `D333_G0A_VISUALIZATION_ARTIFACT_CONTRACT_FAIL_STOP` while preserving the
  physics CSV/provisional JSON.

## Non-goals

No collision rewrite/live-collider ownership scan, target/gate/offset/standoff
tuning, waypoint/approach run, wrist null-space scan, 10-trial gate, gripper
close, grasp, lift, G0b, RL/PPO, randomization, render beyond at most three
diagnostic PNGs, video, VLA, RoArm, B200, cube, or `/half-clone`.

## Runtime result (appended after the physics run)

Verdict: `D333_G0A_CLEAN_STATIC_BODY_ATTRIBUTION_MIXED_STOP`

The sandbox launch could not expose CUDA and stopped before scene creation. The
same frozen command was then run once in the approved host GPU environment. It
produced exactly 200 baseline rows and, after the hard gate passed, 200 target
rows. No parameter or seed changed between the pre-registration and host run.

### 1. Sole-support and witness contract

| Contract | Result |
|---|---:|
| exact ground collider pre-PLAY | `enabled -> disabled` |
| exact ground collider post-PLAY | `disabled` |
| TapTable collider | `enabled` |
| TapTable top error | `0.000000297mm` |
| robot fixed base / max root drift | `true / 0m, 0rad` |
| ContactSensor bodies / filters | `1 / 4`, one-to-one |
| reporter / sleep threshold | `0N / 0` |

All structural checks passed. The global-ground confound was removed rather
than merely filtered out.

### 2. Baseline hard gate

| Metric | D333 result | Gate |
|---|---:|---:|
| first-step object z correction | `0.000003354mm` | `<=0.5mm` |
| last-50 TapTable Fz median | `7.063635349N` | `>1N` |
| tail bottom/table max abs gap | `0.000134554mm` | `<=0.5mm` |
| max baseline XY / tilt | `0.003773945mm / 0.003364521deg` | `<0.5mm / <1deg` |
| max link4/link5/gripper force | `0 / 0 / 0N` | `<0.1N` |

The baseline passed every pre-registered gate. In particular, the D332
`+12.256849mm` robot-free ground pop became approximately zero, while the
TapTable filtered force independently matched the `0.72kg` placeholder weight.

### 3. Clean target result

| Metric | Result |
|---|---:|
| first robot / gripper onset | `step 0 / step 0` |
| first link4 / link5 onset | `-1 / -1` |
| gripper peak force | `76.412754919N @ step 0` |
| gripper force `>=0.1N` | `180/200` rows (`0-1`, `22-199`) |
| XY/tilt disturbance onset | `step 0` |
| center-z disturbance onset | `step 1` |
| support-gap disturbance onset | `-1` |
| max/final XY displacement | `12.598178941 / 9.298849201mm` |
| max/final tilt | `8.074518 / 3.881523deg` |
| max abs/final z delta | `1.936368 / 1.047179mm` |
| peak linear/angular speed | `0.371259m/s / 7.496454rad/s` |
| final actual / commanded TCP error | `6.673174 / 0.817812mm` |

The target object remained table-supported: its maximum absolute bottom/table
gap was only `0.269894mm`, final gap `-0.000389mm`, and TapTable force persisted.
Therefore the step-1 center-z event is not a support-loss event.

### 4. Attribution audit from recorded rows

- Across all target rows, cylinder net force minus the sum of the four filtered
  force vectors had maximum component residual `6.7417e-6N`.
- At step 0 the gripper force was
  `[71.229866,-27.662525,+0.000907]N`; its XY cosine with object displacement
  was `0.999981` and with object velocity `0.951235`.
- The step-0 gripper aggregate contact point was
  `[0.291517,0.003320,0.066635]m`, `11.248mm` below the reset cylinder top.
- This makes a clean `gripper_link` rigid-body attribution strong. It does not
  identify the exact collision shape or prove that the 4mm proxy is the shape
  producing the force. Link5 `0N` also remains non-conclusive without an
  independent link5 positive control.

The pre-registered branch is therefore exactly branch 3: clean support plus
immediate gripper-attributed disturbance without an immediate link5 event.
Neither `GROUND_ARTIFACT_SUPPORTED` nor `LINK5_INTERACTION_SUPPORTED` applies.

### 5. D332/D330 interpretation change

- D332 ground depenetration was a real vertical confound, but not the sole cause
  of its gripper event or object motion. Target step-0 z correction fell from
  `12.707490mm` to `0.480719mm`, while gripper peak force changed from
  `66.866266N` to `76.412755N` and final XY stayed `10.282285 -> 9.298849mm`.
- The prior `pop-into-gripper only` interpretation is refuted.
- The D331 link5 mirror gap-fill hypothesis is downgraded from leading runtime
  cause to an unresolved offline-mirror versus live-body-attribution mismatch.
- D330's swept approach and bimodal regimes remain scene-confounded and were not
  rerun. D333 does not reassign all D330 failures, but it strengthens the
  qualitative conclusion that correct-cylinder substitution alone is not
  sufficient.

### 6. Artifact audit and semantic corrections

- Summary and both CSVs matched under independent recomputation.
- `d333_postrun_csv_reanalysis.json` records source SHA-256 values, force
  closure, onset reconstruction, D332 comparison, and byte-identical D332/D333
  canonical joint CSVs.
- The runtime summary's `max_force_step_by_link=0` for zero-force link4/link5 is
  an `argmax` artifact; the correct event/max step is `-1`. The probe code was
  corrected for future runs without rerunning physics.
- Three PNGs were visually inspected and contain target/actual TCP, link5,
  gripper_link, cylinder, and witness frames. Marker status passed and the one
  RRD is non-empty (`3,021,617` bytes, `200` steps).
- The host console emitted a non-fatal Fabric clone warning, but the one-env
  scene initialized and every stage/sensor/root/row-count contract passed. Raw
  console stderr was not retained as an artifact; no decision relies on it.

### 7. Stop and next decision experiment

G0a remains incomplete. G0b, close/lift, PPO/RL, target tuning, and collision
rewrite remain blocked.

The next single task is D334: at the frozen D333 pre-step pose and recorded
post-step-0 pose, audit link5 and gripper_link collision prims, nearest rigid-
body ownership, source/cook representation, and signed distance to the
cylinder. This must reconcile the D332 link5 mirror overlap with D333's clean
gripper attribution before choosing collision-representation repair versus a
target-family repair. AABB-only reasoning is forbidden.

## Runtime evidence

- `sim_scripts/cyl34_top_view_d333_grasp_g0a_sole_support_static_retest.py`
- `claudedocs/runtime_logs/grasp_track/g0a_d333/g0a_d333_sole_support_static_summary.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d333/d333_contact_baseline_trace.csv`
- `claudedocs/runtime_logs/grasp_track/g0a_d333/d333_teleport_settle_trace.csv`
- `claudedocs/runtime_logs/grasp_track/g0a_d333/d333_postrun_csv_reanalysis.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d333/d333_baseline_final.png`
- `claudedocs/runtime_logs/grasp_track/g0a_d333/d333_target_first_event.png`
- `claudedocs/runtime_logs/grasp_track/g0a_d333/d333_target_final.png`
- `claudedocs/runtime_logs/grasp_track/g0a_d333/d333_sole_support_static_trace_v2.rrd`
