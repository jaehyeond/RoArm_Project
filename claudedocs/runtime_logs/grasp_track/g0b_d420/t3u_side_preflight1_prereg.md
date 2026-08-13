# p16 / t3u_side_preflight1 preregistration — side-grasp instrumentation preflight

Status: **PREREGISTERED / NOT RUN**  
Case: `g0b_d420`  
Canonical prefix: `t3u_side_preflight1`  
Date: 2026-08-11 KST

Supersession (2026-08-12 KST, before any p16 Isaac/PhysX step): the original draft
named p15 candidate index 0 as the representative.  Loading the frozen side_sdg2 bytes
through the parsed-URDF p16 planner falsified that choice: all five index-0 offsets had
final contact-frame residual `8.073208..8.196929 deg`, so zero passed the already-fixed
`<=2 deg` gate.  No gate was relaxed and no physics outcome existed.  The frozen
preflight subject is now candidate index `5`, ID `side_sdg_005_raw_025092`: it is the
**first candidate in p15 canonical order** for which all five offsets pass the unchanged
IK, URDF-limit, and contact-frame gates (`1.066656..1.191453 deg`, maximum waypoint
position error `0.242412 mm`).  Index 7 also passes those pre-physics gates but is not
selected, preventing best-result selection.  The full canonical plan still reports all
eight candidates and every rejected row.

## 1. Scope and authority

This is a non-scientific, failure-capable instrumentation preflight for the user-approved
sim-only D419 exception.  It must execute the complete fixed-base RoArm trajectory and
may fail, but no grasp conclusion may be promoted from this tag.  Its only promotion
question is whether the exact p16 contact, lifecycle, Rerun, camera-frame, PNG, and MP4
contracts are observable without silent-zero or frame-selection failures.

이번 case의 신규 변수: `[side-midpoint grasp point, SDG candidate pose]`.
The p16 lateral pinch-centre offsets are a frozen adapter calibration derived before
physics from the actual attempt3 64+64 convex-hull inner surfaces; they are not tuned
after an outcome.  Object position, dimensions, mass, material placeholders, controller,
and trajectory phases are controls.

## 2. Frozen physical subject

- Upright analytic `UsdGeom.Cylinder`, radius `0.0145 m`, height `0.050 m`, mass
  `0.02483 kg`; object centre exactly
  `[0.4235072423787768, 0.17237803311822986, 0.025] m` (`seed0_S4`).
- Object yaw `0`; support is the existing infinite `z=0` plane.  This is not an actual
  finite desk model.
- Cylinder-authored placeholder material is fixed at static friction `0.40`, dynamic
  friction `0.30`, restitution `0.0`.  Jaw/support materials and effective pair friction
  are not re-authored or claimed as real.
- Before the first task step, every composed clone is read back from the actual stage:
  rigid body and MassAPI at `/Sponge`, `UsdGeom.Cylinder` and CollisionAPI at
  `/Sponge/geometry/mesh`, and the `physics`-purpose material resolved with
  `ComputeBoundMaterial()` to `/Sponge/geometry/material`.  Radius, height, mass,
  gravity, material coefficients, restitution, and both `average` combine modes are
  gated with serialized tight tolerances.  Stage meters-per-unit and kilograms-per-unit
  must both be `1` (within `1e-12`) so the readback has SI meaning.  The result reports
  all authored stage physics materials but does not infer an effective pair friction.
- Frozen attempt3 collision asset: exactly 64 enabled convex-hull parts on `link5` and
  64 on `gripper_link`; both legacy colliders remain present and disabled.  No visual
  mesh proxy may enter physics or pinch calibration.
- Kinematic object attach, any object-to-robot fixed joint, object-follow pose write, and
  object teleport after reset are forbidden.  The only object pose write is the initial
  per-environment reset.  This does not prohibit the robot's preregistered root fixed joint.
- "Fixed-base" is proved from the composed/runtime subject, not inferred from the URDF
  filename.  Isaac Lab's PhysX metatype must report `is_fixed_base=true`; each clone's
  enabled `/Robot/root_joint` must be a `PhysicsFixedJoint` from stage world (empty
  `body0`) to `/Robot/world` (`body1`).  The importer merges the URDF's conceptual
  `base_link` into this composed `world` rigid body.  Its pose/orientation must not drift
  by more than `1e-7 m`/`1e-7` quaternion-component (sign invariant), and linear/angular
  velocity must stay within `1e-7` at every one of 2,340 task steps.
- The primary joint envelope is parsed at runtime from
  `local_assets/roarm_m3/urdf/roarm_m3.urdf`; no literal CLAUDE table is accepted.
  Expected degrees, recorded but not trusted without parsing: base
  `[-180.0004209183,+180.0004209183]`, shoulder
  `[-90.0002104591,+90.0002104591]`, elbow
  `[-57.2957795131,+169.0225495636]`, wrist pitch
  `[-110.0078966651,+110.0078966651]`, wrist roll
  `[-180.0004209183,+180.0004209183]`.  The v6 distribution clip is a separate
  compatibility diagnostic and never replaces the primary limit gate.
- After the cloned articulation exists and before task stepping, p16 reads every clone's
  actual `soft_joint_pos_limits`, the environment's cached lower/upper tensors, and each
  composed `/Robot/joints/<name>` `UsdPhysics.RevoluteJoint` lower/upper attribute.  All
  six ordered joints must equal the parsed URDF authority within `5e-7 rad` for runtime
  tensors and `2e-5 deg` for authored USD.  At every step, the post-clamp applied target
  is recorded separately from the planned target, must equal it within `1e-7 rad`, and
  both applied targets and actual joint positions must remain inside parsed limits
  (actual-position tolerance `1e-5 rad`).  A mismatch is measurement invalidity/abort,
  never a silently changed experiment subject.

## 3. SDG handoff and frame contract

The only candidate input is the forward-only p15 artifact
`t3s_side_sdg2_candidates.json`, schema
`g0b.t3s.side_sdg_candidates.v1`.  p16 records its full SHA-256 and rejects any object
contract mismatch.  The artifact must contain exactly eight filtered candidates in its
frozen canonical rank order.  The raw NVIDIA flying-gripper root transform is provenance
only and explicitly has no RoArm prim/TCP transform.  The decision input is each filtered
candidate's recovered antipodal midpoint and
orthonormal candidate frame:

- link5 `+Z`: approach direction, horizontal radial outward from base to object.
- link5 `+X`: horizontal tangential jaw-closure direction.
- link5 `+Y`: world up.  These axes are right-handed (`+X cross +Y = +Z`).

The NVIDIA candidate axes are an unsigned antipodal proposal.  p16 applies and records
the fixed candidate-to-link5 sign/convention map; it may not infer the map from a physics
outcome.  The raw 40 mm SDG standoff is provenance, not a safe fixed-base path.

These axes are not silently called the RoArm TCP frame.  p16 records the fixed
candidate-to-link5 axis map and the pinned link5-to-TCP offset.  The actual attempt3
geometry determines a lateral pinch-centre offset before any PhysX outcome is visible.

p15's failed `side_sdg1` run is retired (`SIDE_FILTER_TOO_FEW`, six rows, no canonical
artifact).  The forward-only `side_sdg2` changed only sampler surface samples from 1,024
to 4,096; it observed 51,760 raw transforms, 20 filter passes, and selected eight.

p15's handoff declares that p16 consumption is forbidden until its Rerun validation and
manual PNG inspection pass.  Therefore p16 also verifies the complete p15 config-bound
artifact set, `rerun_validation.json pass=true`, and a forward-only
`t3s_side_sdg2_manual_visual_inspection.json`.  That manual record must bind the exact
candidate, inspection-PNG, and validation SHA-256 values; contain at least one written
observation; and mark all four visible frame checks true: midpoint at side midheight,
`+X` horizontal tangent, `+Y` world up, and `+Z` radial outward.  Missing or mismatched
p15 observability evidence aborts before AppLauncher.

## 4. Pre-physics pinch calibration

At every `q5` in `[0, 88.3099849635] deg` with a frozen `0.1 deg` grid, p16 transforms
all hull-surface samples from the 64 moving parts through the authored revolute-joint
frames.  The upright cylinder axis is link5 `+Y`; its circular cross-section is link5
`XZ`.  In the finite-height slab, the fixed inward surface at TCP depth is measured,
and the nominal cylinder centre is `fixed_inner_x + radius`.  The moving jaw's first
contact is the q5 row whose minimum XZ radial distance to that centre is closest to the
exact radius.  A valid calibration requires both bodies, fixed and moving radial
residuals no larger than the `0.5 mm` hull sampling pitch, and an interior q5.

Independent read-only drift witnesses are fixed inner `x=-10.025849566 mm`, nominal
pinch centre `+4.474150434 mm`, and moving first contact approximately
`q5=22.840 +/-0.476 deg` (p16's 0.1-degree scan is expected near 22.8).  Five offsets are
frozen before physics: nominal plus `[0,+0.25,+0.50,+0.75,+1.00] mm`; negative offsets
that pre-penetrate the fixed jaw are forbidden.  All five are retained and reported.
The sole canonical close command is `q5=22.0 deg`.  `21.5 deg` is reserve-only after an
observed 22.0-degree failure and requires a new forward-only tag/prereg; it is not tuned
inside preflight or canonical.

## 5. Full trajectory and instrumentation

One runner step is one `1/200 s` PhysX step (`decimation=1`).  All profiles use the full
schedule: settle `120`; HOME to elevated pregrasp `400`; elevated pregrasp to near-side
staging `400`; staging to final horizontal midpoint `400`; close `400`; hold `120`;
world-vertical lift `500` steps.  The elevated pregrasp is final TCP minus 40 mm radial
plus 40 mm world-up; near staging is minus 5 mm radial plus 10 mm world-up.  HOME is
`[0,0,90,0,0,88.3099849635] deg`; lift is 25 mm.  These waypoints are fixed safety
controls, not research variables.

All six joints use direct position targets through the inherited implicit actuators,
with stiffness `100.0` and damping `5.0`; no policy or learned action is involved.
Contact capacity is fixed at `256` per prim, Fabric cloning is disabled, self-collision
is enabled, and the pinned p14 GPU-PhysX capacity/readback contract remains in force.
There is no object-pose, force, friction, or controller randomization in p16.

Before PhysX, p16 transforms the enabled collision geometry of every moving link and
both 64-part jaws at every one of the 2,340 scheduled joint-command samples.  Every
moving body must remain at least `1.0 mm` above `z=0` throughout all seven phases:
object contact after descend is intentional, but support contact never is.  Runtime
sensors independently gate actual link/support contact.  Before trusting this planned
clearance, p16 must also match the p10/jaw FK body frames to every composed USD moving-body
  frame at the authored initial `[0,0,90,0,0,0] deg` state within `1e-6 m` translation and
  `1e-6` rotation-matrix max-absolute error.  At the delivered final pinch, the IK/FK
  signed adverse pitch has a separate hard `abs(pitch) <=1 degree` gate.  Transformed-hull
  clearance is required in parallel and is never a waiver for a larger pitch residual;
  closure/contact-frame error must also remain `<=2 degrees`.

The ordered trial set is itself a hard gate twice.  Before environment creation it must
be exactly candidate 5 crossed with offsets 0..4 (five planned and five IK/frame-feasible
rows); after transformed-hull clearance it must still be that exact ordered set.  A
different nonempty set is contract drift, not a smaller substitute experiment.

Contact reporters are armed and threshold-read back at `0.0` for the cylinder and every
robot rigid body `world, link1, link2, link3, link4, link5, gripper_link` in every clone
(exactly eight reporter subjects per clone).
The object sensor separately attributes support and all six robot bodies.  Each moving
body has an exact support-plane sensor.  All 15 non-adjacent self-contact pairs across the
composed fixed body `world` plus the six moving bodies are separately observed, including
`world` versus `link2, link3, link4, link5, gripper_link`.  The six adjacent pairs are
excluded and the exact included/excluded inventories are serialized.  Self-collisions are
enabled for p16 and the effective stage value is read back.  Fixed `world`/support contact
is normal and excluded from the task gate; moving-body/`world` contact is not excluded.
For each one-filter support/self sensor, p16 records the actual PhysX filter pattern,
resolves its regex against the composed stage, requires exactly the preregistered ground
collider or every `/World/envs/env_i/Robot/<body_b>` target, and binds that identity to
the `(N,1,1,3)` force-matrix slot.  Shape/count alone is not accepted.

Positive controls are: settled object support force near `m*g`, exact zero-threshold
reporter audit on every clone/body, contact-buffer integrity, and a preregistered
`seed0_S4/theta69` moving-jaw/support witness in the first padding slot.  The witness is
derived from `t3y_workspace1/trial_005948` (`JAW_SUPPORT_CONTACT_FAIL`) with frozen
approach `[22.1475517244,54.0097107355,84.6832482607,-26.5865184743,90,88.3099849635]`,
descend `[22.1475517244,64.0184572171,64.1733068506,-17.7902657428,90,88.3099849635]`,
close q5 `66.4`, and lift
`[22.1475517244,59.7700794312,63.8778329842,-12.9423014136,90,66.4] deg`.  It follows
the p16 seven-phase timing and must produce moving-`gripper_link`/support force
strictly `>0.02 N`.  This is an instrumentation positive control, is traced separately,
and cannot enter active grasp counts.  Its q5 is not a task control and does not relax
the sole task close command `22.0 deg`.

## 6. Task gates (diagnostic only in this tag)

A candidate-offset row is a diagnostic success only if all are true:

1. all waypoint IK position errors `<=3 mm`; final closure/contact-frame error `<=2 deg`;
   delivered final signed adverse pitch `abs(pitch)<=1 deg`; every axis residual is
   reported; every planned and post-clamp applied command remains inside the parsed URDF
   limits and equal within `1e-7 rad`; the all-phase static support-clearance gate passes;
2. arm joints `q0..q4` arrive within `3 deg`; q5 target error is not an arrival gate
   because physical contact is expected to stall closure;
3. no premature pre-close jaw/object contact above `0.02 N`; independently, no
   moving-link/support force above `0.02 N`, no non-jaw/object force above `0.02 N`, and
   no gated non-adjacent self-contact (including moving body versus fixed `world`) above
   `0.02 N`;
4. both jaws load the cylinder on the same physics step during close and again during
   lift, with `max_t min(F_fixed(t),F_moving(t)) >0.01 N` in each phase;
5. corrected object rise is strictly `>6 mm` and final cylinder tilt is below
   `atan(D/H)=30.1137 deg`;
6. contact buffers are unsaturated, the settle force positive control passes, and
   kinematic attach count remains zero.

Every authoritative step also has a numerical-integrity gate.  Joint/object/body poses
and velocities, force tensors, controls, and derived metrics must be finite; object and
moving-body/fixed-body quaternion norms must stay within `1e-3` of one.  Runtime effective
joint limits, applied-target equality, actual joint-limit containment, and fixed-base
pose/velocity stability are serialized and independently recomputed from NPZ.  Raw contact counts must be
finite, integer, nonnegative, and within the fixed 256 capacity.  Contact positions may
be NaN only when the corresponding raw count is zero, are finite when it is positive,
and may never be infinite.  A failed numerical check makes `measurement_valid=false`
before any strict force/lift comparison, preventing `+Inf` from becoming a PASS.

This preflight's scientific verdict is always `null`.  Its only promotion verdict is
`INSTRUMENTATION_PREFLIGHT_PASS` or `INSTRUMENTATION_PREFLIGHT_FAIL`.

## 7. Observability and terminal promotion

Launch exactly eight environments with exactly five active task rows: p15 canonical
candidate index `5` (`side_sdg_005_raw_025092`) crossed with the five frozen pinch
offsets.  Padding slot 5 is the
instrumentation witness above; slots 6--7 are inactive padding.  None of those three slots
enters task counts.  Record every physics step for every active row and the named witness
channels in the canonical NPZ.  Render
exactly candidate index `5` at the nominal offset, selected by candidate ID and explicit
environment slot before physics; never infer the camera/render environment from viewport
order.  Required artifacts include JSON, NPZ,
RRD, RBL, headless PNG, exact RGB frame manifest, and MP4.  Frame count, physics-step
mapping, first/last frame hashes, fps, resolution, and ffmpeg command are recorded.
BasicWriter/asynchronous frame overproduction is forbidden.

The RRD logs all 2,340 steps of that representative, all six moving-body transforms,
the analytic cylinder, support plane, composed enabled-collider vertex clouds, object
contact points/force arrows for support plus all six bodies, all-link support forces,
all 15 non-adjacent self-force diagnostics (including fixed `world` pairs),
target-vs-actual frames, and scalar force/pose/q5
channels.  The RBL is fixed and exported; `rrd verify`, exact entity/timeline/component
validation, a headless `inspection.png`, and a separate `decision_snapshot.png` are
mandatory.  NPZ/JSON remain numerical authority.

MP4 is a second invocation of the exact frozen p16 source:
`--render_trace side_preflight1`.  It loads the bound NPZ after physics has exited, creates
no `SimulationContext` or `PhysicsScene`, performs zero explicit physics-step calls,
stops and commits the timeline immediately after AppLauncher and before new-stage setup,
applies the representative's recorded body/object transforms directly to a fresh USD
stage, and reads RGB synchronously from a Replicator annotator.  A PhysX step callback,
SimulationManager step/time, and timeline time/play state are sampled before and after
every `app.update()` and Replicator step; a stopped `app.update()` is an explicit
counterexample gate.  Every written frame verifies pre/post USD transforms and frozen
joint-source values against NPZ.  It writes exactly
234 frames at 1 frame per 10 source physics steps (`20 fps`, `1280x720`, source steps
10..2340).  Warm-up renders are not written.  Each PNG and its source step are listed in
the manifest; ffmpeg full-decode must return exactly 234 frames.  Before PASS, p16
cross-binds render-start hashes to the physics result's frozen final-dependency snapshot,
then re-hashes at render end its own source, this supervisor, both preregs, p15 handoff,
environment sources, URDF, and all five USD layers and requires equality with render
start.  The video is posthoc inspection evidence and cannot change the physics verdict.

Canonical p16 remains blocked until: runtime instrumentation PASS; external terminal
attestation confirms exit 0, no timeout/signal/failure/process/PGID/GPU residue, and
result/sentinel/phase hash binding; RRD validation PASS; PNG and MP4 are actually opened
and inspected; the representative row agrees across JSON/NPZ/RRD/video.

The actual review is recorded forward-only as
`t3u_side_preflight1_manual_visual_inspection.json`, artifact
`T3U_SIDE_PREFLIGHT_MANUAL_VISUAL_INSPECTION_V1`.  It binds the exact result, terminal
attestation, Rerun validation, inspection PNG, decision-snapshot PNG, RGB manifest, and
MP4 SHA-256 values; includes at least one written observation; and marks four checks true:
target/actual frames visible, RRD axes consistent, full MP4 trajectory visible, and
jaw/object/support relationships visually checked.  p16 canonical validates this exact
record rather than accepting an unbound `pass=true` file.
The manual record has exactly these top-level keys and no extras: `artifact`, `profile`,
the seven named SHA bindings above, `visual_checks`, nonempty `observations`, and `pass`.

## 8. Frozen launch and terminal lifecycle

The physics and isolated render invocations must run sequentially under one detached,
no-retry process-group supervisor,
`sim_scripts/p16_g0b_t3u_detached_physics_render_supervisor.py`.  It creates only new
`t3u_side_preflight1_*` names and never removes or retries a tag.  Its forward-only
contract JSON is exactly:

```json
{"artifact":"T3U_DETACHED_PHYSICS_THEN_RENDER_SUPERVISOR_V2","automatic_retry_count":0,"bounded_waitpid_only":true,"child_parent_death_signal":"SIGTERM","child_preexec_signal_state":"SIGTERM_SIGINT_SIGHUP_SIG_DFL__empty_mask__expected_parent_pid_recheck","detached":true,"kill_after_seconds":20,"physics_then_render_only_on_zero":true,"physics_timeout_seconds":7200,"raw_waitpid_status_authority":true,"render_timeout_seconds":7200,"supervisor_signal_cleanup":"SIGTERM_SIGINT__active_child_pgid_TERM_20s_then_KILL_20s","term_signal":"TERM"}
```

The supervisor must itself be a no-TTY `setsid` session/process-group leader.  Each child
first resets inherited SIGTERM/SIGINT/SIGHUP dispositions to `SIG_DFL`, clears its signal
mask, then sets Linux `PR_SET_PDEATHSIG=SIGTERM` and rechecks the exact pre-fork parent PID;
an already-dead/mismatched parent forces immediate child termination.  The supervisor
installs SIGTERM/SIGINT handlers and
keeps an active-child PID/PGID registry.  Its own signal, exception, or timeout path sends
TERM to the child group, waits at most 20 seconds, then sends KILL if needed and performs
only bounded `WNOHANG` reap attempts.  Cleanup actions and any unreaped residue are fatal
evidence rather than a silent orphan.  It records
supervisor/child PID, SID, PGID and TTY state, raw `waitpid` status, ordinary exit or
signal/core status, timeout and TERM/KILL actions, attempt count exactly one, reaped group
membership, combined stdout, launcher diagnostic, GPU inventories, hashes, and combined
exit.  It must run physics first with the exact p15 artifact SHA supplied as
`--candidates_sha256 67eb07d68268be25c894c47fb6bee79347e2b201dea643c4823a0873bfcde384`,
and only after raw exit zero run:

- `t3u_side_preflight1_supervisor_contract.json`
- `t3u_side_preflight1_supervisor_pid.txt`
- `t3u_side_preflight1_physics_python_pid.txt`
- `t3u_side_preflight1_render_python_pid.txt`
- `t3u_side_preflight1_pgid.txt`
- `t3u_side_preflight1_stdout.log`
- `t3u_side_preflight1_nvidia_smi_before.csv`
- `t3u_side_preflight1_nvidia_smi_supervisor_end.csv`
- `t3u_side_preflight1_supervisor_outcome.json`
- `t3u_side_preflight1_exit_status.txt`, written only after both child invocations and
  the process group are reaped (it must not exist when the physics G0 guard runs).

All names are forward-only under `g0b_d420`; the terminal attestor creates only
`t3u_side_preflight1_nvidia_smi_after.csv` and
`t3u_side_preflight1_terminal_attestation.json`.

```bash
(
  set -o noclobber
  p16_case_dir=/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/g0b_d420
  p16_profile=side_preflight1
  p16_prefix="t3u_${p16_profile}"
  p16_suffixes=(
    results.json plan.json trace.npz timeline.rrd timeline.rbl
    rerun_validation.json decision_snapshot.png inspection.png
    rgb_frames_manifest.json side_grasp.mp4 script.py.txt argv.txt phase.jsonl
    preclose_sentinel.json terminal_attestation.json manual_visual_inspection.json
    failure.json exit_status.txt stdout.log supervisor_launcher.log
    supervisor_pid.txt physics_python_pid.txt render_python_pid.txt pgid.txt
    supervisor_contract.json supervisor_outcome.json nvidia_smi_before.csv
    nvidia_smi_supervisor_end.csv nvidia_smi_after.csv supervisor_failure.json
  )
  for p16_suffix in "${p16_suffixes[@]}"; do
    p16_target="${p16_case_dir}/${p16_prefix}_${p16_suffix}"
    if [[ -e "${p16_target}" ]]; then
      echo "G0 existing target: ${p16_target}" >&2
      exit 3
    fi
  done
  if [[ -e "${p16_case_dir}/${p16_prefix}_rgb_frames" ]]; then
    echo "G0 existing frame directory" >&2
    exit 3
  fi
  nohup setsid /home/cgxr/miniconda3/envs/isaaclab/bin/python \
    /home/cgxr/Documents/Robotics/RoArm_Project/sim_scripts/p16_g0b_t3u_detached_physics_render_supervisor.py \
    --profile "${p16_profile}" \
    --candidates_sha256 67eb07d68268be25c894c47fb6bee79347e2b201dea643c4823a0873bfcde384 \
    </dev/null >"${p16_case_dir}/${p16_prefix}_supervisor_launcher.log" 2>&1 &
  echo "detached supervisor pid=$! (one attempt; no retry)"
)
```

The shell must use `noclobber` and verify the complete supervisor G0 target list before
that redirection creates the one permitted launcher log.  The supervisor repeats G0 for
all other lifecycle, scientific, frame-directory, terminal, manual, and failure names.

After the whole process group is reaped, run:

```bash
/home/cgxr/miniconda3/envs/isaaclab/bin/python \
  /home/cgxr/Documents/Robotics/RoArm_Project/sim_scripts/p16_g0b_t3u_cyld29h50_side_midpoint_parallel_physics.py \
  --terminal_attest side_preflight1
```

The attestor never imports Isaac.  It requires combined exit `0`, absent failure marker,
reaped supervisor/physics/render PIDs, empty recorded PGID, no new GPU PID relative to the
pre-run inventory, independently decoded raw child status, exact outcome and file
bindings, exact phase/result/sentinel/render hashes, actual all-clone object/material and
filter/reporter readbacks, numeric-integrity recomputation from NPZ, RRD PASS, exact
234-frame full-decode PASS, zero observed render physics clocks/callbacks/scenes, and
physics-final/render-start/render-end three-way dependency equality.  It creates V2
terminal attestation but still cannot
replace actual human opening/inspection of both PNGs and MP4.  Canonical promotion
recomputes these semantics and exact check-key sets; it never trusts a lone `pass` field.

## 9. Frozen source and evidence pins

- detached supervisor source:
  `9a1c51dad74831272b7e2dddbe152077a0b787619019a8b3a9242706ea7ba933`
  (`sim_scripts/p16_g0b_t3u_detached_physics_render_supervisor.py`); its only allowed
  invocation is the exact `--profile side_preflight1 --candidates_sha256
  67eb07d68268be25c894c47fb6bee79347e2b201dea643c4823a0873bfcde384`
  argv shown in section 8 under `nohup setsid`.
- p15 `side_sdg2` producer source: `250a3f406f83d3b0cc95be7ccdc666d043e28eb5b5c0f9fb25e450e26ee17240`
- p15 `side_sdg2` prereg: `23acb036cd1a26f577cff8145ef4031f1c4075af3e4e60f1df28a42d86da8330`
- p15 candidates: `67eb07d68268be25c894c47fb6bee79347e2b201dea643c4823a0873bfcde384`
- p15 config: `dc93153cc2b8667b5156538b51140c3ea5eb1f1da19f507e5cd0f1227721638c`
- p15 Rerun validation: `18d98f66da9bb33da20a7965f9b04acf5bb0b9514c88911234f5f8a8959d8cc2`
- p15 inspection PNG: `fb76856e17ba301ccd94e9388387af8b91eef6c004d91dd5df7c2462bb87cc8f`
- p15 manual visual record: `0c363c28cf71c4700496bb81cb118260a450505d7e664f3644873246223c95ce`
- p15 detached stdout: `9fd8b60355774d16d111cc89a4e45e17ebc09622659d44a6616883ddf646d3f8`
- p15 PID record: `074dd59abdb4ecfade74cfaf00e05f150534b479e265101d5f5f12958ff86353`
- p15 exit-status record (`0`): `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`
- p14 helper: `fcaa7b1c6aeea65cd7fd335d9cd17ee5424a53d81764f67642d074a28e3e0133`
- p10 kinematics/planner: `63c6b2127d969e3291da6943eab6da1037034c154a8f21fe447519cbcb2f6cff`
- jaw extractor: `bca4f898023f63f21d540483a169499760038c582ce3a7919d7622e77946e1c3`
- parsed URDF: `64dc8d082cbce9a13a0697a11a0eaeaebbc54bbcd97e7aafaa40da483969dae2`
- witness results: `0f169bfababc458e98912c0aa3592def7935c791b30374235a0f1962f154fb26`
- witness plan: `2871c714a5b2f08519944a280d4a95b352566a0f5e0da53d41bf519282ade5bf`

After bytes are fixed, both p16 prereg full SHA-256 values are pinned in p16 source; the
p15 output SHA is supplied by the exact launch argument and recorded; and p16 computes,
freezes, and records its own executed-source SHA without a self-referential literal.
Any mismatch aborts before AppLauncher.  Source and every decision dependency are
re-hashed after physics; changing p16 during either invocation is forbidden.
