# Session: 2026-06-04 Cube10cm DiffIK Teacher Gate Prep

## Scope

- Professor cube push/tap branch only.
- No B200/SSH/reconnect/pull or `.ssh` copy.
- No Track A grasp/dataset/training/runtime.
- One local GPU IsaacLab tiny 128 runtime was run after explicit approval.
- No 1024/10k scale-up, dataset generation, PPO scale-up, VLA training, or
  Track A runtime was run.

## Term Definitions Used In This Session

- `GPU escalated`: Codex default sandbox hides `/dev/nvidia*`; any IsaacLab/PhysX
  command that needs the local GPU must be run with
  `sandbox_permissions=require_escalated`.
- `tiny 128 gate`: a small 128-environment smoke gate, not training and not a
  dataset run. It checks whether the configured object/controller/trajectory can
  produce stable controlled pushes before any larger 1024/10240 data collection.
- `teacher`: here means the scripted IsaacLab `DifferentialIKController` path
  that produces TCP targets and joint targets from live PhysX Jacobians.
- `teacher-off`: in the learned-policy actor/eval branch, this means disabling
  teacher blending. For this DiffIK probe, the more precise term is
  `DiffIK teacher gate`, because the controller itself is the diagnostic teacher
  and the script already sets training/dataset generation false.

## Verified Context

- Current 3cm env constants remain fixed in `roarm_rl/roarm_cube_push_env.py`:
  `CUBE_SIZE_M=0.030`, mass `0.020kg`, default target displacement `0.040m`,
  strict success displacement `0.030m`.
- Existing 2026-06-04 hierarchical sharded-10k audit remains 3cm evidence only:
  it logs `cube_size_m=0.030000`, `cube_mass_kg=0.020000`, density
  `740.741kg/m^3`, and displacement-only 1/5/10/20/30mm columns.
- Professor's current request is interpreted as a separate 10cm cube/object
  diagnostic, not as Track A grasp and not as a requirement to move the object
  one full object length.

## Code Prep

- Updated `sim_scripts/cube3cm_push_diffik_probe.py` without changing its default
  3cm behavior:
  - added `--cube_size_m`, `--cube_mass_kg`, `--cube_push_target_disp_m`,
    `--cube_success_disp_m`, and `--gate_disp_m`;
  - sets the IsaacLab cuboid size, mass, and z-center from these args before
    `gym.make`;
  - logs cube size, mass, density, object-size reference, target displacement,
    success displacement, and gate displacement;
  - computes the TCP precontact/through points from the configured object size,
    not the old imported 3cm constant;
  - records `disp_over_object_size`, `disp_ge_gate_rate`, and
    1/5/10/20/30mm threshold rates.

## Local Verification

- `python -m py_compile sim_scripts/cube3cm_push_diffik_probe.py`: PASS.
- `python sim_scripts/cube3cm_push_diffik_probe.py --help`: PASS and showed the
  new size/mass/gate args.
- `git diff --check`: PASS.

## Approved Tiny 128 Runtime

After explicit approval, ran:

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab --no-capture-output python -u \
  sim_scripts/cube3cm_push_diffik_probe.py \
  --num_envs 128 \
  --episodes 1 \
  --seed 930 \
  --cube_size_m 0.100 0.100 0.100 \
  --cube_mass_kg 0.720 \
  --cube_push_target_disp_m 0.010 \
  --cube_success_disp_m 0.010 \
  --gate_disp_m 0.010 \
  --out_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_gate128_seed930.csv \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_gate128_seed930_summary.json
```

Result: FAIL as a 1cm gate.

- Summary lines 11-25: controlled `0.1875`, `disp_along_push_mean_m=-0.0001524518520454876`,
  `disp_ge_gate_rate=0.0`.
- Summary lines 27-34: only the 1mm displacement tier reached `0.1875`; 5/10/20/30mm all `0.0`.
- Summary lines 41-52: final TCP target error mean `0.15134897292591631m`,
  low-motion `1.0`, max cube speed mean `0.0004937104492910294m/s`,
  min TCP-cube distance mean `0.13646007765782997m`.
- Summary lines 56 and 63: posewrite during rollout `0` and rollout object
  posewrite false, so the failure is not hidden object pose-writing.
- CSV had 128 rows plus header. Direction analysis showed every direction had
  `ge10mm=0.0`, `low=1.0`, and `diffik_clip_rate=1.0`; mean final TCP target
  error was roughly `0.14-0.17m` by direction.

Interpretation:

- The 10cm/0.72kg object did not fail because 1cm is too strict. It failed
  earlier: the default v1 trajectory/controller did not reliably reach/contact
  the larger object, and the object barely moved.
- Do not scale this to 1024/10k data, dataset generation, PPO/RL, VLA, or Track A.
- The next valid work is small geometry/control diagnosis: contact height,
  precontact/through distance, horizon, and DiffIK joint-step clipping for a
  10cm object, then another tiny gate only after explicit approval.

## Post-Fail Code Review Before Further Edits

After the failed 128 gate, reviewed the relevant code before making another
change.

Findings:

1. Reset z was still 3cm-specific. `sim_scripts/cube3cm_push_diffik_probe.py`
   set `env_cfg.sponge.init_state.pos` from the requested 10cm size, but
   `roarm_rl/roarm_cube_push_env.py` reset used the module constant
   `CUBE_CENTER_Z`, derived from `CUBE_SIZE_M=0.030`. For a 10cm object this is
   a 35mm z-center error.
2. The first 10cm gate mixed geometry validation with random x/y and random push
   direction. That is useful for robustness later, but it is too broad for
   debugging whether one easy, centered push works.
3. The current probe target height mode is a top-margin path. For the professor's
   "push the front face" mental model, the next diagnostic should be able to use
   side-face center height.
4. The failed run's `diffik_clip_rate_mean=1.0` and final TCP target error mean
   `0.15134897292591631m` should be interpreted as a reach/path issue before
   treating mass/friction as the limiting factor.

## Static Patch After Review

Patched without running another IsaacLab/GPU runtime:

- `roarm_rl/roarm_cube_push_env.py`
  - added config fields `cube_size_x_m`, `cube_size_y_m`, `cube_size_z_m`;
  - reset now uses `TABLE_Z + cube_size_z_m / 2` instead of the 3cm
    `CUBE_CENTER_Z` constant;
  - local precontact/teacher target helpers now compute half extents from the
    configured object size;
  - added optional fixed push direction config.
- `sim_scripts/cube3cm_push_diffik_probe.py`
  - passes the configured cube size into the env config, not just the spawn
    object;
  - added `--fixed_cube_x_m`, `--fixed_cube_y_m`, and `--fixed_push_dir`;
  - added `--tcp_height_mode top_margin|side_center`;
  - logs `table_z_m`, `cube_center_z_m`, and per-row `cube_z0_m` so the reset
    height is auditable from logs.

Static verification:

- `python -m py_compile roarm_rl/roarm_cube_push_env.py sim_scripts/cube3cm_push_diffik_probe.py`: PASS.
- `python sim_scripts/cube3cm_push_diffik_probe.py --help`: PASS and showed the
  fixed-position, fixed-direction, and TCP-height-mode args.
- `git diff --check`: PASS.
- Direct `conda run -n isaaclab python -c "from roarm_rl..."` import was not a
  valid check because IsaacLab/Omniverse modules require `SimulationApp` /
  `AppLauncher` before import. No runtime claim is made from that failed import.

## Fixed 16 Candidate Shape

This candidate was run after explicit approval with GPU escalation:

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab --no-capture-output python -u \
  sim_scripts/cube3cm_push_diffik_probe.py \
  --num_envs 16 \
  --episodes 1 \
  --seed 931 \
  --cube_size_m 0.100 0.100 0.100 \
  --cube_mass_kg 0.720 \
  --fixed_cube_x_m 0.300 \
  --fixed_cube_y_m 0.000 \
  --fixed_push_dir 1 0 \
  --tcp_height_mode side_center \
  --precontact_clearance_m 0.010 \
  --push_through_m 0.010 \
  --approach_steps 220 \
  --push_steps 90 \
  --post_steps 40 \
  --max_diffik_joint_step_rad 0.035 \
  --cube_push_target_disp_m 0.010 \
  --cube_success_disp_m 0.010 \
  --gate_disp_m 0.010
```

Interpretation:

- This is not dataset generation, not PPO/RL, not VLA, and not Track A.
- First check `cube_z0_m`, `fixed_push_dir`, `tcp_height_mode`, final TCP target
  error, min TCP-cube distance, and only then the 1cm displacement gate.
- If this fixed easy case fails, do not add randomization; debug geometry/reach
  with trace/video.

## Approved Fixed 16 Runtime

Result: FAIL as a 1cm fixed-geometry gate.

- Summary lines 11-26: controlled `0.0`, `diffik_clip_rate_mean=1.0`,
  `disp_along_push_mean_m=0.000025190412998199463`, and
  `disp_ge_gate_rate=0.0`.
- Summary lines 28-35: 1/5/10/20/30mm rates were all `0.0`; mean xy
  displacement was `0.000027741372491618677m`.
- Summary lines 42-59: final TCP target error mean
  `0.1307708825916052m`, min TCP-cube distance mean
  `0.07594270585104823m`, and min TCP target error mean
  `0.0738731506280601m`.
- Summary lines 63-76: posewrite during rollout remained `0`, rollout object
  posewrite false, `tcp_height_mode=side_center`, and trace CSV was written.
- Per-env CSV lines 2-4 still logged `cube_z0_m=0.03788299858570099`, while
  trace lines 2-5 showed the actual settled cube z was about `0.049999m`.
  Trace line 2 also showed `target_z_m=0.03788299858570099`, so the side-center
  TCP target was generated from the reset buffer, not the settled PhysX center.
- Trace lines 278-281 showed the final target x was `3.359999895095825`, but TCP
  x stayed around `3.2489-3.2494` and TCP z around `0.106-0.107m`, with only
  ~0.03mm object displacement. This is still a reach/target-generation/clipping
  failure before mass/friction conclusions.

## Post Fixed-16 Code Review And Static Patch

Reviewed the code before editing again.

Findings:

1. `roarm_rl/roarm_stack_env.py` uses a plane terrain, while `TABLE_Z` is a
   legacy sponge geometry constant. For the 10cm cuboid, PhysX contact resolution
   settled the actual cube center near `z=0.050m`, not the reset buffer
   `TABLE_Z + 0.050 = 0.037883m`.
2. `sim_scripts/cube3cm_push_diffik_probe.py` took `cube_start_w` from
   `inner._cube_start_w` after settling, so `compute_tcp_targets()` used the reset
   buffer z for side-center targeting.
3. This does not invalidate the failure verdict, but it makes the gate too broad:
   the robot was chasing a target height that did not match the actual settled
   object center.

Patched without another GPU runtime:

- After settle, the probe now copies `inner._sponge_pos_w` into `cube_start_w`
  and `inner._cube_start_w`, then updates the internal target buffer from that
  settled pose.
- Summary now logs `cube_start_z_mean_m` so requested reset z and actual
  diagnostic start z can be compared directly.

Static verification:

- `python -m py_compile sim_scripts/cube3cm_push_diffik_probe.py`: PASS.
- `python sim_scripts/cube3cm_push_diffik_probe.py --help`: PASS.
- `git diff --check`: PASS.

Next valid runtime, only after explicit approval with GPU escalation: rerun the
same fixed 16 gate with the settled-start patch and new output paths. If it still
fails with high TCP target error and `diffik_clip_rate_mean=1.0`, the next step is
not randomization or data generation; it is trace/video diagnosis of reachable TCP
height/orientation and joint-limit/clipping behavior.

## Approved Settled-Start Fixed 16 Runtime

After explicit approval, reran the same 16-env fixed-position/fixed-direction
side-center gate with the settled-start patch.

Result: FAIL as a 1cm gate, but the z mismatch was fixed.

- Summary lines 13 and 22: requested reset-derived `cube_center_z_m=0.037883`,
  but actual settled diagnostic start `cube_start_z_mean_m=0.04999994789250195`.
- Summary lines 25-36: `diffik_clip_rate_mean=1.0`,
  `disp_along_push_mean_m=0.00001109391450881958`, `disp_ge_gate_rate=0.0`,
  and all 1/5/10/20/30mm rates `0.0`.
- Summary lines 43-60: final TCP target error mean
  `0.12522956123575568m`, min TCP target error mean `0.06143945129588246m`,
  and min TCP-cube distance mean `0.07620850298553705m`.
- Summary lines 64-77: posewrite calls `0`, rollout object posewrite false, and
  trace CSV written.
- Trace lines 2-5: target z now matches the settled cube z around `0.050m`, so
  the previous reset-buffer z bug is no longer the immediate explanation.
- Trace lines 278-281: final target x is `3.3600244522094727`, but TCP x remains
  around `3.2484-3.2491` and TCP z around `0.1068-0.1075m`; object displacement
  stays around 0.008-0.012mm.

Local non-GPU kinematics check:

- `sim_scripts/roarm_kinematics.py` uses `TCP_LOCAL_OFFSET_M=0.115428m` in the
  URDF chain, consistent with the runtime TCP offset.
- Pure local DLS IK from HOME solved the side-center precontact target
  `(0.2400246,0,0.0499997)` with `err_mm=1.334`, and the through target
  `(0.3600245,0,0.0499997)` with `err_mm=0.461`.
- Therefore the remaining failure is not a proof that the point is unreachable
  in the project kinematic model. It is now narrowed to the IsaacLab
  DifferentialIK rollout path: link5 Jacobian/body target mapping, position-only
  controller convergence, joint-step clipping, actuator tracking, or collision
  constraints.

Next:

- Do not scale.
- Do not generate 10,240 data yet.
- Next code work must start with review of the DiffIK body/Jacobian mapping and
  actuator target trace. The next runtime should be a smaller trace/video
  diagnostic, not another 128/1024 gate.

## Approved Trace / Target Geometry Diagnostics

Scope:

- Still professor 10cm/0.72kg cube push/tap branch only.
- No B200, no Track A, no dataset generation, no PPO/RL/VLA, no 128/1024/10k.
- All GPU/IsaacLab diagnostics were explicit-approval local tiny 4-env runs.

Code review before edits:

- `sim_scripts/cube3cm_push_diffik_probe.py` lines 291-297 create IsaacLab
  `DifferentialIKController` with position command and DLS.
- Lines 287-289 use `link5_body_idx=5`, fixed-base `jacobi_body_idx=4`, and arm
  joint ids 0-4.
- Lines 457-480 convert TCP target to a link5 body target using current link5
  quaternion and `tcp_local`.
- Lines 539-570 now log before/after target errors, raw delta, clipped delta,
  commanded target, and actual joint position after `env.step`.
- `roarm_rl/roarm_stack_env.py` lines 173-180 show the default arm actuator is
  stiffness `80.0`, damping `4.0`, effort limit `2.5`, velocity limit `3.14`.

Static edits:

- Added `--trace_diffik_diagnostics` and a local analyzer
  `sim_scripts/cube3cm_push_diffik_trace_diagnostic_audit.py`.
- Added default-preserving actuator overrides
  `--arm_stiffness_override`, `--arm_damping_override`,
  `--arm_effort_limit_sim_override`, and `--arm_velocity_limit_sim_override`.
- Added `--through_target_mode {far_face,near_face}`. Code lines 471-475 now
  make the distinction explicit:
  `far_face = cube + half + push_through`; `near_face = cube - half + push_through`.
- Static checks passed: `py_compile`, `--help`, `git diff --check`. The analyzer
  rejected an old non-extended trace as a negative control.

Runtime diagnostics:

- Fixed 4-env far-face baseline, cap `0.035`, seed932: FAIL. Summary lines
  11/25-28/43/65-68 show controlled `0.0`, gate `0.0`, clip `1.0`, final TCP
  error `0.125396907m`, low-motion `1.0`, min TCP-cube `0.076345483m`.
  Analyzer lines 97-100 classify link5 target not reached, joint-step clipping,
  and actuator target-tracking lag.
- cap `0.120`, seed932: positive control but not teacher-ready. Summary lines
  11/25-28/43/65-67 show controlled `1.0`, gate `1.0`, mean displacement
  `0.089894325m`, clip `1.0`, final TCP error `0.094675537m`, low-motion `0.0`.
  This proves motion is possible, but it overshoots the 1cm objective by a lot.
- Long approach with cap `0.035`, seed933: FAIL. Summary lines 19/21/25-28/43/65-68
  show 830 steps, controlled `0.0`, gate `0.0`, clip `1.0`, final TCP error
  `0.124230925m`, low-motion `1.0`, min TCP-cube `0.075801797m`.
- Far-face drive boost, seed934: diagnostic positive but not teacher-ready.
  Summary lines 2-11 record stiffness `400`, damping `20`, effort `25`,
  velocity `12`; lines 21/36-38/53/75-77 show controlled `0.75`, mean
  displacement `0.067951232m`, gate `0.75`, final TCP error `0.101294972m`,
  low-motion `0.25`, clip `1.0`.
- Near-face default actuator, seed935: target geometry improved error but did
  not reach contact. Summary lines 21/35-37/53/75-79/95 show controlled `0.0`,
  gate `0.0`, clip `1.0`, final TCP error `0.061259050m`, low-motion `1.0`,
  min TCP-cube `0.082745695m`, `through_target_mode=near_face`.
- Near-face long approach, seed936: still FAIL. Summary lines 19/21/35-37/53/75-79/95
  show 830 steps, gate `0.0`, final TCP error `0.061997696m`, low-motion `1.0`,
  min TCP-cube `0.083884733m`.
- Near-face drive boost, seed937: gate PASS but overshoot. Summary lines
  21/36-38/53/75-79/91/95 show controlled `1.0`, mean displacement
  `0.050082028m`, gate `1.0`, final TCP error `0.124849608m`, low-motion `0.0`,
  success marker `1.0`, near-face mode. Per-env CSV lines 2-5 all moved about
  5cm, not about 1cm.

Interpretation:

- The old far-face target is a bad 10cm-object teacher target: it asks the TCP to
  travel through the cube to far-face + push-through.
- Near-face target geometry is necessary but not sufficient. It halves the final
  target error, but the default actuator/step path still stops about 8cm from the
  cube center and never contacts.
- Strong cap/drive controls prove the object can move, but they overshoot by
  5-9cm and still show high TCP error/clipping, so they are not teacher data.

Next:

- Do not scale this.
- Design one tiny controlled near-face contact controller: approach/contact
  phases, actual-contact or displacement stop, smaller/contact-aware push
  increments, and actuator/step scheduling. It must be judged on about-1cm
  displacement and low TCP error, not just `disp_ge_gate_rate=1.0`.

## Previous Gate Candidate

This was the initial candidate and has now failed:

```bash
python sim_scripts/cube3cm_push_diffik_probe.py \
  --num_envs 128 \
  --episodes 1 \
  --seed 930 \
  --cube_size_m 0.100 0.100 0.100 \
  --cube_mass_kg 0.720 \
  --cube_push_target_disp_m 0.010 \
  --cube_success_disp_m 0.010 \
  --gate_disp_m 0.010
```

Expected interpretation:

- Primary metric is not 10cm displacement. It is whether a 10cm/0.72kg object can
  be pushed about 1cm in a controlled, no-impact, low-motion-safe way.
- Report `disp_ge_gate_rate`, 1/5/10/20/30mm rates, `disp/object_size`,
  controlled, no-impact, low-motion, TCP target error, and posewrite count.
- The failed result blocks immediate 1024/10240 scale-up, dataset generation,
  PPO scale-up, VLA training, and Track A runtime.

## Metric Correction: Reaction Event Primary For Push/Tap

User clarification:

- For the professor's 10cm/0.72kg cube push/tap objective, final displacement does
  not have to be the primary success criterion.
- If the cube is pushed/tapped, reacts, lifts slightly, and then settles back near
  the original position, that can still satisfy the practical objective.
- Therefore final 1cm displacement is a secondary relocation metric unless the
  professor explicitly asks for sustained relocation.

Static/code update:

- `sim_scripts/cube3cm_push_diffik_probe.py` lines 61-63 add explicit reaction
  thresholds: `--reaction_disp_m`, `--reaction_z_delta_m`, and
  `--reaction_speed_mps`.
- Lines 247-252 validate those thresholds.
- Lines 589-605 track transient max displacement, max z delta, max tip angle, and
  contact/stop state.
- Lines 891-897 compute `reaction_event`; lines 932-936 write max transient
  displacement/z/tip fields and `reaction_event` into each per-env row.
- Lines 1137-1147 summarize max transient displacement, transient gate rates,
  `reaction_event_rate`, max z delta, and max tip angle.
- Static checks passed after the reaction-metric update:
  `python -m py_compile sim_scripts/cube3cm_push_diffik_probe.py sim_scripts/cube3cm_push_diffik_trace_diagnostic_audit.py`,
  `python sim_scripts/cube3cm_push_diffik_probe.py --help`, and
  `git diff --check`.

Measured-stop diagnostics:

- seed938, near slowdown `contact_near_joint_step_scale=0.35`, was too
  conservative. Summary lines 21-31 show measured-stop mode but no stop; lines
  50-52 show final displacement only `1.30385160446167e-05m` and final gate
  `0.0`; lines 68-71 show final TCP error `0.06104395352303982m`, first contact
  `-1`, and first stop `-1`; lines 96-100 show measured contact `0.0` while
  near range was seen.
- seed939, no near slowdown, reached contact/stop and then retracted. Summary
  lines 21-31 show stop seen `1.0`; lines 50-52 show final displacement
  `0.0014313608407974243m` and final gate `0.0`; lines 68-71 show first contact
  mean `261.5` and first stop mean `279.75`; lines 94-100 show max speed
  `0.1447020135819912m/s`, measured contact `1.0`, and near-contact `1.0`.
- seed940, measured-stop freeze, is the key clarified-objective result. Summary
  lines 21-32 show freeze mode and stop seen `1.0`; lines 51-53 show final
  displacement `0.0014494359493255615m` and final gate `0.0`; lines 95-100 show
  max speed `0.14879385754466057m/s`, max 8-15mm transient rate `1.0`, max
  transient displacement mean `0.010990217328071594m`, no overshoot, and transient
  1cm rate `1.0`; lines 101-106 show measured contact and near-contact both
  `1.0`.
- seed940 per-env CSV lines 2-5 show all 4 fixed envs had measured contact,
  contact stop, no overshoot, final displacement about `1.38-1.56mm`, max
  transient displacement about `10.81-11.26mm`, and max speed about
  `0.138-0.164m/s`.
- seed940 trace analyzer still reports controller-quality blockers: diagnostic
  summary lines 2-29 show joint clipping; lines 97-100 list
  `LINK5_BODY_TARGET_NOT_REACHED`, `JOINT_STEP_CLIPPING_DOMINANT`, and
  `ACTUATOR_TARGET_TRACKING_LAG`.

Interpretation:

- Under the old final 1cm relocation gate, seed940 fails because the cube settles
  back to about 1.45mm final displacement.
- Under the clarified push/tap reaction criterion, seed940 is a fixed-geometry
  reaction-event pass: contact happened, the cube transiently moved about 1cm,
  no overshoot was recorded, and speed evidence is strong.
- Do not call the 0.72kg cube too heavy or the task impossible.
- Do not call this teacher/data/RL readiness either. The DiffIK path still clips
  and lags, so the next work is metric definition and tiny reaction screening,
  not direct dataset generation or PPO/RL scale-up.

Scale/randomization analysis:

- NVIDIA's Isaac Lab docs state that Isaac Lab trains with many environments in
  parallel and uses GPU-accelerated workflows; the NVIDIA Isaac Lab developer page
  describes it as GPU-accelerated, agent-ready, and designed to train policies at
  scale.
- The Isaac Gym paper reports a GPU-native simulation/training path with 2-3
  orders of magnitude speedups over conventional CPU-simulator RL. So the
  professor's intuition that large-scale IsaacLab search can find candidates is
  directionally reasonable.
- Domain randomization and ADR papers support simulation-to-real only when the
  randomized distribution is deliberately built to cover the real-world gap.
  They do not say that one selected lucky trajectory is sufficient evidence.
- Simple random search can be competitive in some RL/control settings, but the
  same paper emphasizes high variability across seeds and hyperparameters. That
  reinforces the need for held-out seeds, not cherry-picking.
- Probability check: for per-trial success probability `p`, one or more successes
  in `N=1,000,000` trials happens with probability `1-(1-p)^N`. If `p=1e-6`, this
  is about 63%; if `p=1e-7`, about 9.5%; if `p=1e-8`, about 1%. One success in
  one million can therefore mean "the event exists" but can also mean "the event
  is too rare/brittle to be useful."

Next:

- Define the professor 10cm reaction-event gate explicitly:
  measured contact or `reaction_event`, max transient displacement/z/speed, no
  posewrite, no Track A, and trace/video audit.
- Only after that, ask for approval for a tiny randomized reaction screen. Do not
  run dataset generation, PPO/RL scale-up, VLA, 1024/10k, or Track A from seed940.

Sources:

- `sim_scripts/cube3cm_push_diffik_probe.py:61-63,247-252,589-605,891-897,932-936,1137-1147`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_measuredstop_driveboost_seed938_summary.json:21-118`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_measuredstop_driveboost_noslow_seed939_summary.json:21-118`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_measuredstop_freeze_driveboost_seed940_summary.json:21-118`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_measuredstop_freeze_driveboost_seed940.csv:1-5`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_measuredstop_freeze_driveboost_seed940_trace_diagnostic_summary.json:2-100`
- NVIDIA Isaac Lab docs:
  https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/train-your-first-robot-with-isaac-lab/02-how-isaac-lab-accelerates-reinforcement-learning.html
- NVIDIA Isaac Lab developer page: https://developer.nvidia.com/isaac/lab
- Isaac Gym paper: https://arxiv.org/abs/2108.10470
- Domain Randomization paper: https://arxiv.org/abs/1703.06907
- OpenAI Rubik's Cube ADR paper: https://arxiv.org/abs/1910.07113
- Simple random search RL paper: https://arxiv.org/abs/1803.07055

## Reaction Gate Audit Tool

Purpose:

- Convert the clarified professor push/tap metric into a reusable local audit
  before any new IsaacLab runtime or RL.
- Avoid two bad interpretations:
  1. treating final displacement as the only success metric,
  2. treating speed-only jitter as push/tap success.

Code:

- Added `sim_scripts/cube10cm_reaction_event_gate_audit.py`.
- Lines 1-5 state it reads existing summary/per-env CSV logs only and does not
  run IsaacLab, train, generate data, or touch the robot.
- Lines 73-83 expose thresholds:
  reaction displacement `0.001m`, z delta `0.002m`, speed `0.020m/s`, transient
  gate `0.010m`, overshoot `0.020m`, min reaction/contact rates `1.0`, teacher
  final TCP error `0.030m`, and teacher max clip rate `0.50`.
- Lines 118-138 compute reaction, contact evidence, overshoot, and transient
  gate from CSV rows. This lets the audit work on older seed939/940 CSVs that do
  not yet have an explicit `reaction_event` column.
- Lines 140-167 enforce no posewrite/training/data/grasp-attach and split:
  `reaction_gate_pass`, `final_relocation_pass`, and `teacher_quality_ready`.
- Lines 173-212 write JSON evidence; lines 220-238 print three compact audit
  lines.

Commands:

```bash
python -m py_compile sim_scripts/cube10cm_reaction_event_gate_audit.py sim_scripts/cube3cm_push_diffik_probe.py sim_scripts/cube3cm_push_diffik_trace_diagnostic_audit.py

python sim_scripts/cube10cm_reaction_event_gate_audit.py \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_measuredstop_driveboost_seed938_summary.json \
  --trace_diag_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_measuredstop_driveboost_seed938_trace_diagnostic_summary.json \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_measuredstop_driveboost_seed938_reaction_gate_audit.json

python sim_scripts/cube10cm_reaction_event_gate_audit.py \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_measuredstop_driveboost_noslow_seed939_summary.json \
  --trace_diag_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_measuredstop_driveboost_noslow_seed939_trace_diagnostic_summary.json \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_measuredstop_driveboost_noslow_seed939_reaction_gate_audit.json

python sim_scripts/cube10cm_reaction_event_gate_audit.py \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_measuredstop_freeze_driveboost_seed940_summary.json \
  --trace_diag_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_measuredstop_freeze_driveboost_seed940_trace_diagnostic_summary.json \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_measuredstop_freeze_driveboost_seed940_reaction_gate_audit.json
```

Results:

- seed938 is the negative control and FAILs the reaction gate. Audit JSON lines
  2-3 show computed reaction `0.5` but contact evidence `0.0`; lines 19-20 show
  reaction rate `0.5` and `reaction_gate_pass=false`; lines 24 and 28 show final
  TCP error `0.06104395352303982m` and teacher not ready. This proves the audit
  does not accept speed-only motion without contact evidence.
- seed939 PASSes reaction but is not teacher-ready. Audit JSON lines 2-3 show
  reaction/contact `1.0`; lines 19-20 show `reaction_gate_pass=true`; lines 22,
  24, and 28 show DiffIK clip `1.0`, final TCP error `0.06364072300493717m`, and
  `teacher_quality_ready=false`; line 42 shows transient 1cm gate `0.0`.
- seed940 is the stronger reaction PASS. Audit JSON lines 2-3 show
  reaction/contact `1.0`; lines 14-20 show max displacement
  `0.010990217328071594m`, speed `0.14879385754466057m/s`, no overshoot, and
  `reaction_gate_pass=true`; lines 21-28 show stop/contact `1.0`, final
  displacement gate `0.0`, final TCP error `0.059237909503281116m`, DiffIK clip
  `1.0`, and `teacher_quality_ready=false`; line 42 shows transient 1cm gate
  `1.0`.

Interpretation:

- The professor 10cm/0.72kg object can react under fixed geometry. Do not call it
  too heavy or impossible.
- The current controller is still not teacher/data/RL ready. High DiffIK clip
  and TCP target error remain.
- The next valid runtime, only after explicit GPU approval, is a tiny randomized
  reaction screen using this audit. Do not run dataset generation, PPO/RL
  scale-up, VLA, Track A, 1024/10k, or million-rollout search from these fixed
  runs.

Sources:

- `sim_scripts/cube10cm_reaction_event_gate_audit.py:1-5,73-83,118-138,140-167,173-212,220-238`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_measuredstop_driveboost_seed938_reaction_gate_audit.json:1-44`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_measuredstop_driveboost_noslow_seed939_reaction_gate_audit.json:1-44`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_measuredstop_freeze_driveboost_seed940_reaction_gate_audit.json:1-44`

## Tiny Randomized Reaction Screen Seed941

Scope:

- This was exactly one approved local IsaacLab/GPU runtime for the professor
  10cm/0.72kg cube push/tap branch.
- It was not Track A, not B200, not dataset generation, not PPO/RL scale-up, not
  VLA, and not a 1024/10k or million-rollout sweep.
- Because no durable rule changed beyond D124, `DECISIONS.md` was not extended
  for this seed941 result.

Preflight:

```bash
python -m py_compile sim_scripts/cube10cm_reaction_event_gate_audit.py sim_scripts/cube3cm_push_diffik_probe.py sim_scripts/cube3cm_push_diffik_trace_diagnostic_audit.py
git diff --check
```

Both commands passed with no output.

Runtime:

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab --no-capture-output python -u sim_scripts/cube3cm_push_diffik_probe.py \
  --num_envs 16 \
  --episodes 1 \
  --seed 941 \
  --cube_size_m 0.100 0.100 0.100 \
  --cube_mass_kg 0.720 \
  --cube_push_target_disp_m 0.010 \
  --cube_success_disp_m 0.010 \
  --gate_disp_m 0.010 \
  --tcp_height_mode side_center \
  --through_target_mode near_face \
  --contact_controller_mode measured_stop \
  --contact_stop_target_mode freeze \
  --contact_detect_disp_m 0.001 \
  --contact_stop_disp_m 0.010 \
  --contact_overshoot_disp_m 0.020 \
  --contact_near_joint_step_scale 1.0 \
  --contact_stop_joint_step_scale 0.2 \
  --precontact_clearance_m 0.010 \
  --push_through_m 0.010 \
  --approach_steps 220 \
  --push_steps 90 \
  --post_steps 80 \
  --max_diffik_joint_step_rad 0.035 \
  --arm_stiffness_override 400 \
  --arm_damping_override 20 \
  --arm_effort_limit_sim_override 25 \
  --arm_velocity_limit_sim_override 12 \
  --trace_diffik_diagnostics \
  --trace_env_ids 0 1 2 3 \
  --trace_stride 5 \
  --out_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_rand16_reaction_seed941.csv \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_rand16_reaction_seed941_summary.json \
  --trace_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_rand16_reaction_seed941_trace.csv
```

Posthoc audits:

```bash
python sim_scripts/cube3cm_push_diffik_trace_diagnostic_audit.py \
  --trace_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_rand16_reaction_seed941_trace.csv \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_rand16_reaction_seed941_summary.json \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_rand16_reaction_seed941_trace_diagnostic_summary.json

python sim_scripts/cube10cm_reaction_event_gate_audit.py \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_rand16_reaction_seed941_summary.json \
  --trace_diag_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_rand16_reaction_seed941_trace_diagnostic_summary.json \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_rand16_reaction_seed941_reaction_gate_audit.json
```

Results:

- Runtime produced 16 CSV rows plus header and 312 trace rows plus header.
- Summary JSON lines 21-35 show measured-stop freeze settings, contact stop rate
  `0.4375`, controlled push `0.625`, and the DiffIK controller name.
- Summary lines 48, 69, and 92-100 show DiffIK clip `0.9706730805337429`,
  final TCP target error `0.063001801721839m`, max cube speed
  `0.21616149332839996m/s`, max z delta `0.012441999046131968m`, max transient
  displacement `0.006698679178953171m`, max transient gate `0.4375`, and
  measured contact `0.625`.
- Summary lines 109, 116, 120, and 139 show posewrite calls `0`, reaction event
  rate `1.0`, rollout object posewrite false, and 16 trials.
- Trace diagnostic JSON lines 97-100 report
  `LINK5_BODY_TARGET_NOT_REACHED`, `JOINT_STEP_CLIPPING_DOMINANT`, and
  `ACTUATOR_TARGET_TRACKING_LAG`; line 118 keeps `mechanism_ok=true`.
- Reaction audit JSON lines 2-3 show computed reaction `1.0` but contact
  evidence `0.625`; lines 17-20 show `no_posewrite=true`, overshoot `0.0`,
  reaction rate `1.0`, and `reaction_gate_pass=false`.
- Reaction audit JSON lines 21-28 show contact stop `0.4375`, DiffIK clip
  `0.9706730805337429`, final TCP error `0.063001801721839m`, measured contact
  `0.625`, and `teacher_quality_ready=false`.
- CSV lines 4, 6, 8, 10, 11, and 17 show representative no-contact reaction
  cases: `reaction_event=1` while `measured_contact_seen=0` and
  `contact_stop_seen=0`.

Interpretation:

- seed941 FAILs the reaction gate because contact evidence is incomplete
  (`0.625 < 1.0`). It does not fail on reaction rate, controller identity,
  no-posewrite, or overshoot.
- This is the same class of issue D124 was meant to catch: reaction-like z/speed
  changes are not enough without contact evidence.
- Teacher/RL/data readiness remains false because final TCP error is about
  `0.063m` and DiffIK clipping is about `0.971`.
- No further GPU runtime, dataset generation, PPO/RL scale-up, VLA, Track A,
  1024/10k, or million-rollout search is approved from this run.

Sources:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_rand16_reaction_seed941_summary.json:21-35,48,69,92-100,109,116,120,139`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_rand16_reaction_seed941_trace_diagnostic_summary.json:97-100,118`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_rand16_reaction_seed941_reaction_gate_audit.json:1-44`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_rand16_reaction_seed941.csv:1-17`

## Cap050 Diagnostic Seed942

Purpose:

- Test one narrow hypothesis after seed941: maybe the randomized contact misses
  are mostly caused by the `0.035rad` DiffIK joint-step cap.
- Change only the main cap to `0.050rad`; keep the same 10cm/0.72kg measured-stop
  freeze reaction setup.

Local seed941 direction breakdown:

- `x+`: 3 trials, contact `1.0`, transient gate `1.0`, no overshoot.
- `y-`: 4 trials, contact `1.0`, transient gate `0.75`, no overshoot.
- `y+`: 8 trials, contact `0.375`, transient gate `0.125`, no overshoot.
- `x-`: 1 trial, contact `0.0`, transient gate `0.0`, no overshoot.

Runtime:

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab --no-capture-output python -u sim_scripts/cube3cm_push_diffik_probe.py \
  --num_envs 16 \
  --episodes 1 \
  --seed 942 \
  --cube_size_m 0.100 0.100 0.100 \
  --cube_mass_kg 0.720 \
  --cube_push_target_disp_m 0.010 \
  --cube_success_disp_m 0.010 \
  --gate_disp_m 0.010 \
  --tcp_height_mode side_center \
  --through_target_mode near_face \
  --contact_controller_mode measured_stop \
  --contact_stop_target_mode freeze \
  --contact_detect_disp_m 0.001 \
  --contact_stop_disp_m 0.010 \
  --contact_overshoot_disp_m 0.020 \
  --contact_near_joint_step_scale 1.0 \
  --contact_stop_joint_step_scale 0.2 \
  --precontact_clearance_m 0.010 \
  --push_through_m 0.010 \
  --approach_steps 220 \
  --push_steps 90 \
  --post_steps 80 \
  --max_diffik_joint_step_rad 0.050 \
  --arm_stiffness_override 400 \
  --arm_damping_override 20 \
  --arm_effort_limit_sim_override 25 \
  --arm_velocity_limit_sim_override 12 \
  --trace_diffik_diagnostics \
  --trace_env_ids 0 1 2 3 \
  --trace_stride 5 \
  --out_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_rand16_reaction_cap050_seed942.csv \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_rand16_reaction_cap050_seed942_summary.json \
  --trace_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_rand16_reaction_cap050_seed942_trace.csv
```

Posthoc:

```bash
python sim_scripts/cube3cm_push_diffik_trace_diagnostic_audit.py \
  --trace_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_rand16_reaction_cap050_seed942_trace.csv \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_rand16_reaction_cap050_seed942_summary.json \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_rand16_reaction_cap050_seed942_trace_diagnostic_summary.json

python sim_scripts/cube10cm_reaction_event_gate_audit.py \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_rand16_reaction_cap050_seed942_summary.json \
  --trace_diag_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_rand16_reaction_cap050_seed942_trace_diagnostic_summary.json \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_rand16_reaction_cap050_seed942_reaction_gate_audit.json
```

Results:

- Summary JSON line 48 shows DiffIK clip `0.9432692341506481`, lower than seed941
  but still very high.
- Summary line 69 shows final TCP target error `0.05797268496826291m`.
- Summary lines 92-100 show max speed `0.18447698187083006m/s`, max z delta
  `0.01495442958548665m`, max transient displacement `0.004881829023361206m`,
  transient gate `0.1875`, and measured contact `0.5`.
- Summary lines 109, 116, 120, and 139 show posewrite calls `0`, reaction event
  `1.0`, rollout object posewrite false, and 16 trials.
- Reaction audit lines 2-3 and 17-20 show reaction `1.0`, contact evidence `0.5`,
  no posewrite, no overshoot, and `reaction_gate_pass=false`.
- Reaction audit lines 21-28 show contact stop `0.1875`, final TCP error
  `0.05797268496826291m`, clip `0.9432692341506481`, measured contact `0.5`,
  and `teacher_quality_ready=false`.
- Trace diagnostic lines 97-100 still report `LINK5_BODY_TARGET_NOT_REACHED`,
  `JOINT_STEP_CLIPPING_DOMINANT`, and `ACTUATOR_TARGET_TRACKING_LAG`.

Interpretation:

- cap050 is a failed diagnostic. It did not recover the randomized reaction gate
  and made contact evidence worse than seed941 (`0.625 -> 0.5`).
- The failure is more consistent with direction/geometry-conditioned reach and
  actuator tracking than with a simple global cap shortage.
- Do not keep raising cap as the next move. Do not start dataset generation,
  PPO/RL, VLA, Track A, 1024/10k, or broad random search from this.

Sources:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_rand16_reaction_seed941.csv:1-17`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_rand16_reaction_cap050_seed942_summary.json:48,69,92-100,109,116,120,139`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_rand16_reaction_cap050_seed942_reaction_gate_audit.json:1-44`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_rand16_reaction_cap050_seed942_trace_diagnostic_summary.json:97-100`

## Fixed Y+ Direction Diagnostic Seed943

Purpose:

- Direction-bucket the seed941 randomized failure.
- Keep original cap `0.035` and fix only the push direction to `y+` with
  `--fixed_push_dir 0 1`.
- This is still professor 10cm/0.72kg reaction diagnosis only: no B200, no
  Track A, no dataset generation, no RL/PPO, no VLA.

Runtime:

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab --no-capture-output python -u sim_scripts/cube3cm_push_diffik_probe.py \
  --num_envs 16 \
  --episodes 1 \
  --seed 943 \
  --cube_size_m 0.100 0.100 0.100 \
  --cube_mass_kg 0.720 \
  --cube_push_target_disp_m 0.010 \
  --cube_success_disp_m 0.010 \
  --gate_disp_m 0.010 \
  --fixed_push_dir 0 1 \
  --tcp_height_mode side_center \
  --through_target_mode near_face \
  --contact_controller_mode measured_stop \
  --contact_stop_target_mode freeze \
  --contact_detect_disp_m 0.001 \
  --contact_stop_disp_m 0.010 \
  --contact_overshoot_disp_m 0.020 \
  --contact_near_joint_step_scale 1.0 \
  --contact_stop_joint_step_scale 0.2 \
  --precontact_clearance_m 0.010 \
  --push_through_m 0.010 \
  --approach_steps 220 \
  --push_steps 90 \
  --post_steps 80 \
  --max_diffik_joint_step_rad 0.035 \
  --arm_stiffness_override 400 \
  --arm_damping_override 20 \
  --arm_effort_limit_sim_override 25 \
  --arm_velocity_limit_sim_override 12 \
  --trace_diffik_diagnostics \
  --trace_env_ids 0 1 2 3 \
  --trace_stride 5 \
  --out_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_reaction_seed943.csv \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_reaction_seed943_summary.json \
  --trace_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_reaction_seed943_trace.csv
```

Posthoc note:

- The first reaction audit attempt raced the trace audit file creation because
  those two local posthoc commands were launched in parallel. Trace audit
  completed, and rerunning only the reaction audit succeeded. No GPU runtime was
  repeated for this.

Results:

- Summary lines 75-78 confirm fixed push direction `[0.0, 1.0]`.
- Summary line 48 shows DiffIK clip `1.0`; line 69 shows final TCP target error
  `0.07060193479992449m`.
- Summary lines 95-103 show max speed `0.10947006440255791m/s`, max z delta
  `0.011328218039125204m`, max displacement `0.004163078963756561m`, transient
  gate `0.3125`, and measured contact `0.375`.
- Summary lines 112, 119, 123, and 142 show posewrite calls `0`, reaction event
  `0.9375`, rollout posewrite false, and 16 trials.
- Reaction audit lines 2-3 and 17-20 show reaction `0.9375`, contact evidence
  `0.375`, no posewrite, no overshoot, and `reaction_gate_pass=false`.
- Reaction audit lines 21-28 show contact stop `0.3125`, DiffIK clip `1.0`,
  final TCP error `0.07060193479992449m`, measured contact `0.375`, and
  `teacher_quality_ready=false`.
- Trace diagnostic lines 97-100 still report `LINK5_BODY_TARGET_NOT_REACHED`,
  `JOINT_STEP_CLIPPING_DOMINANT`, and `ACTUATOR_TARGET_TRACKING_LAG`; line 118
  keeps `mechanism_ok=true`.

Interpretation:

- y+ is now a confirmed weak direction bucket. The randomized seed941 failure was
  not just random noise.
- The next useful work is local y+ target path/reach/geometry diagnosis. Do not
  chase this with cap-only escalation, dataset generation, PPO/RL, VLA, Track A,
  1024/10k, or broad random search.

Sources:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_reaction_seed943_summary.json:48,69,75-78,95-103,112,119,123,142`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_reaction_seed943_reaction_gate_audit.json:1-44`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_reaction_seed943_trace_diagnostic_summary.json:97-100,118`

## Y+ Geometry/Reach Posthoc Audit

Purpose:

- Answer why the next work is y+ target path, reach, lateral/height offset, and
  actuator tracking instead of RL, 10240 generation, or another cap-only sweep.
- Use only existing seed943 local logs. No GPU/IsaacLab runtime, no B200, no
  Track A, no dataset generation, no PPO/RL, no VLA.

Local audit tool:

```bash
python sim_scripts/cube10cm_yplus_geometry_reach_audit.py \
  --summary_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_reaction_seed943.csv \
  --trace_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_reaction_seed943_trace.csv \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_reaction_seed943_yplus_geometry_reach_audit.json
```

Console summary:

```text
yplus_geometry line1 trials=16 trace_rows=312 contact_n=6 no_contact_n=10
yplus_geometry line2 contact_max_disp_mean=0.010986278 no_contact_max_disp_mean=0.000069159 contact_final_tcp_err=0.063392016 no_contact_final_tcp_err=0.074927886
yplus_geometry line3 y_le_0_contact=0.625000 y_gt_0_contact=0.125000 x_lt_025_contact=0.111111 x_ge_025_contact=0.714286
yplus_geometry line4 verdict=LOCAL_DIAG_NEXT_GEOMETRY_REACH_REQUIRED
```

Results:

- The new audit script is local only: lines 1-4 say it does not run IsaacLab,
  train, generate data, touch the robot, or reconnect any remote machine.
- Audit JSON lines 2-23 show contact rows `n=6`, line numbers 3/4/8/9/16/17,
  mean final TCP error `0.06339201641579469m`, mean max displacement
  `0.010986278454462687m`, and mean min TCP-target error
  `0.05519321064154307m`.
- Audit JSON lines 31-56 show no-contact rows `n=10`, line numbers
  2/5/6/7/10/11/12/13/14/15, mean final TCP error
  `0.07492788583040237m`, mean max displacement only
  `0.00006915926933288574m`, and mean min TCP-target error
  `0.0668345957994461m`.
- Audit JSON lines 58-122 show workspace asymmetry: `cube_y0_m<=0` contact
  `0.625`, `cube_y0_m>0` contact `0.125`, `cube_x0_m<0.25` contact
  `0.1111111111111111`, and `cube_x0_m>=0.25` contact
  `0.7142857142857143`.
- Audit JSON lines 126-139 show traced contact/no-contact groups both keep large
  final TCP-target vertical error. No-contact is worse at
  `0.06137282773852348m` mean abs z error, with final TCP-cube distance still
  about `0.08283714205026627m`.
- Audit JSON lines 141-240 show traced env0/env3 no-contact cases and env1/env2
  contact cases. Env0 final line 310 and env3 final line 313 keep side-center
  final z errors `0.052595339715480804m` and `0.07015031576156616m`; env1/env2
  contact/stop in better poses but still keep high z error.

Interpretation:

- The y+ failure is not solved by relabeling speed/z/tip reaction as success.
  The no-contact group has reaction-like motion but almost no displacement along
  the commanded push.
- The y+ failure is also not homogeneous random noise. It is workspace-position
  dependent in this 16-env sample, and the traced TCP remains too high relative
  to the side-center target.
- The next useful research is local target-path/reach diagnosis: compare
  side-center target z, actual TCP height, lateral/xy offset, workspace x/y
  bucket, joint clipping, and actuator follow error across contact vs no-contact
  rows.
- Do not start 10cm 10240, dataset generation, PPO/RL, VLA, Track A, or broad
  random search from this evidence.

Sources:

- `sim_scripts/cube10cm_yplus_geometry_reach_audit.py:1-4,141-180,185-218`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_reaction_seed943_yplus_geometry_reach_audit.json:1-245`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_reaction_seed943.csv:1-17`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_reaction_seed943_trace.csv:1-313`

## Y+ Trace Path/Actuator Audit

Purpose:

- Continue the local D127 diagnosis by separating three questions:
  target path, side-center height reach, and actuator clipping/follow lag.
- Use only existing seed943 summary CSV and trace CSV. No GPU/IsaacLab runtime,
  no B200, no Track A, no dataset generation, no PPO/RL, no VLA.

Local audit tool:

```bash
python sim_scripts/cube10cm_yplus_trace_path_actuator_audit.py \
  --summary_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_reaction_seed943.csv \
  --trace_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_reaction_seed943_trace.csv \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_reaction_seed943_yplus_trace_path_actuator_audit.json
```

Console summary:

```text
yplus_trace_path line1 traced_env_count=4 trace_rows=312 target_world_dy_mean=0.019999981 final_target_z_minus_start_cube_z_mean=0.000000086
yplus_trace_path line2 contact_envs=[1, 2] no_contact_envs=[0, 3] contact_final_z_err=0.051741816 no_contact_final_z_err=0.061372828
yplus_trace_path line3 contact_clip_any=1.000000 no_contact_clip_any=1.000000 contact_clip_joint_count=2.596154 no_contact_clip_joint_count=2.538462
yplus_trace_path line4 verdict=LOCAL_DIAG_GEOMETRY_HEIGHT_WORKSPACE_BEFORE_GPU_OR_DATA
```

Results:

- The new script is local only: lines 1-4 say it does not run IsaacLab, use GPU,
  train, generate data, touch the robot, or reconnect any remote machine.
- Audit JSON lines 2-54 show traced contact envs `[1, 2]` and no-contact envs
  `[0, 3]`. Both groups have target world-y delta
  `0.019999980926513672m`, final target z near start-cube z, and `clip_any=1.0`.
- Audit JSON lines 10-16 and 36-42 show final TCP error is mostly vertical:
  z-error fraction is `0.8440865598225584` for contact traced envs and
  `0.858887503603252` for no-contact traced envs.
- Audit JSON lines 112-142 show no-contact env0 final z error
  `0.052595339715480804m` and z-error fraction `0.7493973378831474`.
- Audit JSON lines 834-865 and 1000-1018 show no-contact env3 final z error
  `0.07015031576156616m`, z-error fraction `0.9683776693233566`, target world-y
  delta `0.019999980926513672m`, and worst follow/raw-delta joint 2.
- Audit JSON lines 1022-1038 summarize the interpretation: y+ target path is
  short/lateral-neutral, side-center z is near start-cube height, TCP remains
  several centimeters above target, and clipping/follow lag remain in both
  contact and no-contact traced groups.

Interpretation:

- The y+ target advance exists. Do not diagnose this as a missing target-y bug.
- The current failure is more consistent with side-center height reach,
  workspace-conditioned contact, and actuator clipping/follow lag.
- The next GPU test, if explicitly approved, should be tiny and scoped to one
  geometry/control hypothesis: target height, lateral offset, workspace x/y
  bucket, or actuator tracking. Do not start 10cm 10240, dataset generation,
  PPO/RL, VLA, Track A, or broad random search.

Sources:

- `sim_scripts/cube10cm_yplus_trace_path_actuator_audit.py:1-4,110-168,172-202,223-263`
- `sim_scripts/cube3cm_push_diffik_probe.py:497-520`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_reaction_seed943_yplus_trace_path_actuator_audit.json:1-1043`

## Fast Next Work Toward Data/RL

Purpose:

- Answer "what is next?" without jumping from an unstable 16-env teacher screen
  into 10cm 10240 data generation or RL.
- Keep the next action small enough to run quickly, but structured enough that a
  PASS/FAIL result changes the research path.
- No GPU/IsaacLab runtime was run in this section; this is a local protocol
  decision from existing seed941/942/943 logs.

Verified blockers:

- D124 requires reaction evidence plus contact evidence, no posewrite, and no
  overshoot before a reaction/tap PASS; teacher quality is a separate label and
  remains false when TCP error/clipping are high.
- seed943 fixed-y+ reaction audit JSON lines 2-3 and 17-28 show reaction
  `0.9375`, contact evidence only `0.375`, no posewrite, no overshoot,
  `reaction_gate_pass=false`, final TCP error `0.07060193479992449m`,
  DiffIK clip `1.0`, and `teacher_quality_ready=false`.
- The y+ geometry audit JSON lines 2-23 show contact rows move the cube
  transiently (`0.010986278454462687m` mean max displacement), while lines
  31-56 show no-contact rows move only `0.00006915926933288574m`. That means
  speed/z/tip motion alone is not enough for push/tap data.
- The y+ trace-path audit JSON lines 2-54 and 1022-1038 show the target advances
  in world y by about `0.020m`; the unresolved issue is final vertical
  TCP-target error, clipping, and actuator follow lag, not a missing y target.

Next executable hypothesis:

- The next tiny GPU screen, only after explicit approval, should be a fixed-y+
  target-height/reach discriminator. Keep the measured-stop freeze, near-face
  geometry, 16 envs, original cap `0.035`, and no data/RL. Change only
  `--tcp_center_height_offset_m` from `0.000` to a diagnostic positive offset
  such as `0.050`.
- Rationale: existing traced no-contact envs are several centimeters above the
  side-center target. If a height offset sharply improves measured contact while
  preserving no-posewrite/no-overshoot, the next research is contact-height and
  lateral refinement. If it does not improve contact, the height hypothesis is
  weak and the next narrow axis is fixed workspace x/y buckets or actuator
  tracking.

Decision gate for the next tiny screen:

- Reaction gate PASS requires reaction rate `1.0`, contact evidence `1.0`,
  no-posewrite true, and overshoot `0.0`.
- Teacher/data readiness still requires substantially lower final TCP error and
  DiffIK clipping. A reaction PASS alone may justify a candidate teacher
  refinement, but not 10cm 10240 data generation or PPO/RL.
- The first scalable milestone after this is not "millions of rollouts"; it is a
  10cm teacher micro-gate: y+ fixed screen PASS, then randomized 128 teacher
  screen PASS, then 1024/10k only if teacher_quality_ready becomes true or the
  data objective is explicitly downgraded to reaction-only with contact evidence.

Candidate command skeleton, later executed as seed944:

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab --no-capture-output python -u sim_scripts/cube3cm_push_diffik_probe.py \
  --num_envs 16 \
  --episodes 1 \
  --seed 944 \
  --cube_size_m 0.100 0.100 0.100 \
  --cube_mass_kg 0.720 \
  --cube_push_target_disp_m 0.010 \
  --cube_success_disp_m 0.010 \
  --gate_disp_m 0.010 \
  --fixed_push_dir 0 1 \
  --tcp_height_mode side_center \
  --tcp_center_height_offset_m 0.050 \
  --through_target_mode near_face \
  --contact_controller_mode measured_stop \
  --contact_stop_target_mode freeze \
  --contact_detect_disp_m 0.001 \
  --contact_stop_disp_m 0.010 \
  --contact_overshoot_disp_m 0.020 \
  --contact_near_joint_step_scale 1.0 \
  --contact_stop_joint_step_scale 0.2 \
  --precontact_clearance_m 0.010 \
  --push_through_m 0.010 \
  --approach_steps 220 \
  --push_steps 90 \
  --post_steps 80 \
  --max_diffik_joint_step_rad 0.035 \
  --arm_stiffness_override 400 \
  --arm_damping_override 20 \
  --arm_effort_limit_sim_override 25 \
  --arm_velocity_limit_sim_override 12 \
  --trace_diffik_diagnostics \
  --trace_env_ids 0 1 2 3 \
  --trace_stride 5
```

## Y+ Height050 Runtime Diagnostic

Purpose:

- Test one narrow hypothesis after the local trace audits: if the TCP is several
  centimeters above the side-center target, does raising the target height make
  fixed-y+ measured contact reliable?
- This was exactly one local GPU/IsaacLab tiny screen after approval. No B200, no
  SSH, no Track A, no dataset generation, no PPO/RL, no VLA, no 1024/10k.

Runtime:

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab --no-capture-output python -u sim_scripts/cube3cm_push_diffik_probe.py \
  --num_envs 16 \
  --episodes 1 \
  --seed 944 \
  --cube_size_m 0.100 0.100 0.100 \
  --cube_mass_kg 0.720 \
  --fixed_push_dir 0 1 \
  --tcp_height_mode side_center \
  --tcp_center_height_offset_m 0.050 \
  --through_target_mode near_face \
  --contact_controller_mode measured_stop \
  --contact_stop_target_mode freeze \
  --max_diffik_joint_step_rad 0.035 \
  --trace_diffik_diagnostics \
  --trace_env_ids 0 1 2 3
```

Console summary:

```text
[cube3cm_push_diffik_probe] summary trials=16 controlled_push_rate=0.000000 impact_outlier_rate=0.000000 low_motion_rate=1.000000 disp_along_push_mean_m=-0.000032 disp_ge_gate_rate=0.000000 disp_8_15mm_rate=0.000000 max_disp_along_push_mean_m=0.000059 max_disp_ge_gate_rate=0.000000 reaction_event_rate=0.687500 overshoot_ge_20mm_rate=0.000000 measured_contact_seen_rate=0.000000 contact_stop_seen_rate=0.000000 disp_xy_mean_m=0.000304 posewrite_calls_during_rollout=0 grasped_marker_rate=0.000000
```

Posthoc audits:

```bash
python sim_scripts/cube3cm_push_diffik_trace_diagnostic_audit.py \
  --trace_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_height050_seed944_trace.csv \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_height050_seed944_summary.json \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_height050_seed944_trace_diagnostic_summary.json

python sim_scripts/cube10cm_reaction_event_gate_audit.py \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_height050_seed944_summary.json \
  --trace_diag_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_height050_seed944_trace_diagnostic_summary.json \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_height050_seed944_reaction_gate_audit.json

python sim_scripts/cube10cm_yplus_geometry_reach_audit.py \
  --summary_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_height050_seed944.csv \
  --trace_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_height050_seed944_trace.csv \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_height050_seed944_yplus_geometry_reach_audit.json

python sim_scripts/cube10cm_yplus_trace_path_actuator_audit.py \
  --summary_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_height050_seed944.csv \
  --trace_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_height050_seed944_trace.csv \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_height050_seed944_yplus_trace_path_actuator_audit.json
```

Results:

- Summary JSON lines 47-49 and 69 show no dataset generation, DiffIK clip
  `0.9491987340152264`, and final TCP-target error
  `0.022889409447088838m`.
- Summary JSON lines 95-103 show max speed `0.06781863939249888m/s`, max z
  delta `0.004550501937046647m`, max displacement only
  `0.000058706849813461304m`, max 1cm gate `0.0`, and measured contact `0.0`.
- Summary JSON lines 112, 119, 123, and 142 show no posewrite, reaction
  `0.6875`, rollout posewrite false, and 16 trials.
- Reaction audit JSON lines 2-3 and 17-27 show reaction `0.6875`, contact
  evidence `0.0`, no posewrite, no overshoot, reaction gate false, final TCP
  error `0.022889409447088838m`, clip `0.9491987340152264`, measured contact
  `0.0`, and `teacher_quality_ready=false`.
- Reaction audit JSON line 41 shows transient 1cm gate `0.0`.
- Trace diagnostic JSON lines 97-100 still show `JOINT_STEP_CLIPPING_DOMINANT`
  and `ACTUATOR_TARGET_TRACKING_LAG`.
- Y+ trace-path audit JSON lines 20-35 show all traced envs are no-contact,
  final target z is `0.050000098533928394m` above start cube z, final TCP-cube
  distance is about `0.08266180194914341m`, and final error is still mostly z
  error.
- Y+ trace-path audit JSON lines 1027-1035 show the target still advances about
  `0.02000001072883606m` in world y.

Interpretation:

- Height050 rejects the shortcut "raise target height and go to data/RL." The
  target is easier to track in final TCP-error terms, but it stops creating cube
  contact.
- This is a stricter warning than seed943: lower TCP error is not sufficient if
  contact evidence collapses.
- The next narrow hypothesis should not be another height-only move. It should
  isolate fixed workspace x/y, small lateral offset, or actuator tracking while
  preserving reaction/contact/no-overshoot gates.
- Do not start 10cm 10240, dataset generation, PPO/RL, VLA, Track A, or broad
  random search from seed944.

Sources:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_height050_seed944_summary.json:47-49,69,95-103,112,119,123,142`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_height050_seed944_reaction_gate_audit.json:1-42`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_height050_seed944_trace_diagnostic_summary.json:97-100`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_height050_seed944_yplus_trace_path_actuator_audit.json:20-35,1027-1035`

## Y+ Good-Workspace Runtime Diagnostic

Purpose:

- Properly separate the three proposed next axes: fixed workspace x/y bucket,
  small lateral offset, and actuator tracking.
- Chosen axis: fixed workspace x/y. It is the cleanest first discriminator
  because the probe already exposes `--fixed_cube_x_m` and `--fixed_cube_y_m`,
  while generic lateral offset is not exposed for v1 fixed-y+ without code edits,
  and actuator changes are harder to interpret while contact itself is unstable.
- This was exactly one local GPU/IsaacLab tiny screen after approval. No B200, no
  SSH, no Track A, no dataset generation, no PPO/RL, no VLA, no 1024/10k.

Why this bucket:

- seed943 y+ geometry audit lines 2-23 showed contact rows centered around
  `cube_x0_m=0.2944993774096171`, `cube_y0_m=-0.043889790773391724`.
- The same audit lines 31-56 showed no-contact rows centered around
  `cube_x0_m=0.25363860130310056`, `cube_y0_m=0.05025864243507385`.
- Audit lines 58-122 showed contact was higher in `cube_y0_m<=0` and
  `cube_x0_m>=0.25` bins.
- Therefore seed945 fixed the cube near the contact-group mean:
  `x=0.295`, `y=-0.044`, with the same fixed y+ direction and original
  side-center height.

Runtime:

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab --no-capture-output python -u sim_scripts/cube3cm_push_diffik_probe.py \
  --num_envs 16 \
  --episodes 1 \
  --seed 945 \
  --cube_size_m 0.100 0.100 0.100 \
  --cube_mass_kg 0.720 \
  --fixed_cube_x_m 0.295 \
  --fixed_cube_y_m -0.044 \
  --fixed_push_dir 0 1 \
  --tcp_height_mode side_center \
  --tcp_center_height_offset_m 0.000 \
  --through_target_mode near_face \
  --contact_controller_mode measured_stop \
  --contact_stop_target_mode freeze \
  --max_diffik_joint_step_rad 0.035 \
  --trace_diffik_diagnostics \
  --trace_env_ids 0 1 2 3
```

Console summary:

```text
[cube3cm_push_diffik_probe] summary trials=16 controlled_push_rate=1.000000 impact_outlier_rate=0.000000 low_motion_rate=0.000000 disp_along_push_mean_m=0.009423 disp_ge_gate_rate=0.000000 disp_8_15mm_rate=1.000000 max_disp_along_push_mean_m=0.009829 max_disp_ge_gate_rate=0.187500 reaction_event_rate=1.000000 overshoot_ge_20mm_rate=0.000000 measured_contact_seen_rate=1.000000 contact_stop_seen_rate=0.187500 disp_xy_mean_m=0.009429 posewrite_calls_during_rollout=0 grasped_marker_rate=0.000000
```

Posthoc audits:

```bash
python sim_scripts/cube3cm_push_diffik_trace_diagnostic_audit.py \
  --trace_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_seed945_trace.csv \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_seed945_summary.json \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_seed945_trace_diagnostic_summary.json

python sim_scripts/cube10cm_reaction_event_gate_audit.py \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_seed945_summary.json \
  --trace_diag_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_seed945_trace_diagnostic_summary.json \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_seed945_reaction_gate_audit.json

python sim_scripts/cube10cm_yplus_geometry_reach_audit.py \
  --summary_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_seed945.csv \
  --trace_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_seed945_trace.csv \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_seed945_yplus_geometry_reach_audit.json

python sim_scripts/cube10cm_yplus_trace_path_actuator_audit.py \
  --summary_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_seed945.csv \
  --trace_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_seed945_trace.csv \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_seed945_yplus_trace_path_actuator_audit.json
```

Results:

- Summary JSON lines 47-53 show no dataset generation, clip `1.0`, final
  displacement `0.009423360228538513m`, and final 1cm gate `0.0`.
- Summary JSON lines 69-77 show final TCP error `0.0655147316865623m`, fixed
  cube x/y, and fixed y+.
- Summary JSON lines 95-103 show max speed `0.13854316715151072m/s`, max z delta
  `0.015773175051435828m`, max displacement `0.009829461574554443m`, transient
  1cm gate `0.1875`, and measured contact `1.0`.
- Summary JSON lines 112, 119, 123, and 142 show no posewrite, reaction `1.0`,
  rollout posewrite false, and 16 trials.
- Reaction audit JSON lines 2-3 and 17-28 show reaction `1.0`, contact evidence
  `1.0`, no posewrite, no overshoot, reaction gate true, final TCP error
  `0.0655147316865623m`, clip `1.0`, measured contact `1.0`, and
  `teacher_quality_ready=false`.
- Reaction audit JSON line 42 shows transient 1cm gate only `0.1875`.
- Y+ geometry audit JSON lines 2-33 show all 16 rows are contact rows with mean
  cube position `x=0.2950093150138855`, `y=-0.044006768614053726`, mean max
  displacement `0.009829461574554443m`, and final TCP error
  `0.0655147316865623m`.
- Y+ trace-path audit JSON lines 2-34 show all traced envs `[0,1,2,3]` contact,
  target world-y delta `0.02000001072883606m`, final z-error fraction
  `0.8575224903003924`, and `clip_any=1.0`.
- Trace diagnostic JSON lines 97-100 still show
  `LINK5_BODY_TARGET_NOT_REACHED`, `JOINT_STEP_CLIPPING_DOMINANT`, and
  `ACTUATOR_TARGET_TRACKING_LAG`.

Interpretation:

- Workspace x/y is now a real y+ contact discriminator. This was not just speed
  relabeling, missing y-target motion, or a height-only problem.
- Good workspace gives a reaction/contact PASS, but not a teacher-quality PASS.
  Final TCP error and DiffIK clipping are still too high, and final 1cm
  relocation remains false.
- The next tiny decision is sharper now: either map the boundary of this good
  workspace window, or test actuator tracking inside the good bucket. Do not
  start 10cm 10240, dataset generation, PPO/RL, VLA, Track A, or broad random
  search from seed945.

Sources:

- `sim_scripts/cube3cm_push_diffik_probe.py:43-45,268-276,497-520,657-686`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_reaction_seed943_yplus_geometry_reach_audit.json:1-122`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_seed945_summary.json:47-53,69-77,95-103,112,119,123,142`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_seed945_reaction_gate_audit.json:1-43`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_seed945_yplus_geometry_reach_audit.json:1-136`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_seed945_yplus_trace_path_actuator_audit.json:1-34`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_seed945_trace_diagnostic_summary.json:97-100`

## Y+ Good-Workspace Lateral Diagnostic

Purpose:

- Continue the step-by-step failure isolation after seed945:
  workspace was good enough for contact/reaction, but final TCP target error and
  clipping remained high.
- Inspect the traced final TCP-target offset before changing actuator settings.
  The seed945 traced envs showed final TCP x error around `+0.027m` to `+0.036m`
  and z error around `+0.055m` to `+0.059m`.
- Since seed944 proved a large positive height offset kills contact, this section
  tests lateral alignment first, not another height-only move.

Code change:

- Added default-preserving `--base_lateral_offset_m` to
  `sim_scripts/cube3cm_push_diffik_probe.py`.
- Code lines 51, 378, 439, and 1104 show the parser arg, console print, base
  trajectory application, and summary JSON logging.
- Default is `0.0`, so existing 3cm and 10cm runs are unchanged unless the new
  flag is passed.

Lateral sign:

- For fixed y+, the lateral direction is `(-1, 0)` from the code's
  `lateral_dir = (-push_y, push_x)`.
- seed945 final TCP was positive in x relative to the target, so the target must
  move toward positive x.
- Therefore the diagnostic uses negative lateral offset:
  `--base_lateral_offset_m -0.020`.

Runtime:

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab --no-capture-output python -u sim_scripts/cube3cm_push_diffik_probe.py \
  --num_envs 16 \
  --episodes 1 \
  --seed 946 \
  --cube_size_m 0.100 0.100 0.100 \
  --cube_mass_kg 0.720 \
  --fixed_cube_x_m 0.295 \
  --fixed_cube_y_m -0.044 \
  --fixed_push_dir 0 1 \
  --tcp_height_mode side_center \
  --tcp_center_height_offset_m 0.000 \
  --base_lateral_offset_m -0.020 \
  --through_target_mode near_face \
  --contact_controller_mode measured_stop \
  --contact_stop_target_mode freeze \
  --max_diffik_joint_step_rad 0.035 \
  --trace_diffik_diagnostics \
  --trace_env_ids 0 1 2 3
```

Console summary:

```text
[cube3cm_push_diffik_probe] summary trials=16 controlled_push_rate=1.000000 impact_outlier_rate=0.000000 low_motion_rate=0.000000 disp_along_push_mean_m=0.011250 disp_ge_gate_rate=1.000000 disp_8_15mm_rate=1.000000 max_disp_along_push_mean_m=0.011251 max_disp_ge_gate_rate=1.000000 reaction_event_rate=1.000000 overshoot_ge_20mm_rate=0.000000 measured_contact_seen_rate=1.000000 contact_stop_seen_rate=1.000000 disp_xy_mean_m=0.011266 posewrite_calls_during_rollout=0 grasped_marker_rate=0.000000
```

Posthoc audits:

```bash
python sim_scripts/cube3cm_push_diffik_trace_diagnostic_audit.py \
  --trace_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_seed946_trace.csv \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_seed946_summary.json \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_seed946_trace_diagnostic_summary.json

python sim_scripts/cube10cm_reaction_event_gate_audit.py \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_seed946_summary.json \
  --trace_diag_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_seed946_trace_diagnostic_summary.json \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_seed946_reaction_gate_audit.json

python sim_scripts/cube10cm_yplus_geometry_reach_audit.py \
  --summary_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_seed946.csv \
  --trace_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_seed946_trace.csv \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_seed946_yplus_geometry_reach_audit.json

python sim_scripts/cube10cm_yplus_trace_path_actuator_audit.py \
  --summary_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_seed946.csv \
  --trace_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_seed946_trace.csv \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_seed946_yplus_trace_path_actuator_audit.json
```

Results:

- Summary JSON lines 48-55 show no dataset generation, clip `1.0`, final
  displacement `0.011250250041484833m`, final 1cm gate `1.0`, and
  `disp_over_object_size_mean=0.11250250041484833`.
- Summary JSON lines 70-80 show final TCP error `0.06282096705399454m`, fixed
  good workspace x/y, and fixed y+.
- Summary JSON lines 96-105 show max speed `0.13885411759838462m/s`, max z delta
  `0.016476489370688796m`, max displacement `0.011251196265220642m`, max gate
  `1.0`, and measured contact `1.0`.
- Summary JSON lines 113, 120, 124, and 143 show no posewrite, reaction `1.0`,
  rollout posewrite false, and 16 trials.
- Reaction audit JSON lines 2-8 and 17-28 show reaction `1.0`, contact evidence
  `1.0`, final relocation pass true, no posewrite, no overshoot, reaction gate
  true, final TCP error `0.06282096705399454m`, clip `1.0`, measured contact
  `1.0`, and `teacher_quality_ready=false`.
- Reaction audit JSON line 42 shows transient 1cm gate `1.0`.
- Geometry audit JSON lines 2-33 show all 16 rows in the contact group, with mean
  max displacement `0.011251196265220642m`, final TCP error
  `0.06282096705399454m`, and no no-contact rows.
- Geometry audit JSON lines 123-128 show traced contact `n=4`, final xy error
  `0.021463414385781712m`, and final z error `0.05878029018640518m`.
- Trace-path audit JSON lines 2-34 show all traced envs `[0,1,2,3]` contact,
  final z-error fraction `0.9393462154426221`, and `clip_any=1.0`.
- Trace diagnostic JSON lines 97-100 still show
  `LINK5_BODY_TARGET_NOT_REACHED`, `JOINT_STEP_CLIPPING_DOMINANT`, and
  `ACTUATOR_TARGET_TRACKING_LAG`.

Interpretation:

- Lateral alignment was a real problem, and a small negative lateral offset fixed
  the object-level outcome in the good y+ workspace.
- This is the first fixed-y+ 10cm/0.72kg candidate in this session that passes
  reaction/contact and final 1cm relocation with no overshoot/posewrite.
- It is still not 10cm dataset/RL readiness because the controller is clipped
  every rollout and the link target is not tracked well enough under the current
  teacher-quality gate.
- The next useful step is not to generate 10240 samples. It is either a tiny
  robustness check of this exact candidate or an actuator/IK tracking cleanup
  inside this now-working geometry.

Sources:

- `sim_scripts/cube3cm_push_diffik_probe.py:51,378,439,1104`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_seed945_yplus_geometry_reach_audit.json:138-238`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_seed946_summary.json:48-55,70-80,96-105,113,120,124,143`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_seed946_reaction_gate_audit.json:1-43`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_seed946_yplus_geometry_reach_audit.json:1-33,123-128`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_seed946_yplus_trace_path_actuator_audit.json:1-34`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_seed946_trace_diagnostic_summary.json:97-100`
